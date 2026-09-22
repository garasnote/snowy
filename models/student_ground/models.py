"""Student segmentation model architectures.

Variations:
  1. DINOv2-Base + L-Decoder  (SegDINO-style, ~1.1M trainable params)
  2. DINOv2-Large + L-Decoder (~1.3M trainable params)
  3. DINOv3-Base + L-Decoder  (register tokens, 16×16 patches)
  4. DINOv3-Large + L-Decoder (satellite pretrained, 16×16 patches)
  5. EfficientNet-B0 + UNet Decoder (CNN, ~2M total)
  6. MobileNetV3-Small + UNet Decoder (CNN baseline, ~0.5M trainable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import MODEL_REGISTRY


# ── DINOv2 + L-Decoder (SegDINO-style) ──────────────────────────────────────

class LDecoder(nn.Module):
    """Lightweight MLP decoder for multi-level DINOv2 features.

    Takes features from 4 transformer layers, aligns channels,
    concatenates, fuses via 1x1 convolutions, and predicts a
    single-channel segmentation map.
    """

    def __init__(self, hidden_dim=768, proj_dim=256, num_layers=4):
        super().__init__()
        # Per-layer channel alignment
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])

        # Fusion after concatenation
        self.fusion = nn.Sequential(
            nn.Conv2d(proj_dim * num_layers, proj_dim, 1),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim, 1),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )

        self.head = nn.Conv2d(proj_dim, 1, 1)

    def forward(self, multi_level_features, grid_h, grid_w):
        """
        Args:
            multi_level_features: list of (B, N_patches, hidden_dim) tensors
            grid_h, grid_w: spatial dimensions of the patch grid

        Returns:
            logits: (B, 1, grid_h, grid_w) — upsample to full res externally
        """
        projected = []
        for feat, proj in zip(multi_level_features, self.projections):
            # feat: (B, N_patches, hidden_dim)
            x = proj(feat)  # (B, N_patches, proj_dim)
            B, N, C = x.shape
            x = x.reshape(B, grid_h, grid_w, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            projected.append(x)

        # Concatenate along channel dim
        fused = torch.cat(projected, dim=1)  # (B, proj_dim*4, H, W)
        fused = self.fusion(fused)           # (B, proj_dim, H, W)
        logits = self.head(fused)            # (B, 1, H, W)
        return logits


class DINOv2Student(nn.Module):
    """Frozen DINOv2 backbone + trainable L-Decoder for binary segmentation."""

    def __init__(self, variant="dinov2_base", use_depth=False):
        super().__init__()
        cfg = MODEL_REGISTRY[variant]
        self.variant = variant
        self.input_size = cfg["input_size"]
        self.patch_size = cfg["patch_size"]
        self.extract_layers = cfg["extract_layers"]
        self.use_depth = use_depth

        # Load backbone (frozen)
        from transformers import AutoModel, AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(cfg["backbone"])
        self.backbone = AutoModel.from_pretrained(cfg["backbone"])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Grid dimensions
        self.grid_h = self.input_size // self.patch_size
        self.grid_w = self.input_size // self.patch_size
        hidden_dim = cfg["hidden_dim"]

        # Depth encoder: maps 1-channel depth → (grid_h, grid_w, hidden_dim)
        # tokens that get added to the first extracted feature level.
        if use_depth:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 64, 7, stride=4, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((self.grid_h, self.grid_w)),
            )

        # Trainable decoder
        self.decoder = LDecoder(
            hidden_dim=hidden_dim,
            proj_dim=256,
            num_layers=len(cfg["extract_layers"]),
        )

    @torch.no_grad()
    def extract_features(self, pixel_values):
        """Extract multi-level features from frozen backbone."""
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, 1+N_patches, hidden_dim)

        features = []
        for layer_idx in self.extract_layers:
            h = hidden_states[layer_idx]
            h = h[:, 1:, :]  # remove CLS token → (B, N_patches, hidden_dim)
            features.append(h)
        return features

    def forward(self, pixel_values, output_size=None):
        """
        Args:
            pixel_values: (B, 3, H, W) or (B, 4, H, W) if use_depth
            output_size: (H, W) to upsample logits to (default: input_size)

        Returns:
            logits: (B, 1, out_H, out_W)
        """
        if self.use_depth and pixel_values.shape[1] == 4:
            rgb = pixel_values[:, :3, :, :]
            depth = pixel_values[:, 3:4, :, :]
        else:
            rgb = pixel_values
            depth = None

        features = self.extract_features(rgb)

        # Fuse depth into first feature level
        if depth is not None:
            depth_feat = self.depth_encoder(depth)  # (B, hidden_dim, grid_h, grid_w)
            B, C, H, W = depth_feat.shape
            depth_tokens = depth_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            features[0] = features[0] + depth_tokens

        logits = self.decoder(features, self.grid_h, self.grid_w)

        # Upsample to requested output size
        if output_size is None:
            output_size = (self.input_size, self.input_size)
        if logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size,
                                   mode="bilinear", align_corners=False)
        return logits

    def trainable_parameters(self):
        """Return only the decoder + depth encoder parameters for the optimizer."""
        params = list(self.decoder.parameters())
        if self.use_depth:
            params += list(self.depth_encoder.parameters())
        return params


# ── DINOv3 + L-Decoder (register tokens) ────────────────────────────────────

class DINOv3Student(nn.Module):
    """Frozen DINOv3 backbone + trainable L-Decoder for binary segmentation.

    DINOv3 differs from DINOv2 in its token layout:
        [CLS, REG1, ..., REGn, PATCH1, ..., PATCHn]
    We extract patch tokens by taking the last grid_h*grid_w tokens.
    """

    def __init__(self, variant="dinov3_base", use_depth=False):
        super().__init__()
        cfg = MODEL_REGISTRY[variant]
        self.variant = variant
        self.input_size = cfg["input_size"]
        self.patch_size = cfg["patch_size"]
        self.extract_layers = cfg["extract_layers"]
        self.use_depth = use_depth

        # Load backbone (frozen)
        from transformers import AutoModel, AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(cfg["backbone"])
        self.backbone = AutoModel.from_pretrained(cfg["backbone"])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Grid dimensions
        self.grid_h = self.input_size // self.patch_size
        self.grid_w = self.input_size // self.patch_size
        self.n_patch_tokens = self.grid_h * self.grid_w
        hidden_dim = cfg["hidden_dim"]

        # Depth encoder: same approach as DINOv2Student
        if use_depth:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 64, 7, stride=4, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((self.grid_h, self.grid_w)),
            )

        # Trainable decoder
        self.decoder = LDecoder(
            hidden_dim=hidden_dim,
            proj_dim=256,
            num_layers=len(cfg["extract_layers"]),
        )

    @torch.no_grad()
    def extract_features(self, pixel_values):
        """Extract multi-level features from frozen backbone."""
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        features = []
        for layer_idx in self.extract_layers:
            h = hidden_states[layer_idx]
            # Take last n_patch_tokens (skip CLS + register tokens)
            h = h[:, -self.n_patch_tokens:, :]
            features.append(h)
        return features

    def forward(self, pixel_values, output_size=None):
        if self.use_depth and pixel_values.shape[1] == 4:
            rgb = pixel_values[:, :3, :, :]
            depth = pixel_values[:, 3:4, :, :]
        else:
            rgb = pixel_values
            depth = None

        features = self.extract_features(rgb)

        if depth is not None:
            depth_feat = self.depth_encoder(depth)
            B, C, H, W = depth_feat.shape
            depth_tokens = depth_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            features[0] = features[0] + depth_tokens

        logits = self.decoder(features, self.grid_h, self.grid_w)

        if output_size is None:
            output_size = (self.input_size, self.input_size)
        if logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size,
                                   mode="bilinear", align_corners=False)
        return logits

    def trainable_parameters(self):
        params = list(self.decoder.parameters())
        if self.use_depth:
            params += list(self.depth_encoder.parameters())
        return params


# ── MobileNetV3 + UNet Decoder ──────────────────────────────────────────────

class UNetDecoder(nn.Module):
    """Simple UNet-style decoder with skip connections."""

    def __init__(self, encoder_channels, decoder_channels=(128, 64, 32, 16)):
        """
        Args:
            encoder_channels: list of channel counts from encoder stages (low→high res)
                              e.g. [96, 48, 24, 16] for MobileNetV3-Small
            decoder_channels: output channels for each decoder stage
        """
        super().__init__()
        # Reverse encoder channels (decode from deepest to shallowest)
        enc_ch = list(reversed(encoder_channels))
        dec_ch = list(decoder_channels)

        self.up_blocks = nn.ModuleList()
        in_ch = enc_ch[0]
        for i, out_ch in enumerate(dec_ch):
            skip_ch = enc_ch[i + 1] if i + 1 < len(enc_ch) else 0
            self.up_blocks.append(
                UpBlock(in_ch + skip_ch, out_ch)
            )
            in_ch = out_ch

        self.head = nn.Conv2d(dec_ch[-1], 1, 1)

    def forward(self, features, output_size=None):
        """
        Args:
            features: list of (B, C, H, W) from encoder, shallow→deep order
            output_size: optional (H, W) to resize final output

        Returns:
            logits: (B, 1, H, W)
        """
        feats = list(reversed(features))
        x = feats[0]

        for i, block in enumerate(self.up_blocks):
            if i + 1 < len(feats):
                skip = feats[i + 1]
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
            else:
                x = F.interpolate(x, scale_factor=2,
                                  mode="bilinear", align_corners=False)
            x = block(x)

        x = self.head(x)
        if output_size is not None and x.shape[2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MobileNetStudent(nn.Module):
    """MobileNetV3-Small encoder + UNet decoder for binary segmentation."""

    # Feature extraction points in MobileNetV3-Small.features:
    #   After block 0  (idx 0):  stride 2,  16 ch
    #   After block 1  (idx 1):  stride 4,  16 ch
    #   After block 3  (idx 3):  stride 8,  24 ch
    #   After block 8  (idx 8):  stride 16, 48 ch
    #   After block 11 (idx 11): stride 16, 96 ch  (but effective deepest)
    EXTRACT_INDICES = [1, 3, 8, 11]
    ENCODER_CHANNELS = [16, 24, 48, 96]

    def __init__(self, freeze_encoder=True, use_depth=False):
        super().__init__()
        self.input_size = MODEL_REGISTRY["mobilenet"]["input_size"]
        self.use_depth = use_depth

        import torchvision.models as models
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.encoder = backbone.features

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.freeze_encoder = freeze_encoder

        # Depth adapter: maps 1-channel depth → 16-channel features that are
        # added to the output of encoder block 0 (first skip connection).
        # This keeps the pretrained RGB pathway intact.
        if use_depth:
            self.depth_adapter = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),  # match block 0 stride
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            )

        self.decoder = UNetDecoder(
            encoder_channels=self.ENCODER_CHANNELS,
            decoder_channels=(128, 64, 32, 16),
        )

    def forward(self, pixel_values, output_size=None):
        """
        Args:
            pixel_values: (B, 3, H, W) or (B, 4, H, W) if use_depth
            output_size: target (H, W) for the output

        Returns:
            logits: (B, 1, H, W)
        """
        if self.use_depth and pixel_values.shape[1] == 4:
            rgb = pixel_values[:, :3, :, :]
            depth = pixel_values[:, 3:4, :, :]
        else:
            rgb = pixel_values
            depth = None

        features = self._extract_features(rgb, depth)
        logits = self.decoder(features)

        if output_size is None:
            output_size = (self.input_size, self.input_size)
        if logits.shape[2:] != tuple(output_size):
            logits = F.interpolate(logits, size=output_size,
                                   mode="bilinear", align_corners=False)
        return logits

    def _extract_features(self, x, depth=None):
        """Run encoder and capture intermediate features."""
        if self.freeze_encoder:
            with torch.no_grad():
                features = self._forward_encoder(x)
        else:
            features = self._forward_encoder(x)

        # Fuse depth into the first feature map (block 0 output, 16 ch)
        if depth is not None and self.use_depth:
            depth_feat = self.depth_adapter(depth)
            if depth_feat.shape[2:] != features[0].shape[2:]:
                depth_feat = F.interpolate(
                    depth_feat, size=features[0].shape[2:],
                    mode="bilinear", align_corners=False)
            features[0] = features[0] + depth_feat

        return features

    def _forward_encoder(self, x):
        features = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in self.EXTRACT_INDICES:
                features.append(x)
        return features

    def trainable_parameters(self):
        """Return only decoder parameters + depth adapter (if encoder is frozen)."""
        if self.freeze_encoder:
            params = list(self.decoder.parameters())
            if self.use_depth:
                params += list(self.depth_adapter.parameters())
            return params
        return self.parameters()


# ── EfficientNet-B0 + UNet Decoder ───────────────────────────────────────────

class EfficientNetStudent(nn.Module):
    """EfficientNet-B0 encoder + UNet decoder for binary segmentation.

    Lightweight CNN alternative — suitable for CPU inference.
    """

    # Feature extraction points in EfficientNet-B0.features:
    #   After block 1 (idx 1): stride 2,  16 ch
    #   After block 2 (idx 2): stride 4,  24 ch
    #   After block 3 (idx 3): stride 8,  40 ch
    #   After block 5 (idx 5): stride 16, 112 ch
    EXTRACT_INDICES = [1, 2, 3, 5]
    ENCODER_CHANNELS = [16, 24, 40, 112]

    def __init__(self, freeze_encoder=True, use_depth=False):
        super().__init__()
        self.input_size = MODEL_REGISTRY["efficientnet"]["input_size"]
        self.use_depth = use_depth

        import torchvision.models as models
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.encoder = backbone.features

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.freeze_encoder = freeze_encoder

        # Depth adapter: same approach as MobileNetStudent
        if use_depth:
            self.depth_adapter = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            )

        self.decoder = UNetDecoder(
            encoder_channels=self.ENCODER_CHANNELS,
            decoder_channels=(128, 64, 32, 16),
        )

    def forward(self, pixel_values, output_size=None):
        if self.use_depth and pixel_values.shape[1] == 4:
            rgb = pixel_values[:, :3, :, :]
            depth = pixel_values[:, 3:4, :, :]
        else:
            rgb = pixel_values
            depth = None

        features = self._extract_features(rgb, depth)
        logits = self.decoder(features)

        if output_size is None:
            output_size = (self.input_size, self.input_size)
        if logits.shape[2:] != tuple(output_size):
            logits = F.interpolate(logits, size=output_size,
                                   mode="bilinear", align_corners=False)
        return logits

    def _extract_features(self, x, depth=None):
        if self.freeze_encoder:
            with torch.no_grad():
                features = self._forward_encoder(x)
        else:
            features = self._forward_encoder(x)

        # Fuse depth into first feature map (16 ch, same as MobileNet)
        if depth is not None and self.use_depth:
            depth_feat = self.depth_adapter(depth)
            if depth_feat.shape[2:] != features[0].shape[2:]:
                depth_feat = F.interpolate(
                    depth_feat, size=features[0].shape[2:],
                    mode="bilinear", align_corners=False)
            features[0] = features[0] + depth_feat

        return features

    def _forward_encoder(self, x):
        features = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in self.EXTRACT_INDICES:
                features.append(x)
        return features

    def trainable_parameters(self):
        if self.freeze_encoder:
            params = list(self.decoder.parameters())
            if self.use_depth:
                params += list(self.depth_adapter.parameters())
            return params
        return self.parameters()


# ── ConvNeXt-Tiny + UNet Decoder ─────────────────────────────────────────────

class ConvNeXtStudent(nn.Module):
    """ConvNeXt-Tiny encoder (frozen) + UNet decoder for binary segmentation."""

    ENCODER_CHANNELS = [96, 192, 384, 768]

    def __init__(self, freeze_encoder=True, use_depth=False):
        super().__init__()
        self.input_size = MODEL_REGISTRY.get("convnext", {}).get("input_size", 512)
        self.use_depth = use_depth

        import torchvision.models as models
        backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
        self.encoder = backbone.features  # Sequential of 8 stages

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.freeze_encoder = freeze_encoder

        if use_depth:
            self.depth_adapter = nn.Sequential(
                nn.Conv2d(1, 96, 4, stride=4, padding=0),
                nn.BatchNorm2d(96),
                nn.GELU(),
            )

        self.decoder = UNetDecoder(
            encoder_channels=self.ENCODER_CHANNELS,
            decoder_channels=(256, 128, 64, 32),
        )

    def forward(self, pixel_values, output_size=None):
        if self.use_depth:
            rgb = pixel_values[:, :3]
            depth = pixel_values[:, 3:4]
        else:
            rgb = pixel_values

        # ConvNeXt features: stages at [0,1], [2,3], [4,5], [6,7]
        skips = []
        x = rgb
        stage_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        for start, end in stage_pairs:
            for idx in range(start, end + 1):
                x = self.encoder[idx](x)
            if self.use_depth and start == 0:
                depth_feat = self.depth_adapter(depth)
                if depth_feat.shape[2:] != x.shape[2:]:
                    depth_feat = F.interpolate(depth_feat, size=x.shape[2:], mode="bilinear", align_corners=False)
                x = x + depth_feat
            skips.append(x)

        if output_size is None:
            output_size = pixel_values.shape[2:]
        return self.decoder(skips, output_size)

    def trainable_parameters(self):
        if self.freeze_encoder:
            params = list(self.decoder.parameters())
            if self.use_depth:
                params += list(self.depth_adapter.parameters())
            return params
        return self.parameters()


# ── ViT-Small + LoRA + LDecoder ──────────────────────────────────────────────

class ViTLoRAStudent(nn.Module):
    """ViT-Small/16 with LoRA adapters (PEFT) + L-Decoder.

    Encoder stays frozen except for LoRA rank-r adapters on QKV projections.
    """

    def __init__(self, lora_rank=8, use_depth=False):
        super().__init__()
        self.use_depth = use_depth
        cfg = MODEL_REGISTRY.get("vit_lora", {})
        self.input_size = cfg.get("input_size", 518)
        self.patch_size = 16
        self.embed_dim = 384
        self.extract_layers = cfg.get("extract_layers", [3, 6, 9, 12])

        import timm
        self.encoder = timm.create_model("vit_small_patch16_224", pretrained=True,
                                         img_size=self.input_size)
        # Freeze all encoder params
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Replace each block's attn.qkv with LoRA-wrapped version
        self.lora_layers = nn.ModuleList()
        for block in self.encoder.blocks:
            lora_qkv = LoRALinear(block.attn.qkv, rank=lora_rank)
            block.attn.qkv = lora_qkv
            self.lora_layers.append(lora_qkv)

        self._features = {}
        for idx in self.extract_layers:
            self.encoder.blocks[idx - 1].register_forward_hook(
                self._make_hook(idx)
            )

        n_layers = len(self.extract_layers)
        self.decoder = LDecoder(
            hidden_dim=self.embed_dim,
            num_layers=n_layers,
        )

        if use_depth:
            self.depth_adapter = nn.Sequential(
                nn.Conv2d(1, self.embed_dim, self.patch_size,
                          stride=self.patch_size),
                nn.LayerNorm([self.embed_dim]),
            )

    def _make_hook(self, idx):
        def hook(module, input, output):
            self._features[idx] = output
        return hook

    def forward(self, pixel_values, output_size=None):
        if self.use_depth:
            rgb = pixel_values[:, :3]
        else:
            rgb = pixel_values

        self._features.clear()
        _ = self.encoder.forward_features(rgb)

        h = w = self.input_size // self.patch_size
        multi_layer = []
        for idx in self.extract_layers:
            feat = self._features[idx]
            # Remove CLS token if present
            if feat.shape[1] == h * w + 1:
                feat = feat[:, 1:, :]
            feat = feat[:, :h * w, :]
            multi_layer.append(feat)

        if output_size is None:
            output_size = pixel_values.shape[2:]
        logits = self.decoder(multi_layer, h, w)
        if logits.shape[2:] != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

    def trainable_parameters(self):
        params = list(self.decoder.parameters())
        for lora in self.lora_layers:
            params += list(lora.down.parameters()) + list(lora.up.parameters())
        if self.use_depth:
            params += list(self.depth_adapter.parameters())
        return params


class LoRALinear(nn.Module):
    """Drop-in replacement for a frozen Linear that adds a trainable low-rank path.

    output = frozen_linear(x) + up(down(x)) * scaling
    """

    def __init__(self, frozen_linear: nn.Linear, rank=8, alpha=16):
        super().__init__()
        in_dim = frozen_linear.in_features
        out_dim = frozen_linear.out_features
        self.frozen = frozen_linear
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.frozen(x) + self.up(self.down(x)) * self.scaling


# ── Factory ──────────────────────────────────────────────────────────────────

def build_model(variant="dinov2_base"):
    """Build a student segmentation model by variant name.

    Supports '_rgbd' suffix on any variant to enable depth fusion, e.g.
    'dinov2_base_rgbd', 'dinov3_large_rgbd', 'efficientnet_rgbd'.

    Args:
        variant: model name, optionally with '_rgbd' suffix

    Returns:
        model: nn.Module with .trainable_parameters() method
    """
    # Parse RGBD suffix
    use_depth = variant.endswith("_rgbd")
    base_variant = variant.removesuffix("_rgbd") if use_depth else variant

    if base_variant in ("dinov2_base", "dinov2_large"):
        return DINOv2Student(variant=base_variant, use_depth=use_depth)
    elif base_variant in ("dinov3_base", "dinov3_large"):
        return DINOv3Student(variant=base_variant, use_depth=use_depth)
    elif base_variant == "efficientnet":
        return EfficientNetStudent(freeze_encoder=True, use_depth=use_depth)
    elif base_variant == "mobilenet":
        return MobileNetStudent(freeze_encoder=True, use_depth=use_depth)
    elif base_variant == "convnext":
        return ConvNeXtStudent(freeze_encoder=True, use_depth=use_depth)
    elif base_variant == "vit_lora":
        return ViTLoRAStudent(lora_rank=8, use_depth=use_depth)
    else:
        raise ValueError(f"Unknown variant: {variant}. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())} "
                         f"(append '_rgbd' for depth fusion)")
