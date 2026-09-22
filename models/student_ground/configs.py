"""Model registry and hyperparameter defaults for student segmentation."""

MODEL_REGISTRY = {
    "dinov2_base": {
        "backbone": "facebook/dinov2-base",
        "hidden_dim": 768,
        "extract_layers": [3, 6, 9, 12],
        "input_size": 518,
        "patch_size": 14,
        "batch_size": 16,
        "lr": 1e-3,
    },
    "dinov2_large": {
        "backbone": "facebook/dinov2-large",
        "hidden_dim": 1024,
        "extract_layers": [6, 12, 18, 24],
        "input_size": 518,
        "patch_size": 14,
        "batch_size": 8,
        "lr": 5e-4,
    },
    "dinov3_base": {
        "backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "hidden_dim": 768,
        "extract_layers": [3, 6, 9, 12],
        "input_size": 512,
        "patch_size": 16,
        "batch_size": 16,
        "lr": 1e-3,
        "has_registers": True,
    },
    "dinov3_large": {
        "backbone": "facebook/dinov3-vitl16-pretrain-sat493m",
        "hidden_dim": 1024,
        "extract_layers": [6, 12, 18, 24],
        "input_size": 512,
        "patch_size": 16,
        "batch_size": 8,
        "lr": 5e-4,
        "has_registers": True,
    },
    "efficientnet": {
        "backbone": "efficientnet_b0",
        "input_size": 512,
        "batch_size": 32,
        "lr": 1e-3,
    },
    "mobilenet": {
        "backbone": "mobilenet_v3_small",
        "input_size": 512,
        "batch_size": 32,
        "lr": 1e-3,
    },
    "convnext": {
        "backbone": "convnext_tiny",
        "input_size": 512,
        "batch_size": 16,
        "lr": 1e-3,
    },
    "vit_lora": {
        "backbone": "vit_small_patch16_224",
        "hidden_dim": 384,
        "extract_layers": [3, 6, 9, 12],
        "input_size": 518,
        "patch_size": 16,
        "batch_size": 16,
        "lr": 5e-4,
    },
    "mobilenet_rgbd": {
        "backbone": "mobilenet_v3_small",
        "input_size": 512,
        "batch_size": 32,
        "lr": 1e-3,
        "use_depth": True,
    },
}

def get_config(variant):
    """Look up model config, handling '_rgbd' suffix transparently."""
    if variant in MODEL_REGISTRY:
        return MODEL_REGISTRY[variant]
    base = variant.removesuffix("_rgbd")
    if base in MODEL_REGISTRY:
        cfg = dict(MODEL_REGISTRY[base])
        cfg["use_depth"] = True
        return cfg
    raise KeyError(f"Unknown variant: {variant}. "
                   f"Choose from: {list(MODEL_REGISTRY.keys())} (+ '_rgbd' suffix)")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
