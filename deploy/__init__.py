"""Deployable inference artefact for the snowy release.

Modules
-------
audit      checkpoint inspection that works with or without torch; refuses
           partial loads instead of leaving layers at random init.
contract   the POST /predict request/response schema, as data.
chain      one frame -> {qa_score, psg_raw, psg_smoothed, ground_mask, snow_mask}.
acceptance run the chain on annotated frames and diff against published numbers.
api        FastAPI app implementing the documented endpoint.
app        Gradio demo for a free-tier CPU Space.

Nothing in this package modifies or reimplements the research code: it imports
student/ground/snow modules and the feature extractors from the repo.
"""
