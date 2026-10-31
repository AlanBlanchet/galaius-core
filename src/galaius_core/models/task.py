"""What a model does: one vocabulary for every model, hosted or on a machine."""

from typing import Literal

#: What a model does, named by the Hugging Face Hub `pipeline_tag` ids (huggingface.js
#: `PIPELINE_DATA`) — an existing cross-vendor vocabulary, so a hosted API model and a local
#: checkpoint doing the same job share one task; the PORTS it exposes may still differ by
#: `Placement.target` (`model_task_ports`'s `placement` argument) — a machine-local detector reads
#: image PATHS already on that machine (never uploaded), a hosted one is handed the real file.
ModelTask = Literal[
    "text-generation", "text-to-image", "text-to-video", "object-detection", "image-segmentation",
    "feature-extraction", "text-to-speech", "automatic-speech-recognition", "image-to-3d", "text-to-3d",
    "depth-estimation", "keypoint-detection", "image-to-text", "text-to-music", "text-to-sound",
]
