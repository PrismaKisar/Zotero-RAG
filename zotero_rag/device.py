"""Single source of truth for compute-device selection.

Every model-owning module used to auto-detect on its own, and they disagreed:
the reranker and the QA engine picked ``mps`` on Apple Silicon while the
embedding manager only ever looked at CUDA and silently settled for ``cpu``.
"""

import torch

SUPPORTED_DEVICES = ("cpu", "cuda", "mps")


def resolve_device(device: str | None = None) -> str:
    """Return the device to run on, auto-detecting when ``device`` is None.

    Args:
        device: Explicit device name, or None to auto-detect.

    Returns:
        One of ``SUPPORTED_DEVICES``.

    Raises:
        ValueError: If an explicit device name is not supported.
    """
    if device is not None:
        resolved = device.lower()
        if resolved not in SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported device '{device}'; expected one of {SUPPORTED_DEVICES}")
        return resolved

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
