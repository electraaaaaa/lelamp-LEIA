"""
Voice pipeline abstraction layer.

Supports multiple voice backends:
- livekit-realtime: Realtime AI via LiveKit agents (OpenAI, Grok, Gemini, Azure, etc.)
- local: Faster Whisper + Ollama + Piper with direct ALSA
"""

from .base import PipelineBase, PipelineSession
from .livekit_realtime import entrypoint, PROVIDERS, DEFAULT_VOICES
from .local_pipeline import run_local_pipeline


def get_pipeline(config: dict) -> PipelineBase:
    """
    Factory function to get the configured pipeline.

    Args:
        config: Full application config dict

    Returns:
        Configured pipeline instance
    """
    pipeline_config = config.get("pipeline", {})
    pipeline_type = pipeline_config.get("type", "livekit-realtime")

    if pipeline_type == "local":
        from .local_pipeline import LocalPipeline
        return LocalPipeline(config)
    else:
        # Both "livekit" (legacy) and "livekit-realtime" use the realtime pipeline
        from .livekit_realtime import LiveKitRealtimePipeline
        return LiveKitRealtimePipeline(config)


__all__ = [
    "get_pipeline",
    "PipelineBase",
    "PipelineSession",
    "entrypoint",
    "run_local_pipeline",
    "PROVIDERS",
    "DEFAULT_VOICES",
]
