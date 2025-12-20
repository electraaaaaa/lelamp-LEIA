"""
Base classes for voice pipelines.

Provides abstract interfaces that both LiveKit and Local pipelines implement.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
import asyncio


class PipelineSession(ABC):
    """Abstract session for voice interaction."""

    @abstractmethod
    async def start(self):
        """Start the voice session."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the voice session."""
        pass

    @abstractmethod
    async def generate_reply(self, instructions: str):
        """Generate a spoken reply from instructions."""
        pass

    @abstractmethod
    def interrupt(self):
        """Interrupt current speech."""
        pass

    @abstractmethod
    def set_audio_enabled(self, enabled: bool):
        """Enable/disable audio input."""
        pass


class PipelineBase(ABC):
    """Abstract base class for voice pipelines."""

    def __init__(self, config: dict):
        self.config = config
        self.agent = None
        self.session: Optional[PipelineSession] = None
        self._on_user_speech_start: Optional[Callable] = None
        self._on_user_speech_end: Optional[Callable[[str], None]] = None
        self._on_agent_speech_start: Optional[Callable] = None
        self._on_agent_speech_end: Optional[Callable] = None
        self._on_user_transcription: Optional[Callable[[str], None]] = None

    @abstractmethod
    async def initialize(self, agent) -> PipelineSession:
        """
        Initialize pipeline with agent instance.

        Args:
            agent: LeLamp agent instance with function tools

        Returns:
            PipelineSession for controlling the voice interaction
        """
        pass

    @abstractmethod
    async def run(self):
        """Main pipeline loop (blocking)."""
        pass

    def on_user_speech_start(self, callback: Callable):
        """Register callback for when user starts speaking."""
        self._on_user_speech_start = callback

    def on_user_speech_end(self, callback: Callable[[str], None]):
        """Register callback for when user finishes speaking (with transcript)."""
        self._on_user_speech_end = callback

    def on_agent_speech_start(self, callback: Callable):
        """Register callback for when agent starts speaking."""
        self._on_agent_speech_start = callback

    def on_agent_speech_end(self, callback: Callable):
        """Register callback for when agent finishes speaking."""
        self._on_agent_speech_end = callback

    def on_user_transcription(self, callback: Callable[[str], None]):
        """Register callback for user transcription."""
        self._on_user_transcription = callback
