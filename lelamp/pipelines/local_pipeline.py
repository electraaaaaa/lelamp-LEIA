"""
Local voice pipeline using Faster Whisper + Ollama + Piper.

Uses direct ALSA audio I/O for voice processing, but can optionally
stream video to LiveKit for data collection.
"""

import asyncio
import logging
import os
import time
from typing import Callable, Optional
from pathlib import Path

import lelamp.globals as g
from lelamp.agent import LeLamp
from lelamp.service.callbacks import init_callback_service
from lelamp.user_data import USER_DATA_DIR, get_device_serial_short

from .base import PipelineBase, PipelineSession
from .tool_registry import extract_tools_from_agent
from lelamp.service.local_voice import (
    LocalAudioIO,
    LocalSTTService,
    LocalLLMService,
    LocalTTSService,
)
from lelamp.service.local_voice.llm_service import split_sentences, clean_response

# Optional LiveKit for video streaming
try:
    from livekit import rtc, api
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    rtc = None
    api = None

logger = logging.getLogger(__name__)

# Import metrics service
try:
    from lelamp.service.metrics_service import PipelineStage

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("Metrics service not available")


def get_metrics_service():
    """Get metrics service from globals to ensure same instance as WebSocket."""
    return g.metrics_service


def _get_env_value(key: str) -> Optional[str]:
    """Get a value from .env file."""
    env_path = USER_DATA_DIR / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip() == key:
                        return v.strip()
        except Exception:
            pass
    return None


class LiveKitVideoStreamer:
    """
    Handles LiveKit room connection for video streaming only.

    Used by local pipeline to stream video to LiveKit for data collection
    while using local ALSA audio for voice processing.
    """

    def __init__(self, config: dict):
        self.config = config
        self.room: Optional["rtc.Room"] = None
        self._connected = False
        self._room_name = f"lelamp-{get_device_serial_short()}"

        # Load credentials
        self.url = os.getenv("LIVEKIT_URL", "").strip() or _get_env_value("LIVEKIT_URL") or ""
        self.api_key = os.getenv("LIVEKIT_API_KEY", "").strip() or _get_env_value("LIVEKIT_API_KEY") or ""
        self.api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip() or _get_env_value("LIVEKIT_API_SECRET") or ""

    @property
    def is_configured(self) -> bool:
        """Check if LiveKit credentials are configured."""
        return bool(self.url and self.api_key and self.api_secret)

    @property
    def is_connected(self) -> bool:
        """Check if connected to LiveKit room."""
        return self._connected and self.room is not None

    def _create_token(self, identity: str) -> str:
        """Create a LiveKit access token for room connection."""
        if not api:
            raise RuntimeError("LiveKit API not available")

        token = api.AccessToken(self.api_key, self.api_secret)
        token.identity = identity
        token.name = f"LeLamp Local ({identity})"

        # Grant permissions for video publishing only
        token.add_grant(api.VideoGrant(
            room=self._room_name,
            room_join=True,
            can_publish=True,
            can_publish_sources=["camera"],
            can_subscribe=False,  # We don't need to receive anything
        ))

        return token.to_jwt()

    async def connect(self) -> bool:
        """Connect to LiveKit room for video streaming."""
        if not LIVEKIT_AVAILABLE:
            logger.warning("LiveKit not available, video streaming disabled")
            return False

        if not self.is_configured:
            logger.warning("LiveKit not configured, video streaming disabled")
            return False

        try:
            # Create room
            self.room = rtc.Room()

            # Create access token
            identity = f"local_{get_device_serial_short()}"
            token = self._create_token(identity)

            logger.info(f"Connecting to LiveKit room: {self._room_name}")

            # Connect to room
            await self.room.connect(self.url, token)

            self._connected = True
            g.livekit_room = self.room  # Store in globals for VisionService
            logger.info(f"✓ Connected to LiveKit room for video streaming: {self._room_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to LiveKit: {e}", exc_info=True)
            self._connected = False
            self.room = None
            return False

    async def disconnect(self):
        """Disconnect from LiveKit room."""
        if self.room:
            try:
                await self.room.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from LiveKit: {e}")
            finally:
                self._connected = False
                self.room = None
                g.livekit_room = None
                logger.info("Disconnected from LiveKit room")


class LocalSession(PipelineSession):
    """Local voice session implementation."""

    def __init__(self, pipeline: "LocalPipeline"):
        self.pipeline = pipeline
        self._running = False
        self._voice_task: Optional[asyncio.Task] = None

    async def start(self, start_voice_loop: bool = True):
        """Start the voice session.

        Args:
            start_voice_loop: If True (default), start the voice loop immediately.
                             Set to False to delay voice loop start (e.g., for greeting playback).
        """
        print("[LOCAL SESSION] start() called")
        self._running = True

        # Start services
        print("[LOCAL SESSION] Starting TTS...")
        await self.pipeline.tts.start()
        print("[LOCAL SESSION] Starting audio I/O...")
        self.pipeline.audio_io.start()

        # Warm up LLM
        print("[LOCAL SESSION] Warming up LLM...")
        await self.pipeline.llm.warm_up()
        print("[LOCAL SESSION] LLM warmed up")

        if start_voice_loop:
            # Start voice loop
            self.start_voice_loop()

    def start_voice_loop(self):
        """Start the voice loop task."""
        if self._voice_task is None or self._voice_task.done():
            print("[LOCAL SESSION] Creating voice loop task...")
            self._voice_task = asyncio.create_task(self._voice_loop())
            logger.info("Local voice session started")
            print("[LOCAL SESSION] Voice loop task created!")

    async def stop(self):
        """Stop the voice session."""
        self._running = False
        if self._voice_task:
            self._voice_task.cancel()
            try:
                await self._voice_task
            except asyncio.CancelledError:
                pass

        await self.pipeline.tts.stop()
        self.pipeline.audio_io.stop()
        logger.info("Local voice session stopped")

    async def generate_reply(self, instructions: str):
        """Generate and speak a reply from instructions."""
        await self.pipeline._speak_response(instructions)

    def interrupt(self):
        """Interrupt current speech."""
        # Clear output queue to stop playback
        while not self.pipeline.audio_io.output_queue.empty():
            try:
                self.pipeline.audio_io.output_queue.get_nowait()
            except:
                break
        # Unmute mic
        self.pipeline.audio_io.mute_mic(False)

    def set_audio_enabled(self, enabled: bool):
        """Enable/disable audio input."""
        self.pipeline.audio_io.mute_mic(not enabled)

    async def _voice_loop(self):
        """Main voice interaction loop."""
        metrics = get_metrics_service() if METRICS_AVAILABLE else None

        logger.info(f"Voice loop started - METRICS_AVAILABLE={METRICS_AVAILABLE}, metrics={metrics is not None}")
        print(f"[LOCAL PIPELINE] Voice loop started - metrics available: {metrics is not None}")

        # Set initial state to listening
        if metrics:
            metrics.set_agent_state("listening")
            logger.info("Set initial state to 'listening'")
            print("[LOCAL PIPELINE] Set state to 'listening'")
        else:
            print("[LOCAL PIPELINE] WARNING: No metrics service - state will not be tracked!")

        import sys
        chunk_count = 0
        none_count = 0
        print("[VOICE LOOP] Entering main loop", flush=True)
        while self._running:
            try:
                # We're in listening state while waiting for audio
                if metrics:
                    metrics.set_agent_state("listening")

                # Get audio chunk from microphone (blocking with timeout)
                audio_chunk = await asyncio.to_thread(
                    self.pipeline.audio_io.get_audio_chunk, 0.1
                )

                if audio_chunk is None:
                    none_count += 1
                    if none_count % 50 == 0:
                        print(f"[VOICE LOOP] Got {none_count} None chunks (no audio)", flush=True)
                    continue

                chunk_count += 1
                if chunk_count % 50 == 0:
                    print(f"[VOICE LOOP] Received {chunk_count} chunks, last chunk size: {len(audio_chunk)}", flush=True)

                # Process with STT (includes VAD)
                result = self.pipeline.stt.process_audio_chunk(audio_chunk)

                if result:
                    text, stt_time, audio_duration = result

                    if not text:
                        continue

                    # User speech detected
                    if metrics:
                        metrics.set_agent_state("user_speaking")
                        metrics.set_user_speaking(True)

                    logger.info(f"User said: {text}")

                    # Start metrics tracking
                    if metrics:
                        metrics.start_turn()
                        metrics.record_timestamp(PipelineStage.STT_END)
                        metrics.set_user_text(text)

                    # Trigger user speech callbacks
                    if self.pipeline._on_user_speech_start:
                        self.pipeline._on_user_speech_start()
                    if self.pipeline._on_user_speech_end:
                        self.pipeline._on_user_speech_end(text)
                    if self.pipeline._on_user_transcription:
                        self.pipeline._on_user_transcription(text)

                    # Mark activity on agent
                    if hasattr(self.pipeline.agent, "_mark_activity"):
                        self.pipeline.agent._mark_activity()

                    # User done speaking - transition to thinking
                    if metrics:
                        metrics.set_user_speaking(False)
                        metrics.set_agent_state("thinking")

                    # Mute mic during response
                    self.pipeline.audio_io.mute_mic(True)

                    # Generate and speak response
                    await self.pipeline._process_user_input(text, metrics)

                    # Unmute mic and reset STT state to prevent false triggers
                    self.pipeline.stt.reset()
                    self.pipeline.audio_io.clear_input_queue()
                    self.pipeline.audio_io.mute_mic(False)

                    # End metrics turn and go back to listening
                    if metrics:
                        metrics.end_turn()
                        metrics.set_agent_state("listening")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in voice loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("Voice loop ended")


class LocalPipeline(PipelineBase):
    """Pipeline using local STT/LLM/TTS with direct ALSA audio.

    Optionally streams video to LiveKit for data collection while
    using local ALSA audio for voice processing.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        local_config = config.get("pipeline", {}).get("local", {})

        # Get LELAMP_DIR for resolving relative paths
        lelamp_dir = Path(os.environ.get("LELAMP_DIR", Path.home() / "lelampv2"))

        # Expand paths - resolve relative paths against LELAMP_DIR
        piper_path = local_config.get("piper_path")
        if piper_path:
            p = Path(piper_path).expanduser()
            if not p.is_absolute():
                p = lelamp_dir / p
            piper_path = str(p)

        voices_dir = local_config.get("voices_dir")
        if voices_dir:
            p = Path(voices_dir).expanduser()
            if not p.is_absolute():
                p = lelamp_dir / p
            voices_dir = str(p)

        # Initialize services
        self.audio_io = LocalAudioIO()

        self.stt = LocalSTTService(
            model_size=local_config.get("whisper_model", "tiny"),
            compute_type=local_config.get("whisper_compute", "int8"),
            silence_threshold=local_config.get("silence_threshold"),
            silence_duration=local_config.get("silence_duration"),
            min_audio_length=local_config.get("min_audio_length"),
        )

        self.llm = LocalLLMService(
            model=local_config.get("ollama_model", "llama3.2:3b"),
            ollama_url=local_config.get("ollama_url", "http://localhost:11434"),
            history_length=local_config.get("history_length", 10),
        )

        self.tts = LocalTTSService(
            piper_path=piper_path,
            voices_dir=voices_dir,
            voice=local_config.get("voice", "ryan-medium.onnx"),
        )

        # TTS streaming config
        self.flush_interval = local_config.get("flush_interval", 0.2)
        self.max_response_length = local_config.get("max_response_length", 75)
        self.min_batch_sentences = local_config.get("min_batch_sentences", 1)
        self.batch_flush_interval = local_config.get("batch_flush_interval", 0.75)

        # LiveKit video streaming (optional - for data collection)
        self.livekit_streamer: Optional[LiveKitVideoStreamer] = None
        vision_config = config.get("vision", {})
        self.stream_video_to_livekit = vision_config.get("stream_to_livekit_local", True)

        if self.stream_video_to_livekit and LIVEKIT_AVAILABLE:
            self.livekit_streamer = LiveKitVideoStreamer(config)
            if self.livekit_streamer.is_configured:
                logger.info("LiveKit video streaming configured for local pipeline")
            else:
                logger.info("LiveKit not configured, video streaming disabled")

    async def initialize(self, agent) -> PipelineSession:
        """Initialize pipeline with LeLamp agent."""
        self.agent = agent

        # Set system prompt from agent's instructions
        if hasattr(agent, "_instructions"):
            self.llm.set_system_prompt(agent._instructions)
        elif hasattr(agent, "instructions"):
            self.llm.set_system_prompt(agent.instructions)

        # Extract and register function tools from agent
        tools = extract_tools_from_agent(agent)
        self.llm.register_tools_from_list(tools)
        logger.info(f"Registered {len(tools)} function tools with Ollama")

        # Connect to LiveKit for video streaming (if configured)
        if self.livekit_streamer and self.livekit_streamer.is_configured:
            connected = await self.livekit_streamer.connect()
            if connected:
                # Setup video streaming via VisionService
                await self._setup_livekit_video()

        # Create session
        self.session = LocalSession(self)
        return self.session

    async def _setup_livekit_video(self):
        """Setup VisionService for LiveKit video streaming."""
        if not self.livekit_streamer or not self.livekit_streamer.is_connected:
            return

        vision_service = g.vision_service
        if not vision_service:
            logger.warning("VisionService not available for LiveKit video streaming")
            return

        try:
            # Set the LiveKit room on VisionService
            vision_service.livekit_room = self.livekit_streamer.room
            vision_service.publish_video = True

            # Setup video publishing
            await vision_service.setup_livekit_publishing()
            logger.info("✓ VisionService configured for LiveKit video streaming")
        except Exception as e:
            logger.error(f"Failed to setup LiveKit video streaming: {e}", exc_info=True)

    async def run(self):
        """Main pipeline run (blocking)."""
        print("[PIPELINE.RUN] Entered, calling session.start()...")
        await self.session.start()
        print("[PIPELINE.RUN] session.start() returned, entering main loop...")

        # Keep running until stopped
        try:
            while self.session._running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("[PIPELINE.RUN] Cancelled")
            pass
        finally:
            print("[PIPELINE.RUN] Stopping session...")
            await self.session.stop()

    async def _process_user_input(self, text: str, metrics=None):
        """Process user input and generate streamed response with TTS."""
        if metrics:
            metrics.record_timestamp(PipelineStage.LLM_START)

        # Trigger speaking start callback
        if self._on_agent_speech_start:
            self._on_agent_speech_start()

        # Trigger RGB animation if available
        if hasattr(self.agent, "rgb_service") and self.agent.rgb_service:
            self.agent.rgb_service.dispatch(
                "animation", {"name": "speaking", "color": (0, 150, 255)}
            )

        response_text = ""
        full_response = ""
        last_flush_time = time.time()
        batch_sentences = []
        first_tts_sent = False

        # Stream from LLM with TTS
        async for token in self.llm.generate_response(text):
            response_text += token

            current_time = time.time()
            should_flush = (
                current_time - last_flush_time >= self.flush_interval
                or len(response_text) > self.max_response_length
            )

            if should_flush:
                response_text = clean_response(response_text)
                sentences = split_sentences(response_text)

                if sentences:
                    *complete_sentences, response_text = sentences
                    batch_sentences.extend(complete_sentences)

                    should_speak = (
                        len(batch_sentences) >= self.min_batch_sentences
                        or current_time - last_flush_time >= self.batch_flush_interval
                    )

                    if should_speak and batch_sentences:
                        combined = " ".join(
                            s.strip() for s in batch_sentences if s.strip()
                        )

                        if combined:
                            if not first_tts_sent:
                                first_tts_sent = True
                                if metrics:
                                    metrics.record_timestamp(PipelineStage.TTS_START)
                                    metrics.set_agent_state("speaking")

                            logger.info(f"Agent: {combined}")
                            full_response += combined + " "

                            # Synthesize and play
                            async for audio_chunk in self.tts.synthesize(combined):
                                self.audio_io.play_audio(audio_chunk)

                        batch_sentences = []
                        last_flush_time = current_time

        # Handle remaining text
        response_text = clean_response(response_text)
        sentences = split_sentences(response_text)
        if response_text and not sentences:
            sentences = [response_text]
        batch_sentences.extend(sentences)

        combined = " ".join(s.strip() for s in batch_sentences if s.strip())
        if combined:
            logger.info(f"Agent: {combined}")
            full_response += combined + " "

            async for audio_chunk in self.tts.synthesize(combined):
                self.audio_io.play_audio(audio_chunk)

        # Handle silent response: if agent outputs just "." it means intentional silence
        if full_response.strip() == ".":
            logger.info("Agent chose to remain silent (selective response)")
            full_response = ""
        # Fallback if no response generated (but NOT for intentional silence)
        elif not full_response.strip():
            fallback = "Sorry, I didn't catch that. Could you repeat?"
            logger.info(f"Agent (fallback): {fallback}")

            async for audio_chunk in self.tts.synthesize(fallback):
                self.audio_io.play_audio(audio_chunk)

            full_response = fallback

        # Wait for playback to complete
        await self.audio_io.wait_for_playback()

        if metrics:
            metrics.record_timestamp(PipelineStage.TTS_END)
            metrics.set_agent_text(full_response.strip())

        # Trigger speaking end callback
        if self._on_agent_speech_end:
            self._on_agent_speech_end()

        # Return to idle RGB animation
        if hasattr(self.agent, "rgb_service") and self.agent.rgb_service:
            rgb_config = self.config.get("rgb", {})
            default_anim = rgb_config.get("default_animation", "aura_glow")
            default_color = tuple(rgb_config.get("default_color", [255, 255, 255]))
            self.agent.rgb_service.dispatch(
                "animation", {"name": default_anim, "color": default_color}
            )

    async def _speak_response(self, text: str):
        """Directly speak a text response (for generate_reply)."""
        # Mute mic
        print(f"[_speak_response] Muting mic, text: {text[:50]}...")
        self.audio_io.mute_mic(True)

        try:
            # Trigger speaking callback
            if self._on_agent_speech_start:
                self._on_agent_speech_start()

            # Split into sentences and speak
            sentences = split_sentences(clean_response(text))
            if not sentences:
                sentences = [text]

            # Buffer all audio first, then play - prevents choppy playback
            all_audio_chunks = []
            for sentence in sentences:
                if sentence.strip():
                    logger.info(f"Agent (direct): {sentence}")
                    async for audio_chunk in self.tts.synthesize(sentence):
                        all_audio_chunks.append(audio_chunk)

            # Now queue all audio for playback
            for chunk in all_audio_chunks:
                self.audio_io.play_audio(chunk)

            # Wait for playback
            print("[_speak_response] Waiting for playback to complete...")
            await self.audio_io.wait_for_playback()
            print("[_speak_response] Playback complete!")

            if self._on_agent_speech_end:
                self._on_agent_speech_end()
        except Exception as e:
            logger.error(f"Error in _speak_response: {e}", exc_info=True)
            print(f"[_speak_response] ERROR: {e}")
        finally:
            # Always unmute mic, even if there was an error
            # Reset STT state to prevent false triggers from audio bleed
            print("[_speak_response] Resetting STT and clearing input queue")
            self.stt.reset()
            self.audio_io.clear_input_queue()
            # Small delay to let audio settle before unmuting
            import asyncio
            await asyncio.sleep(0.2)
            print("[_speak_response] Unmuting mic (finally block)")
            self.audio_io.mute_mic(False)


async def run_local_pipeline():
    """
    Run the local voice pipeline (Faster Whisper + Ollama + Piper).

    This is the main entry point for local pipeline mode.
    Sets up the agent, callbacks, and runs the voice loop.
    Optionally streams video to LiveKit for data collection.
    """
    from . import get_pipeline

    print("\n" + "="*60)
    print("LOCAL VOICE PIPELINE")
    print("="*60)
    print("Using: Faster Whisper + Ollama + Piper")
    print("Audio: Direct ALSA (local processing)")
    if LIVEKIT_AVAILABLE:
        print("Video: LiveKit streaming available (if configured)")
    else:
        print("Video: LiveKit not available")
    print("="*60 + "\n")

    # Create LeLamp agent
    agent = LeLamp()

    # Initialize callback service for timer/alarm handling
    init_callback_service(agent)

    # Make agent accessible globally
    # Note: Services are already in globals (initialized by server.py)
    # Agent just consumes them - no need to copy
    g.lelamp_agent = agent

    # Get local pipeline
    pipeline = get_pipeline(g.CONFIG)

    # Initialize pipeline with agent (also connects to LiveKit for video if configured)
    session = await pipeline.initialize(agent)
    agent.agent_session = session
    agent.event_loop = asyncio.get_running_loop()
    g.agent_session_global = session

    # Log video streaming status
    if hasattr(pipeline, 'livekit_streamer') and pipeline.livekit_streamer:
        if pipeline.livekit_streamer.is_connected:
            print("✓ LiveKit video streaming active for data collection")
        elif pipeline.livekit_streamer.is_configured:
            print("⚠ LiveKit configured but connection failed")
        else:
            print("ℹ LiveKit not configured, video streaming disabled")

    # Register callbacks for animations
    def on_user_speech_start():
        if agent.rgb_service:
            agent.rgb_service.dispatch("animation", {"name": "listening", "color": (0, 255, 100)})
        if agent.animation_service:
            agent.animation_service.dispatch("play", "listening")

    def on_agent_speech_start():
        if agent.rgb_service:
            agent.rgb_service.dispatch("animation", {"name": "speaking", "color": (0, 150, 255)})

    def on_agent_speech_end():
        rgb_config = g.CONFIG.get("rgb", {})
        default_anim = rgb_config.get("default_animation", "aura_glow")
        default_color = tuple(rgb_config.get("default_color", [255, 255, 255]))
        if agent.rgb_service:
            agent.rgb_service.dispatch("animation", {"name": default_anim, "color": default_color})

    pipeline.on_user_speech_start(on_user_speech_start)
    pipeline.on_agent_speech_start(on_agent_speech_start)
    pipeline.on_agent_speech_end(on_agent_speech_end)

    logger.info("Local pipeline initialized")
    print("[RUN_LOCAL_PIPELINE] Pipeline initialized, starting session...")

    # Start the session WITHOUT voice loop (to avoid interference during greeting)
    await session.start(start_voice_loop=False)
    print("[RUN_LOCAL_PIPELINE] Session started, playing startup greeting...")

    # Play startup greeting (now TTS is available)
    await pipeline._speak_response(
        "I am now running on local AI. How can I help you?"
    )
    print("[RUN_LOCAL_PIPELINE] Startup greeting done, starting voice loop...")

    # NOW start the voice loop (after greeting is done and mic is clean)
    session.start_voice_loop()
    print("[RUN_LOCAL_PIPELINE] Voice loop started, entering main loop...")

    # Run the voice loop (blocking) - session already started
    try:
        while session._running:
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass
    finally:
        await session.stop()
        # Cleanup LiveKit connection
        if hasattr(pipeline, 'livekit_streamer') and pipeline.livekit_streamer:
            await pipeline.livekit_streamer.disconnect()
    print("[RUN_LOCAL_PIPELINE] pipeline.run() returned (this should not happen normally)")
