"""
LiveKit Realtime voice pipeline supporting multiple AI providers.

Supported providers:
- openai: OpenAI Realtime API (GPT-4o)
- grok: xAI Grok Voice Agent API
- gemini: Google Gemini Live API
- azure: Azure OpenAI Realtime API
- aws: Amazon Nova Sonic
- ultravox: Ultravox Realtime

Contains the entrypoint function that LiveKit agents framework calls.
"""

import asyncio
import logging
import threading
from typing import Optional

from livekit import agents
from livekit.agents import (
    AgentSession,
    RoomInputOptions,
)
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)

# Optional provider imports - gracefully handle missing plugins
try:
    from livekit.plugins import xai
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    xai = None


try:
    from livekit.plugins import google
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    google = None

try:
    from livekit.plugins import azure
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    azure = None

# Note: AWS and Ultravox commented out until dependencies are tested
# try:
#     from livekit.plugins import aws
#     AWS_AVAILABLE = True
# except ImportError:
#     AWS_AVAILABLE = False
#     aws = None

# try:
#     from livekit.plugins import ultravox
#     ULTRAVOX_AVAILABLE = True
# except ImportError:
#     ULTRAVOX_AVAILABLE = False
#     ultravox = None

import lelamp.globals as g
from lelamp.agent import LeLamp
from lelamp.service.callbacks import init_callback_service
from lelamp.service.wake import WakeService
from lelamp.service.theme import ThemeSound, get_theme_service
from lelamp.service.metrics_service import get_metrics_service, PipelineStage

from .base import PipelineBase, PipelineSession

logger = logging.getLogger(__name__)

# Available providers and their status
PROVIDERS = {
    "openai": {"available": True, "name": "OpenAI Realtime", "key_env": "OPENAI_API_KEY"},
    "grok": {"available": XAI_AVAILABLE, "name": "xAI Grok", "key_env": "XAI_API_KEY"},
    "gemini": {"available": GOOGLE_AVAILABLE, "name": "Google Gemini Live", "key_env": "GOOGLE_API_KEY"},
    "azure": {"available": AZURE_AVAILABLE, "name": "Azure OpenAI", "key_env": "AZURE_OPENAI_API_KEY"},
    # "aws": {"available": AWS_AVAILABLE, "name": "AWS Nova Sonic", "key_env": "AWS_ACCESS_KEY_ID"},
    # "ultravox": {"available": ULTRAVOX_AVAILABLE, "name": "Ultravox", "key_env": "ULTRAVOX_API_KEY"},
}

# Default voices per provider
DEFAULT_VOICES = {
    "openai": "ballad",
    "grok": "Charon",
    "gemini": "Puck",
    "azure": "alloy",
    # "aws": "default",
    # "ultravox": "default",
}

# Garbage transcription filter - patterns that indicate motor noise or non-speech
GARBAGE_PATTERNS = [
    # Non-English that Whisper hallucinates from motor noise
    "Музыкальная", "музыка", "композиция",  # Russian
    "Untertitelung", "Untertitel",  # German subtitles
    "Sous-titres", "Sous-titre",  # French subtitles
    "字幕", "謝謝", "请订阅",  # Chinese
    "ご視聴", "チャンネル登録",  # Japanese
    "구독", "좋아요",  # Korean
    # Common Whisper hallucinations on noise
    "Thank you for watching",
    "Thanks for watching",
    "Please subscribe",
    "Like and subscribe",
    "See you next time",
    "Bye bye",
    "...",  # Just ellipsis
    "♪", "♫",  # Music symbols
]


def is_garbage_transcription(text: str) -> bool:
    """Check if transcription is likely motor noise or hallucination."""
    if not text or len(text.strip()) < 2:
        return True
    text_lower = text.lower().strip()
    # Check for known garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if pattern.lower() in text_lower:
            return True
    # Check if mostly non-ASCII (likely non-English hallucination)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.5:
        return True
    return False


def create_realtime_model(provider: str, voice: str, turn_detection: Optional[str] = "server_vad"):
    """
    Factory function to create the appropriate realtime model.

    Args:
        provider: Provider name (openai, grok, gemini, azure)
        voice: Voice ID for the provider
        turn_detection: Turn detection mode ("server_vad" or None)

    Returns:
        Configured RealtimeModel instance

    Raises:
        ValueError: If provider is not available or not supported
    """
    if provider == "openai":
        # Get VAD settings from config (microphone section)
        mic_config = g.CONFIG.get("microphone", {})
        vad_mode = mic_config.get("vad_mode", "server")

        if vad_mode == "local":
            # Disable OpenAI's server VAD - we'll use local Silero VAD instead
            logger.info(f"Creating OpenAI RealtimeModel (voice={voice}, server VAD DISABLED - using local VAD)")
            return openai.realtime.RealtimeModel(
                voice=voice,
                turn_detection=None,  # Disable server-side VAD
            )
        else:
            # Use server VAD with custom settings
            from livekit.plugins.openai.realtime.realtime_model import TurnDetection

            server_vad_threshold = mic_config.get("server_vad_threshold", 0.75)
            server_silence_duration = mic_config.get("server_silence_duration_ms", 600)

            # Higher threshold = needs louder speech to trigger (reduces echo sensitivity)
            vad_config = TurnDetection(
                type="server_vad",
                threshold=server_vad_threshold,
                silence_duration_ms=server_silence_duration,
                prefix_padding_ms=300,
            )
            logger.info(f"Creating OpenAI RealtimeModel (voice={voice}, server VAD threshold={server_vad_threshold}, silence={server_silence_duration}ms)")
            return openai.realtime.RealtimeModel(
                voice=voice,
                turn_detection=vad_config,
            )

    elif provider == "grok":
        if not XAI_AVAILABLE:
            raise ValueError("xAI plugin not installed. Run: uv add livekit-plugins-xai")
        # xAI has server VAD enabled by default - don't pass turn_detection string
        # Pass None to disable, or omit to use default ServerVad
        logger.info(f"Creating xAI Grok RealtimeModel (voice={voice}, using default server VAD)")
        if turn_detection is None:
            # Explicitly disable turn detection
            return xai.realtime.RealtimeModel(
                voice=voice,
                turn_detection=None,
            )
        else:
            # Use xAI's default ServerVad (already configured)
            return xai.realtime.RealtimeModel(
                voice=voice,
            )

    elif provider == "gemini":
        if not GOOGLE_AVAILABLE:
            raise ValueError("Google plugin not installed. Run: uv add livekit-plugins-google")
        logger.info(f"Creating Google Gemini RealtimeModel (voice={voice})")
        return google.realtime.RealtimeModel(
            voice=voice,
        )

    elif provider == "azure":
        if not AZURE_AVAILABLE:
            raise ValueError("Azure plugin not installed. Run: uv add livekit-plugins-azure")
        logger.info(f"Creating Azure OpenAI RealtimeModel (voice={voice})")
        return azure.realtime.RealtimeModel(
            voice=voice,
            turn_detection=turn_detection,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(PROVIDERS.keys())}")


async def entrypoint(ctx: agents.JobContext):
    """
    LiveKit agent entrypoint.

    Called by the LiveKit agents framework when a new session starts.
    Sets up the agent, VAD, metrics, vision, and session callbacks.
    """
    CONFIG = g.CONFIG

    agent = LeLamp()

    # Make agent accessible globally
    # Note: Services are already in globals (initialized by server.py)
    # Agent just consumes them - no need to copy
    g.lelamp_agent = agent

    # Initialize callback service for timer/alarm handling
    init_callback_service(agent)

    # Initialize wake service for sleep mode
    try:
        wake_service = WakeService(
            wake_phrases=["wake up", "hey lamp", "wake"],
            model_size="tiny"  # Fastest Whisper model for RPi
        )
        g.wake_service = wake_service
        logger.info("Wake service initialized with Whisper (will activate during sleep)")
    except Exception as e:
        logger.warning(f"Wake service not available: {e}")
        g.wake_service = None

    # ==========================================================================
    # SIMPLIFIED TURN DETECTION ARCHITECTURE
    # ==========================================================================
    # Let each realtime provider handle turn detection natively.
    # No local VAD, no mic muting - trust provider's server-side VAD + echo cancellation.
    # This enables natural conversation with interruptions.
    # ==========================================================================

    # Get provider and voice from config
    pipeline_config = CONFIG.get("pipeline", {})
    provider = pipeline_config.get("provider", "openai")
    default_voice = DEFAULT_VOICES.get(provider, "alloy")
    voice = pipeline_config.get("voice", default_voice)

    logger.info(f"Using provider: {provider}, voice: {voice}")

    # Create the realtime model - each provider uses its native/default turn detection
    # OpenAI: pass "server_vad" string
    # Grok/Gemini: use their built-in defaults (don't pass anything)
    try:
        realtime_model = create_realtime_model(provider, voice, turn_detection="server_vad")
    except ValueError as e:
        logger.error(f"Failed to create realtime model: {e}")
        raise

    # VAD mode from config - "local" uses Silero VAD, "server" uses provider's VAD
    mic_config = CONFIG.get("microphone", {})
    vad_mode = mic_config.get("vad_mode", "server")
    local_vad_threshold = mic_config.get("local_vad_threshold", 0.5)

    if vad_mode == "local":
        # Use local Silero VAD for turn detection
        # This gives us control over VAD and prevents echo from triggering server VAD
        vad_model = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.3,
            activation_threshold=local_vad_threshold,
        )
        turn_detection_mode = "vad"  # Use local VAD for turn detection
        logger.info(f"Using LOCAL Silero VAD (threshold={local_vad_threshold}) - server VAD disabled")
    else:
        # No local VAD - let provider handle it server-side
        vad_model = None
        turn_detection_mode = "realtime_llm"  # Let LLM's server VAD handle it
        logger.info(f"Using {provider}'s native server-side VAD (no local VAD)")

    # Endpointing settings from config
    endpointing_config = CONFIG.get("endpointing", {})

    # Session config - VAD mode determines turn detection
    session = AgentSession(
        llm=realtime_model,
        vad=vad_model,
        allow_interruptions=True,
        discard_audio_if_uninterruptible=True,
        turn_detection=turn_detection_mode,
        # Endpointing - give user time to speak
        min_endpointing_delay=endpointing_config.get("min_delay", 0.8),
        max_endpointing_delay=endpointing_config.get("max_delay", 6.0),
        min_consecutive_speech_delay=endpointing_config.get("min_consecutive_speech_delay", 0.5),
    )

    # Store session reference and event loop in agent for timer notifications
    agent.agent_session = session
    agent.event_loop = asyncio.get_running_loop()
    agent.vad_model = vad_model
    g.agent_session_global = session
    g.livekit_room = ctx.room
    g.livekit_ctx = ctx
    logger.info(f"Stored event loop for timer notifications: {agent.event_loop}")
    logger.info(f"Event loop is running: {agent.event_loop.is_running()}")
    if vad_model:
        logger.info(f"VAD model stored for dynamic threshold (normal={agent.vad_normal_threshold}, music={agent.vad_music_threshold})")

    # Start idle timeout checker task
    agent.idle_timeout_task = asyncio.create_task(agent._check_idle_timeout())
    logger.info("Idle timeout checker started")

    # Start workflow progression checker task
    agent.workflow_progression_task = asyncio.create_task(agent._check_workflow_progression())
    logger.info("Workflow progression checker started")

    # ============================================================================
    # Metrics Collection Hooks
    # ============================================================================
    metrics = get_metrics_service()

    # Volume ducking state
    volume_restore_task = None
    original_volume = CONFIG.get("volume", 100)
    ducked_volume = 50  # Volume when user is speaking during music

    async def restore_volume_after_delay():
        """Restore volume after user stops speaking."""
        await asyncio.sleep(2.0)
        if agent.spotify_service and agent.spotify_service._is_playing:
            agent._set_system_volume(original_volume)
            print(f"\033[93m🔊 VOLUME: Restored to {original_volume}%\033[0m")

    # Track user state changes (speaking/listening)
    @session.on("user_state_changed")
    def on_user_state_changed(event):
        nonlocal volume_restore_task
        if event.new_state == "speaking":
            metrics.start_turn()
            metrics.record_timestamp(PipelineStage.VAD_START)
            metrics.set_user_speaking(True)
            logger.info("METRICS: User started speaking")

            # Volume ducking: lower music when user speaks
            if agent.spotify_service and agent.spotify_service._is_playing:
                # Cancel any pending restore
                if volume_restore_task and not volume_restore_task.done():
                    volume_restore_task.cancel()
                agent._set_system_volume(ducked_volume)
                print(f"\033[93m🔉 VOLUME: Ducked to {ducked_volume}% (user speaking)\033[0m")

        elif event.old_state == "speaking":
            metrics.record_timestamp(PipelineStage.VAD_END)
            metrics.set_user_speaking(False)
            logger.info("METRICS: User stopped speaking")

            # Schedule volume restore after delay
            if agent.spotify_service and agent.spotify_service._is_playing:
                volume_restore_task = asyncio.create_task(restore_volume_after_delay())

    # Track user transcription
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        metrics.record_timestamp(PipelineStage.STT_END)
        if hasattr(event, 'transcript') and event.transcript:
            transcript = event.transcript

            # Always log raw transcription for debugging
            debug_transcription = mic_config.get("debug_transcription", False)
            if debug_transcription:
                logger.info(f"🎙️ RAW TRANSCRIPTION: '{transcript}'")

            # Filter out garbage transcriptions (motor noise, non-English hallucinations)
            # Can be disabled via config for debugging
            filter_garbage = mic_config.get("filter_garbage_transcription", True)
            if filter_garbage and is_garbage_transcription(transcript):
                logger.warning(f"🗑️ Filtered garbage transcription: {transcript[:50]}...")
                # Interrupt any pending response to this garbage
                try:
                    session.interrupt()
                    session.clear_user_turn()
                except Exception:
                    pass
                return

            metrics.set_user_text(transcript)
            logger.info(f"🎤 USER SAID: {transcript}")

    # Track agent state changes for detailed pipeline metrics
    @session.on("agent_state_changed")
    def on_agent_state_metrics(event):
        metrics.set_agent_state(event.new_state)

        # Track when agent starts thinking (LLM processing)
        if event.new_state == "thinking":
            metrics.record_timestamp(PipelineStage.LLM_START)
            logger.info("METRICS: Agent thinking (LLM start)")

        # Track when agent starts speaking (TTS first audio)
        elif event.new_state == "speaking":
            metrics.record_timestamp(PipelineStage.AUDIO_PLAY_START)
            metrics.record_timestamp(PipelineStage.TTS_FIRST_AUDIO)
            logger.info("METRICS: Agent speaking (audio start)")
            # Notify audio service - this triggers mic gating
            if g.audio_service:
                g.audio_service.set_playing_state(True)

        # Track when agent finishes speaking
        elif event.old_state == "speaking" and event.new_state in ("listening", "idle"):
            metrics.record_timestamp(PipelineStage.AUDIO_PLAY_END)
            metrics.record_timestamp(PipelineStage.TTS_END)
            metrics.end_turn()
            logger.info("METRICS: Turn complete")
            # Notify audio service - this releases mic gating after delay
            if g.audio_service:
                g.audio_service.set_playing_state(False)

    # Track conversation items (user and agent messages)
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        if hasattr(event, 'item') and event.item:
            item = event.item
            role = getattr(item, 'role', None)
            item_type = getattr(item, 'type', 'unknown')

            logger.debug(f"METRICS: conversation_item_added - role={role}, type={item_type}, item_class={type(item).__name__}")

            # Try to extract text using text_content property first (ChatMessage has this)
            text = None
            if hasattr(item, 'text_content') and item.text_content:
                text = item.text_content
            elif hasattr(item, 'content'):
                # Fallback: manually extract from content list
                if isinstance(item.content, str):
                    text = item.content
                elif isinstance(item.content, list):
                    text_parts = []
                    for c in item.content:
                        if isinstance(c, str):
                            text_parts.append(c)
                        elif hasattr(c, 'text'):
                            text_parts.append(c.text)
                    text = " ".join(text_parts) if text_parts else None

            if text:
                if role == 'assistant':
                    metrics.set_agent_text(text)
                    logger.info(f"METRICS: Agent said: {text[:80]}...")
                elif role == 'user':
                    # Also capture user text from conversation items as backup
                    metrics.set_user_text(text)
                    logger.info(f"METRICS: User said: {text[:80]}...")
            else:
                # Log when we get a conversation item but can't extract text
                if role == 'assistant':
                    logger.debug("METRICS: Agent conversation item without text (audio-only response)")

    # Track LiveKit's built-in metrics
    @session.on("metrics_collected")
    def on_metrics_collected(event):
        m = event.metrics
        metric_type = getattr(m, 'type', 'unknown')

        # Record to metrics service for WebUI
        metrics.record_livekit_metrics(m)

        if metric_type == 'realtime_model_metrics':
            # Realtime model metrics - most relevant for your setup
            logger.info(f"METRICS [Realtime]: TTFT={m.ttft*1000:.1f}ms, duration={m.duration*1000:.1f}ms, tokens={m.total_tokens}")
            if m.ttft > 0:
                metrics.record_timestamp(PipelineStage.LLM_FIRST_TOKEN)

        elif metric_type == 'llm_metrics':
            logger.info(f"METRICS [LLM]: TTFT={m.ttft*1000:.1f}ms, duration={m.duration*1000:.1f}ms, tokens={m.total_tokens}")

        elif metric_type == 'tts_metrics':
            logger.info(f"METRICS [TTS]: TTFB={m.ttfb*1000:.1f}ms, duration={m.duration*1000:.1f}ms, audio={m.audio_duration:.2f}s")

        elif metric_type == 'stt_metrics':
            logger.info(f"METRICS [STT]: duration={m.duration*1000:.1f}ms, audio={m.audio_duration:.2f}s")

        elif metric_type == 'vad_metrics':
            # VAD metrics are too noisy - skip logging, just update metrics service
            pass

        elif metric_type == 'eou_metrics':
            logger.info(f"METRICS [EOU]: utterance_delay={m.end_of_utterance_delay*1000:.1f}ms, transcription_delay={m.transcription_delay*1000:.1f}ms")

    # Block any transcriptions during sleep - re-disable audio aggressively
    @session.on("user_input_transcribed")
    def on_user_speech_sleep_check(event):
        if agent.is_sleeping:
            transcript = event.transcript.lower() if hasattr(event, 'transcript') and event.transcript else ""
            logger.warning(f"🛑 BLOCKED transcription during sleep: '{transcript}'")
            # Re-disable audio input/output (belt and suspenders)
            try:
                session.input.set_audio_enabled(False)
                session.output.set_audio_enabled(False)
                logger.info("🛑 Re-disabled audio input/output")
            except Exception as e:
                logger.error(f"Error re-disabling audio: {e}")

    # Also block when agent tries to speak during sleep
    @session.on("agent_state_changed")
    def on_agent_state_sleep_block(event):
        if agent.is_sleeping and event.new_state in ("thinking", "speaking"):
            logger.warning(f"🛑 Agent trying to {event.new_state} during sleep - interrupting")
            try:
                session.interrupt()
            except Exception:
                pass  # May fail if nothing to interrupt, that's ok

    # Mark activity when agent speaks (for idle timeout)
    @session.on("agent_state_changed")
    def on_activity_tracking(event):
        if event.new_state == "speaking":
            agent._mark_activity()

    # ============================================================================
    # Vision Setup
    # ============================================================================
    vision_config = CONFIG.get("vision", {})
    vision_enabled = vision_config.get("enabled", False)
    publish_video = vision_config.get("publish_livekit_video", False)
    publish_image = vision_config.get("publish_livekit_image", False)

    vision_service = g.vision_service

    # Setup LiveKit video publishing BEFORE session starts (if enabled)
    if vision_enabled and publish_video and vision_service:
        vision_service.livekit_room = ctx.room
        await vision_service.setup_livekit_publishing()
        logger.info(f"Vision enabled with provider: {vision_config.get('provider', 'openai')}, video track published")

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
            video_enabled=vision_enabled,
        ),
    )

    # Setup image frame callback AFTER session starts (for periodic still images to OpenAI)
    if vision_enabled and publish_image and vision_service:
        # Get the realtime session once in async context and cache it
        realtime_session = None
        if hasattr(session, 'llm') and callable(getattr(session.llm, 'session', None)):
            try:
                realtime_session = session.llm.session()
                logger.info(f"Realtime session type: {type(realtime_session)}")
                logger.info(f"Realtime session attributes: {[attr for attr in dir(realtime_session) if not attr.startswith('_')]}")
            except Exception as e:
                logger.error(f"Failed to get realtime session: {e}", exc_info=True)

        def on_image_frame(base64_image: str):
            """Callback to send image frames to Realtime API"""
            try:
                if realtime_session is None:
                    if not hasattr(on_image_frame, '_warned'):
                        logger.warning("No realtime session available for image sending")
                        on_image_frame._warned = True
                    return

                # Access the chat context and add an image message
                if hasattr(realtime_session, 'chat_ctx'):
                    from livekit.agents.llm import ImageContent, ChatMessage

                    # Check chat_ctx type
                    if not hasattr(on_image_frame, '_checked_ctx'):
                        logger.info(f"chat_ctx type: {type(realtime_session.chat_ctx)}")
                        on_image_frame._checked_ctx = True

                    # Create image content
                    img_content = ImageContent(image=f"data:image/jpeg;base64,{base64_image}")

                    # Add to chat context
                    if hasattr(realtime_session.chat_ctx, 'messages'):
                        realtime_session.chat_ctx.messages.append(
                            ChatMessage(role="user", content=[img_content])
                        )
                        if not hasattr(on_image_frame, '_sent_count'):
                            on_image_frame._sent_count = 0
                        on_image_frame._sent_count += 1
                        if on_image_frame._sent_count % 10 == 0:
                            logger.info(f"Added {on_image_frame._sent_count} images to chat context")
                    else:
                        if not hasattr(on_image_frame, '_warned_no_messages'):
                            logger.warning("chat_ctx has no messages attribute")
                            on_image_frame._warned_no_messages = True
                else:
                    if not hasattr(on_image_frame, '_warned_no_ctx'):
                        logger.warning("Realtime session has no chat_ctx attribute")
                        on_image_frame._warned_no_ctx = True
            except Exception as e:
                if not hasattr(on_image_frame, '_error_count'):
                    on_image_frame._error_count = 0
                on_image_frame._error_count += 1
                if on_image_frame._error_count <= 3:
                    logger.error(f"Error sending image to Realtime API: {e}", exc_info=True)

        vision_service.set_image_callback(on_image_frame)
        logger.info(f"Vision image callback registered at {vision_config.get('image_fps', 1.0)} fps")

    # ============================================================================
    # Ollama Vision Service - Local scene understanding
    # ============================================================================
    ollama_config = vision_config.get("ollama", {})
    ollama_enabled = ollama_config.get("enabled", False)

    if ollama_enabled:
        try:
            from lelamp.service.vision.ollama_vision_service import (
                OllamaVisionService,
                init_ollama_vision_service
            )

            # Get camera from existing vision service if available
            camera_index = CONFIG.get("face_tracking", {}).get("camera_index", 0)
            resolution = tuple(CONFIG.get("face_tracking", {}).get("resolution", [640, 480]))

            # Initialize Ollama vision service
            ollama_vision = init_ollama_vision_service(
                ollama_url=ollama_config.get("url", "http://localhost:11434"),
                model=ollama_config.get("model", "gemma3:270m"),
                analysis_interval=ollama_config.get("analysis_interval", 5.0),
                camera_index=camera_index,
                resolution=resolution,
                max_frame_size=vision_config.get("max_frame_size", 512),
                instructions_file=ollama_config.get("instructions_file")
            )

            # Share camera with existing vision service if available
            if vision_service and vision_service.cap:
                ollama_vision.set_camera(vision_service.cap)

            # Store in globals
            g.ollama_vision_service = ollama_vision

            # Setup proactive comment callback (agent comments on significant changes)
            if ollama_config.get("proactive_comments", True):
                def on_scene_change(context, change_description):
                    """Callback when significant scene change is detected"""
                    if agent.is_sleeping:
                        return  # Don't comment while sleeping

                    if agent.agent_session and agent.event_loop:
                        def _comment_on_change():
                            agent.agent_session.generate_reply(
                                instructions=f"SCENE CHANGE DETECTED: {change_description}. "
                                "If appropriate, make a brief, natural comment about this change. "
                                "Only comment if it's something worth noting - don't interrupt important conversation."
                            )
                        agent.event_loop.call_soon_threadsafe(_comment_on_change)
                        logger.info(f"Triggered proactive comment for scene change: {change_description}")

                ollama_vision.on_scene_change(on_scene_change)

            # Setup context update callback (for system prompt injection)
            if ollama_config.get("inject_system_prompt", True):
                def on_context_update(context):
                    """Update global scene context for system prompt injection"""
                    g.current_scene_context = context.to_prompt_string()

                ollama_vision.on_context_update(on_context_update)

            # Start the service
            ollama_vision.start(event_loop=asyncio.get_running_loop())
            logger.info(f"Ollama vision service started (model={ollama_config.get('model')}, interval={ollama_config.get('analysis_interval')}s)")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama vision service: {e}", exc_info=True)

    # ============================================================================
    # Initial Instructions
    # ============================================================================
    initial_instructions = "When you wake up, starts with Tadaaaa. Only speak in English, never in Vietnamese."

    if vision_enabled:
        vision_instructions = vision_config.get("instructions", "")
        if vision_instructions:
            initial_instructions += f"\n\n{vision_instructions}"

    # Add Ollama vision context if enabled
    if ollama_enabled and ollama_config.get("inject_system_prompt", True):
        initial_instructions += (
            "\n\nYou have scene awareness through a local vision AI. "
            "You can use the describe_scene() or get_scene_details() tools to see what's around you. "
            "The scene is analyzed periodically and you may comment on significant changes naturally."
        )

    # Start the agent
    await session.generate_reply(
        instructions=initial_instructions
    )


class LiveKitRealtimeSession(PipelineSession):
    """Wrapper around LiveKit AgentSession."""

    def __init__(self, livekit_session):
        self._session = livekit_session

    async def start(self):
        """Session is started by LiveKit framework."""
        pass

    async def stop(self):
        """Session is stopped by LiveKit framework."""
        pass

    async def generate_reply(self, instructions: str):
        """Generate a spoken reply."""
        await self._session.generate_reply(instructions=instructions)

    def interrupt(self):
        """Interrupt current speech."""
        try:
            self._session.interrupt()
        except Exception:
            pass

    def set_audio_enabled(self, enabled: bool):
        """Enable/disable audio input."""
        try:
            self._session.input.set_audio_enabled(enabled)
        except Exception:
            pass


class LiveKitRealtimePipeline(PipelineBase):
    """
    Pipeline using LiveKit agents with multiple realtime AI providers.

    The entrypoint() function above is the main implementation,
    called by the LiveKit worker via agents.cli.run_app().
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._livekit_session = None

    async def initialize(self, agent) -> PipelineSession:
        """
        Initialize is called after LiveKit sets up the session.

        In LiveKit mode, the session is created by the entrypoint function,
        not here. This method is called to wrap that session.
        """
        self.agent = agent
        return None

    def set_livekit_session(self, session):
        """
        Set the LiveKit AgentSession (called from main.py).

        Args:
            session: LiveKit AgentSession instance
        """
        self._livekit_session = session
        self.session = LiveKitRealtimeSession(session)
        return self.session

    async def run(self):
        """
        Run is handled by LiveKit's agents.cli.run_app().

        This method is not used in LiveKit mode - the worker
        runs the entrypoint function directly.
        """
        raise NotImplementedError(
            "LiveKit pipeline is run via agents.cli.run_app(), not pipeline.run()"
        )

    def on_user_speech_start(self, callback):
        """Callbacks are handled via LiveKit session events."""
        self._on_user_speech_start = callback

    def on_user_speech_end(self, callback):
        """Callbacks are handled via LiveKit session events."""
        self._on_user_speech_end = callback

    def on_agent_speech_start(self, callback):
        """Callbacks are handled via LiveKit session events."""
        self._on_agent_speech_start = callback

    def on_agent_speech_end(self, callback):
        """Callbacks are handled via LiveKit session events."""
        self._on_agent_speech_end = callback
