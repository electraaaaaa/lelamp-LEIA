"""
LeLamp Agent - Core agent class with function tools.

This is the main agent class that combines all function tool mixins.
Services are initialized by server.py and consumed from globals.
The agent adds AI-specific behavior on top of the hardware services.
"""

import os
import json
import logging
import subprocess
import asyncio
from typing import Optional

# Import function tool mixins
from lelamp.functions import (
    MotorFunctions,
    RGBFunctions,
    AnimationFunctions,
    AudioFunctions,
    TimerFunctions,
    WorkflowFunctions,
    SensorFunctions,
    SleepFunctions,
    VisionFunctions,
    SpotifyFunctions,
    LocationFunctions,
)

# Import theme for startup sound
from lelamp.service.theme import ThemeSound

# Import globals
import lelamp.globals as g

logger = logging.getLogger(__name__)


class LeLamp(
    MotorFunctions,
    RGBFunctions,
    AnimationFunctions,
    AudioFunctions,
    TimerFunctions,
    WorkflowFunctions,
    SensorFunctions,
    SleepFunctions,
    VisionFunctions,
    SpotifyFunctions,
    LocationFunctions,
):
    """
    Core LeLamp agent with all function tools.

    Combines hardware control (motors, RGB, audio) with AI capabilities
    through LiveKit's agent framework.
    """

    def __init__(self, config: dict = None) -> None:
        """
        Initialize LeLamp agent.

        Services are already initialized by server.py and available in globals.
        The agent consumes these services and adds AI-specific behavior.

        Args:
            config: Configuration dict. If None, uses global CONFIG.
        """
        # Use provided config or fall back to global
        if config is None:
            config = g.CONFIG
        self.config = config

        # Get services from globals (initialized by server.py)
        self.animation_service = g.animation_service
        self.rgb_service = g.rgb_service
        self.audio_service = g.audio_service
        self.theme_service = g.theme_service
        self.workflow_service = g.workflow_service
        self.spotify_service = g.spotify_service

        # Check if motors are available
        self.motors_enabled = self.animation_service is not None
        self.rgb_enabled = self.rgb_service is not None

        # Session references (set by pipeline)
        self.agent_session = None
        self.event_loop = None

        # VAD model reference for dynamic threshold adjustment
        self.vad_model = None
        self.vad_config = config.get("vad", {})
        self.vad_normal_threshold = self.vad_config.get("activation_threshold", 0.35)
        self.vad_music_threshold = self.vad_config.get("music_threshold", 0.7)

        # Sleep mode state
        self.is_sleeping = False

        # Activity tracking for idle timeout
        self.last_activity_time = None
        self.idle_timeout_task = None
        self.workflow_progression_task = None
        self.rgb_config = config.get("rgb", {})

        # Load personality instructions
        base_instructions = self._load_personality(config)
        super().__init__(instructions=base_instructions)

        # Agent-specific service setup
        self._setup_workflow_service()
        self._setup_spotify_service()
        self._setup_alarm_service()

        # Play startup animation and sound
        if self.theme_service:
            self.theme_service.play(ThemeSound.STARTUP)

        if self.animation_service:
            self.animation_service.dispatch("play", "wake_up")

        if self.rgb_service:
            default_anim = self.rgb_config.get("default_animation", "aura_glow")
            self.rgb_service.dispatch("animation", {
                "name": default_anim,
                "color": tuple(self.rgb_config.get("default_color", [255, 255, 255]))
            })

        logger.info("LeLamp agent initialized (using services from globals)")

    def _load_personality(self, config: dict) -> str:
        """Load personality/base instructions from config."""
        personality = config.get("personality", {})

        # Check if a character file is specified
        character_file = personality.get("character_file")
        character_data = {}

        if character_file and os.path.exists(character_file):
            with open(character_file, 'r', encoding='utf-8') as f:
                character_data = json.load(f)
            logger.info(f"Loaded character from {character_file}")

        # Load base instructions template
        instructions_file = personality.get("instructions_file")

        if instructions_file and os.path.exists(instructions_file):
            with open(instructions_file, 'r', encoding='utf-8') as f:
                base_instructions = f.read()
            logger.info(f"Loaded instructions from {instructions_file}")
        else:
            base_instructions = personality.get(
                "instructions",
                "You are {name} — {description}."
            )

        # Build template variables
        # robot_name comes from config, name comes from character file
        robot_name = personality.get("name", "LeLamp")
        template_vars = {
            "name": character_data.get("name") or robot_name,
            "robot_name": robot_name,
            "description": character_data.get("description") or personality.get("description", "a friendly robot lamp"),
            "speech_style": character_data.get("speech_style", ""),
            "ideals": character_data.get("ideals", ""),
            "flaws": character_data.get("flaws", ""),
            "bio": character_data.get("bio", ""),
            "city": config.get("location", {}).get("city", "your city"),
            "region": config.get("location", {}).get("region", "your region"),
            "country": config.get("location", {}).get("country", "your country"),
            "timezone": config.get("location", {}).get("timezone", "UTC"),
        }

        try:
            base_instructions = base_instructions.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template variable {e} not found, using as-is")

        return base_instructions

    def _setup_workflow_service(self):
        """Configure workflow service for this agent."""
        if self.workflow_service:
            self.workflow_service.set_agent(self)
            logger.info("Workflow service connected to agent")

    def _setup_spotify_service(self):
        """Configure Spotify service callbacks for this agent."""
        if self.spotify_service and self.animation_service:
            self.animation_service.connect_spotify_service(self.spotify_service)
            self.spotify_service.on_playback_state_change = self._on_spotify_playback_change
            logger.info("Spotify service connected to agent")

    def _setup_alarm_service(self):
        """Start alarm service if available."""
        if g.alarm_service:
            g.alarm_service.start()
            logger.info("Alarm service started")

    # =========================================================================
    # Volume Control
    # =========================================================================

    def _set_system_volume(self, volume_percent: int):
        """Set system playback volume."""
        try:
            subprocess.run(
                ["amixer", "sset", "PCM", f"{volume_percent}%"],
                capture_output=True, timeout=5
            )
        except Exception:
            pass

    def _set_system_microphone_volume(self, volume_percent: int):
        """Set system microphone/capture volume."""
        try:
            for control in ['Capture', 'ADC', 'ADC PCM']:
                subprocess.run(
                    ["amixer", "sset", control, f"{volume_percent}%"],
                    capture_output=True, timeout=5
                )
        except Exception:
            pass

    # =========================================================================
    # VAD Threshold Control
    # =========================================================================

    def _set_vad_threshold(self, threshold: float):
        """Dynamically adjust VAD activation threshold."""
        if self.vad_model and hasattr(self.vad_model, '_opts'):
            try:
                self.vad_model._opts.activation_threshold = threshold
                logger.debug(f"VAD threshold set to {threshold:.2f}")
            except Exception as e:
                logger.warning(f"Could not set VAD threshold: {e}")

    # =========================================================================
    # Activity Tracking
    # =========================================================================

    def _mark_activity(self):
        """Mark that activity occurred (for idle timeout)."""
        import time
        self.last_activity_time = time.time()

    def _reset_to_default_animation(self):
        """Reset RGB to default animation."""
        if self.is_sleeping:
            return

        default_anim = self.rgb_config.get("default_animation", "ripple")
        default_color = tuple(self.rgb_config.get("default_color", [0, 0, 150]))

        if self.rgb_service:
            self.rgb_service.dispatch("animation", {
                "name": default_anim,
                "color": default_color
            })

    async def _check_idle_timeout(self):
        """Background task that resets to default animation after idle."""
        import time
        idle_timeout = self.rgb_config.get("idle_timeout_seconds", 30)

        while True:
            try:
                await asyncio.sleep(5)

                if self.is_sleeping or self.last_activity_time is None:
                    continue

                if time.time() - self.last_activity_time >= idle_timeout:
                    self._reset_to_default_animation()
                    self.last_activity_time = None
            except Exception as e:
                logger.error(f"Idle timeout error: {e}")

    # =========================================================================
    # Spotify Callbacks
    # =========================================================================

    def _on_spotify_playback_change(self, is_playing: bool):
        """Handle Spotify playback state changes."""
        if is_playing:
            if self.animation_service:
                self.animation_service.enable_modifier("music")
                self.animation_service.start_dance_mode()
            self._set_vad_threshold(self.vad_music_threshold)
        else:
            if self.animation_service:
                self.animation_service.stop_dance_mode()
                self.animation_service.disable_modifier("music")
            self._set_vad_threshold(self.vad_normal_threshold)

    # =========================================================================
    # Workflow Progression
    # =========================================================================

    async def _check_workflow_progression(self):
        """Background task that checks workflow progression triggers."""
        while True:
            try:
                await asyncio.sleep(30)

                if self.is_sleeping or not self.workflow_service:
                    continue

                active_runs = self.workflow_service.db.get_active_runs()
                if not active_runs:
                    continue

                for run in active_runs:
                    await self._check_workflow_run(run)

            except Exception as e:
                logger.error(f"Workflow progression error: {e}")

    async def _check_workflow_run(self, run: dict):
        """Check a single workflow run for progression triggers."""
        from datetime import datetime
        from lelamp.service.workflows.db_manager import RunStatus

        run_id = run.get('run_id')
        workflow_id = run.get('workflow_id')
        started_at = run.get('started_at')
        trigger_type = run.get('trigger_type')
        trigger_data_str = run.get('trigger_data', '{}')

        # Validate alarm-triggered workflows
        if trigger_type == "alarm_trigger":
            try:
                trigger_data = json.loads(trigger_data_str) if isinstance(trigger_data_str, str) else trigger_data_str
                alarm_id = trigger_data.get('alarm_id')

                if alarm_id and g.alarm_service:
                    alarm = g.alarm_service.get_alarm(alarm_id)
                    if not alarm:
                        logger.info(f"Alarm {alarm_id} deleted - cancelling workflow {workflow_id}")
                        if self.workflow_service.current_run_id == run_id:
                            self.workflow_service.stop_workflow(RunStatus.CANCELLED)
                        else:
                            self.workflow_service.db.complete_run(run_id, RunStatus.CANCELLED)
                        return
            except Exception as e:
                logger.error(f"Error validating alarm workflow: {e}")

        # Check for time_interval progression triggers
        workflow_path = os.path.join(
            self.workflow_service.workflows_dir,
            workflow_id,
            "workflow.json"
        )

        if not os.path.exists(workflow_path):
            return

        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)

        for trigger in workflow_data.get("progression_triggers", []):
            if trigger.get("type") != "time_interval":
                continue

            interval_seconds = trigger.get("interval_seconds", 300)
            started_time = datetime.fromisoformat(
                started_at.replace('Z', '+00:00') if 'Z' in started_at else started_at
            )
            elapsed = (datetime.now(started_time.tzinfo) - started_time).total_seconds()
            intervals_passed = int(elapsed / interval_seconds)

            if intervals_passed > 0 and (elapsed % interval_seconds) < 35:
                action = trigger.get("action")
                if action == "prompt_agent" and self.agent_session and self.event_loop:
                    prompt = trigger.get("prompt", "Continue to next step")

                    def _progress():
                        self.agent_session.generate_reply(
                            instructions=f"WORKFLOW PROGRESSION: {prompt}"
                        )
                    self.event_loop.call_soon_threadsafe(_progress)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def stop(self):
        """
        Clean up agent-specific state.

        Note: Services are owned by server.py, not the agent.
        The agent just disconnects itself from callbacks.
        Services continue running for WebUI access.
        """
        logger.info("Stopping LeLamp agent...")

        # Disconnect Spotify callbacks
        if self.spotify_service:
            self.spotify_service.on_playback_state_change = None

        # Disconnect from workflow service
        if self.workflow_service:
            self.workflow_service.set_agent(None)

        # Clear session references
        self.agent_session = None
        self.event_loop = None

        logger.info("LeLamp agent stopped (services continue running)")
