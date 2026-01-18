"""
LeLamp Runtime Bootloader.

Minimal entrypoint that starts the WebUI server (which initializes all hardware
services) and then starts the appropriate AI pipeline if enabled.

Architecture:
  - server.py initializes ALL hardware services → stored in globals
  - Agent (lelamp.py) consumes services from globals
  - This file orchestrates startup and cleanup
"""
# Ignore warnings from protobuf to keep log clean
import warnings
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import atexit
import logging
import signal
import sys
import time

from dotenv import load_dotenv

# Load .env from user directory first (if exists), then fallback to repo .env
from lelamp.user_data import get_env_path
_env_path = get_env_path()
if _env_path.exists():
    load_dotenv(_env_path, override=True)
    print(f"Loaded .env from: {_env_path}")
else:
    load_dotenv()

from lelamp.service.alarm import AlarmService
from lelamp.service.metrics_service import get_metrics_service



def check_audio_hardware() -> tuple[bool, bool, str]:
    """
    Check if audio hardware (microphone/speaker) is available.

    Returns:
        (has_microphone, has_speaker, error_message)
    """
    import subprocess

    has_mic = False
    has_speaker = False
    error_msg = ""

    try:
        # Check for capture devices (microphones)
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "card" in result.stdout.lower():
            has_mic = True
    except Exception as e:
        error_msg = f"Could not check microphone: {e}"

    try:
        # Check for playback devices (speakers)
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "card" in result.stdout.lower():
            has_speaker = True
    except Exception as e:
        if error_msg:
            error_msg += f"; Could not check speaker: {e}"
        else:
            error_msg = f"Could not check speaker: {e}"

    return has_mic, has_speaker, error_msg

# Import WebUI server (this initializes all hardware services)
from lelamp.service.webui import start_webui_server

# Import global config and services
import lelamp.globals as g
CONFIG = g.CONFIG
save_config = g.save_config

# Initialize services that aren't part of hardware layer
g.alarm_service = AlarmService()
g.metrics_service = get_metrics_service()


def cleanup_services():
    """Clean up all services on exit."""
    logging.info("Cleaning up services...")

    # Stop LiveKit service first
    try:
        if g.livekit_service:
            g.livekit_service.stop()
            logging.info("LiveKit service stopped")
    except Exception as e:
        logging.error(f"Error stopping LiveKit service: {e}")

    # All hardware services are in globals (initialized by server.py)
    services = [
        (g.alarm_service, "Alarm service"),
        (g.vision_service, "Vision service"),
        (g.animation_service, "Animation service"),
        (g.audio_service, "Audio service"),
        (g.rgb_service, "RGB service"),
        (g.theme_service, "Theme service"),
        # (g.spotify_service, "Spotify service"),
    ]

    for service, name in services:
        try:
            if service and hasattr(service, 'stop'):
                service.stop()
                logging.info(f"{name} stopped")
        except Exception as e:
            logging.error(f"Error stopping {name}: {e}")

    # Wake service
    try:
        if g.wake_service and hasattr(g.wake_service, 'is_running') and g.wake_service.is_running():
            g.wake_service.stop()
            logging.info("Wake word service stopped")
    except Exception as e:
        logging.error(f"Error stopping wake word service: {e}")

    # Workflow service - stop any active workflow
    try:
        if g.workflow_service and hasattr(g.workflow_service, 'active_workflow') and g.workflow_service.active_workflow:
            g.workflow_service.stop_workflow()
            logging.info("Workflow service stopped")
    except Exception as e:
        logging.error(f"Error stopping workflow service: {e}")

    # Data collection service
    try:
        if g.datacollection_service:
            g.datacollection_service.stop()
            logging.info("Data collection service stopped")
    except Exception as e:
        logging.error(f"Error stopping data collection service: {e}")

    logging.info("Cleanup complete")


def signal_handler(signum, frame):
    """Handle Ctrl-C gracefully."""
    print("\n\nShutting down gracefully...")
    cleanup_services()
    sys.exit(0)


if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup_services)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start WebUI server first - this initializes ALL hardware services
    # Services are stored in globals and used by both WebUI and Agent
    # Looking into this function will help you understand how the code works
    start_webui_server()

    # Check if agent is enabled
    agent_config = CONFIG.get("agent", {})
    agent_enabled = agent_config.get("enabled", True)

    has_mic, has_speaker, audio_error = check_audio_hardware()

    if not has_mic:
        logging.warning("No microphone detected - disabling voice agent")
        agent_enabled = False

    elif not has_speaker:
        logging.warning("No speaker detected - audio output may not work")

    # Keep the process running (for WebUI) but don't start agent
    def keep_alive(signum, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, keep_alive)
    signal.signal(signal.SIGTERM, keep_alive)

    if not has_mic:
        logging.warning("No microphone detected - disabling voice agent")
        agent_enabled = False

        print("\n" + "="*60)
        print("NO MICROPHONE DETECTED")
        print("="*60)
        print("The voice agent requires a microphone to function.")
        print("")
        print("Please connect USB audio hardware and restart the service.")
        print("")
        print("To check available devices:")
        print("  arecord -l   # List capture devices")
        print("  aplay -l     # List playback devices")
        print("")
        print("WebUI is still available for configuration.")
        print("="*60 + "\n")

    elif not has_speaker:
        logging.warning("No speaker detected - audio output may not work")
        print("\n" + "="*60)
        print("WARNING: No speaker detected")
        print("="*60)
        print("Audio output may not work. Connect USB audio hardware.")
        print("="*60 + "\n")

    if not agent_enabled:
        # Only show "disabled" message if agent was explicitly disabled in config
        # (not if credentials are just missing)
        if agent_config.get("enabled", True) is False:
            logging.warning("AI Agent disabled in config.yaml")
            print("\n" + "="*60)
            print("AI AGENT DISABLED")
            print("="*60)
            print("The AI agent is disabled in config.yaml")
            print("")
            print("To enable: set agent.enabled = true and restart")
            print("="*60 + "\n")

        # Keep the process running (for WebUI) but don't start agent
        def keep_alive(signum, frame):
            print("\nShutting down...")
            sys.exit(0)

        signal.signal(signal.SIGINT, keep_alive)
        signal.signal(signal.SIGTERM, keep_alive)

        print("Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
    else:
        # Start appropriate pipeline
        if pipeline_type == "local":
            print("\nStarting LOCAL voice pipeline...")
            asyncio.run(run_local_pipeline())
        else:
            # livekit-realtime or legacy livekit
            provider = CONFIG.get("pipeline", {}).get("provider", "openai")
            print(f"\nStarting LIVEKIT-REALTIME voice pipeline (provider={provider})...")

            # Run LiveKit agent directly from main thread
            # (agents.cli.run_app needs to run in main thread for signal handlers)
            if g.livekit_service:
                from livekit import agents

                # Set environment variables for LiveKit
                creds = g.livekit_service.credentials
                if creds:
                    os.environ["LIVEKIT_URL"] = creds.url
                    os.environ["LIVEKIT_API_KEY"] = creds.api_key
                    os.environ["LIVEKIT_API_SECRET"] = creds.api_secret

                    # Set provider-specific API key
                    from lelamp.service.livekit.livekit_service import PROVIDER_API_KEYS
                    provider_key = creds.current_provider_key
                    env_var = PROVIDER_API_KEYS.get(creds.provider, f"{creds.provider.upper()}_API_KEY")
                    os.environ[env_var] = provider_key
                    logging.info(f"Set {env_var} for provider: {creds.provider}")

                # Get worker options and run directly in main thread
                worker_options = g.livekit_service.get_worker_options()
                if worker_options:
                    try:
                        agents.cli.run_app(worker_options)
                    except KeyboardInterrupt:
                        print("\nShutting down...")
                    except Exception as e:
                        logging.error(f"LiveKit agent error: {e}", exc_info=True)
                else:
                    logging.error("Failed to get worker options")
            else:
                logging.error("LiveKit service not available")
                print("Error: LiveKit service not initialized")


    
