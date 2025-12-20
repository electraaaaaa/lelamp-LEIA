"""
Motor calibration API endpoints.

Provides endpoints for the motor calibration workflow.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from api.deps import load_config, save_config
import lelamp.globals as g

router = APIRouter()

# Global calibration service instance for this module
_calibration_service = None


@router.post("/start")
async def start_calibration():
    """Start motor calibration process."""
    global _calibration_service

    try:
        from lelamp.service.motors.calibration_service import CalibrationService

        config = load_config()
        port = config.get("motors", {}).get("port", "/dev/lelamp")

        _calibration_service = CalibrationService(port=port)
        result = _calibration_service.connect()

        if result["success"]:
            g.calibration_in_progress = True

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/status")
async def get_calibration_status():
    """Get current calibration status."""
    if _calibration_service is None:
        return {
            "connected": False,
            "step": "not_started",
            "calibration_required": g.calibration_required,
        }

    status = _calibration_service.get_status()
    status["calibration_required"] = g.calibration_required

    # Include min/max ranges if they've been recorded
    status["range_mins"] = _calibration_service.range_mins
    status["range_maxs"] = _calibration_service.range_maxes
    status["homing_offsets"] = _calibration_service.homing_offsets

    return status


@router.get("/positions")
async def get_calibration_positions():
    """Get current motor positions during calibration."""
    if _calibration_service is None:
        return {"success": False, "error": "Calibration not started"}

    positions = _calibration_service.get_current_positions()
    return {
        "success": True,
        "positions": positions,
        "step": _calibration_service.current_step,
        "range_mins": _calibration_service.range_mins,
        "range_maxs": _calibration_service.range_maxes,
    }


@router.post("/record-homing")
async def record_homing():
    """Record homing positions (center/zero point)."""
    if _calibration_service is None:
        return {"success": False, "error": "Calibration not started"}

    return _calibration_service.record_homing()


@router.post("/start-range")
async def start_range_recording():
    """Start recording range of motion."""
    if _calibration_service is None:
        return {"success": False, "error": "Calibration not started"}

    return _calibration_service.start_range_recording()


@router.post("/record-ranges")
async def record_ranges():
    """Record min/max range for all motors."""
    if _calibration_service is None:
        return {"success": False, "error": "Calibration not started"}

    return _calibration_service.record_ranges()


@router.post("/finalize")
async def finalize_calibration():
    """Finalize and save calibration."""
    global _calibration_service

    if _calibration_service is None:
        return {"success": False, "error": "Calibration not started"}

    result = _calibration_service.finalize_calibration()

    if result["success"]:
        # Load and update config
        config = load_config()

        # Enable motors in config
        if "motors" not in config:
            config["motors"] = {}
        config["motors"]["enabled"] = True

        # Mark calibration step as complete
        if "setup" not in config:
            config["setup"] = {}
        if "steps_completed" not in config["setup"]:
            config["setup"]["steps_completed"] = {}
        config["setup"]["steps_completed"]["motor_calibration"] = True

        save_config(config)

        # Clear calibration flags
        g.calibration_required = False
        g.calibration_in_progress = False

        # Disconnect calibration service
        _calibration_service.disconnect()
        _calibration_service = None

    return result


@router.post("/cancel")
async def cancel_calibration():
    """Cancel calibration and disconnect."""
    global _calibration_service

    if _calibration_service:
        _calibration_service.disconnect()
        _calibration_service = None

    g.calibration_in_progress = False

    return {"success": True}
