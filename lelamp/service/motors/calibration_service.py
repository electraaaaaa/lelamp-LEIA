"""
Calibration Service for WebUI-based motor calibration

This service provides a non-blocking, step-by-step calibration process
for use with the WebUI setup wizard.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from lelamp.follower import LeLampFollower, LeLampFollowerConfig
from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode

logger = logging.getLogger(__name__)


LAMP_ID = "lelamp"


class CalibrationService:
    """Service for interactive motor calibration via WebUI"""

    MOTOR_NAMES = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]

    def __init__(self, port: str):
        self.port = port
        self.robot_config = LeLampFollowerConfig(port=port, id=LAMP_ID)
        self.robot: Optional[LeLampFollower] = None

        # Calibration state
        self.current_step = "not_started"  # not_started, connected, homing, recording, complete
        self.current_motor_index = 0
        self.homing_offsets: Dict[str, int] = {}
        self.range_mins: Dict[str, int] = {}
        self.range_maxes: Dict[str, int] = {}
        self.calibration_data: Dict[str, MotorCalibration] = {}

    def connect(self) -> Dict[str, any]:
        """Connect to the robot without calibration"""
        try:
            if self.robot is None:
                self.robot = LeLampFollower(self.robot_config)

            if not self.robot.is_connected:
                self.robot.connect(calibrate=False)

            # Set all motors to position mode and disable torque for manual movement
            for motor in self.robot.bus.motors:
                self.robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.robot.bus.disable_torque()

            self.current_step = "connected"
            logger.info("Connected to robot for calibration")

            return {"success": True, "step": self.current_step}
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return {"success": False, "error": str(e)}

    def disconnect(self):
        """Disconnect from robot"""
        if self.robot and self.robot.is_connected:
            self.robot.disconnect()
            self.robot = None
        self.current_step = "not_started"

    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions of all motors.

        Also updates min/max tracking when in recording mode.
        """
        if not self.robot or not self.robot.is_connected:
            return {}

        try:
            # Always use raw positions during calibration (normalized won't work without calibration)
            positions = self.robot.bus.sync_read("Present_Position", normalize=False)

            # If in recording mode, update min/max
            if self.current_step == "recording":
                for motor, pos in positions.items():
                    if motor in self.range_mins:
                        self.range_mins[motor] = min(self.range_mins[motor], pos)
                    if motor in self.range_maxes:
                        self.range_maxes[motor] = max(self.range_maxes[motor], pos)

            return positions
        except Exception as e:
            logger.error(f"Error reading positions: {e}")
            return {}

    def record_homing(self) -> Dict[str, any]:
        """Record homing positions (middle of range)"""
        if not self.robot or not self.robot.is_connected:
            return {"success": False, "error": "Not connected"}

        try:
            # Record current positions as homing offsets
            self.homing_offsets = self.robot.bus.set_half_turn_homings()
            self.current_step = "homing"
            logger.info(f"Recorded homing offsets: {self.homing_offsets}")

            return {
                "success": True,
                "step": self.current_step,
                "homing_offsets": self.homing_offsets
            }
        except Exception as e:
            logger.error(f"Failed to record homing: {e}")
            return {"success": False, "error": str(e)}

    def start_range_recording(self) -> Dict[str, any]:
        """Start recording range of motion"""
        if not self.robot or not self.robot.is_connected:
            return {"success": False, "error": "Not connected"}

        try:
            self.current_step = "recording"
            self.current_motor_index = 0

            # Initialize min/max with current positions
            positions = self.robot.bus.sync_read("Present_Position", normalize=False)
            self.range_mins = positions.copy()
            self.range_maxes = positions.copy()

            logger.info("Started range recording")

            return {
                "success": True,
                "step": self.current_step,
                "instructions": "Move all joints through their full range of motion"
            }
        except Exception as e:
            logger.error(f"Failed to start range recording: {e}")
            return {"success": False, "error": str(e)}

    def record_ranges(self) -> Dict[str, any]:
        """Finalize the recorded min/max ranges for all motors"""
        if not self.robot or not self.robot.is_connected:
            return {"success": False, "error": "Not connected"}

        try:
            # Validate that we have recorded some range for each motor
            same_min_max = [
                motor for motor in self.range_mins
                if self.range_mins.get(motor) == self.range_maxes.get(motor)
            ]
            if same_min_max:
                return {
                    "success": False,
                    "error": f"Motors with no range recorded: {', '.join(same_min_max)}. Move each joint!"
                }

            logger.info(f"Recorded ranges - mins: {self.range_mins}, maxes: {self.range_maxes}")

            return {
                "success": True,
                "range_mins": self.range_mins,
                "range_maxes": self.range_maxes
            }
        except Exception as e:
            logger.error(f"Failed to record ranges: {e}")
            return {"success": False, "error": str(e)}

    def finalize_calibration(self) -> Dict[str, any]:
        """Finalize and save calibration data"""
        if not self.robot or not self.robot.is_connected:
            return {"success": False, "error": "Not connected"}

        if not self.homing_offsets or not self.range_mins or not self.range_maxes:
            return {"success": False, "error": "Incomplete calibration data"}

        try:
            # Build calibration data
            self.calibration_data = {}
            for motor, m in self.robot.bus.motors.items():
                self.calibration_data[motor] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=self.homing_offsets[motor],
                    range_min=self.range_mins[motor],
                    range_max=self.range_maxes[motor],
                )

            # Write calibration to robot
            self.robot.bus.write_calibration(self.calibration_data)

            # Save to file
            self.robot.calibration = self.calibration_data
            self.robot._save_calibration()

            self.current_step = "complete"
            calibration_path = self.robot.calibration_fpath

            logger.info(f"Calibration saved to {calibration_path}")

            # Play calibration complete theme sound
            try:
                from lelamp.service.theme import get_theme_service, ThemeSound
                theme = get_theme_service()
                if theme:
                    theme.play(ThemeSound.CALIBRATION_COMPLETE)
            except Exception as e:
                logger.warning(f"Could not play calibration complete sound: {e}")

            return {
                "success": True,
                "step": self.current_step,
                "calibration_path": str(calibration_path),
                "message": "Calibration complete!"
            }
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict[str, any]:
        """Get current calibration status"""
        return {
            "connected": self.robot is not None and self.robot.is_connected,
            "step": self.current_step,
            "current_motor": self.current_motor_index,
            "total_motors": len(self.MOTOR_NAMES),
            "motor_names": self.MOTOR_NAMES,
            "has_homing": bool(self.homing_offsets),
            "has_ranges": bool(self.range_mins and self.range_maxes)
        }
