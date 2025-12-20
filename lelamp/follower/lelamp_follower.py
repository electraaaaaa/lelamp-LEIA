#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import os
import yaml
from functools import cached_property
from typing import Any
from pathlib import Path

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.motors_bus import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from .config_lelamp_follower import LeLampFollowerConfig

# Import user data for calibration paths and config
from lelamp.user_data import get_calibration_path, save_calibration, USER_CALIBRATION_DIR, get_config_path

def _load_motor_config():
    """Load motor configuration from ~/.lelamp/config.yaml"""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

logger = logging.getLogger(__name__)


class LeLampFollower(Robot):
    """
    LeLamp Follower Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = LeLampFollowerConfig
    name = "lelamp_follower"

    def __init__(self, config: LeLampFollowerConfig):
        # Set up calibration directory using user_data (~/.lelamp/calibration/)
        # This ensures calibration survives reinstalls
        self._calibration_dir_override = USER_CALIBRATION_DIR
        self._calibration_dir_override.mkdir(parents=True, exist_ok=True)

        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "base_yaw": Motor(1, "sts3215", norm_mode_body),
                "base_pitch": Motor(2, "sts3215", norm_mode_body),
                "elbow_pitch": Motor(3, "sts3215", norm_mode_body),
                "wrist_roll": Motor(4, "sts3215", norm_mode_body),
                "wrist_pitch": Motor(5, "sts3215", norm_mode_body),
            },
            calibration=self.calibration,
        )

        self.cameras = make_cameras_from_configs(self.config.cameras)

    @property
    def calibration_dir(self) -> Path:
        """Override to use local calibration directory instead of cache."""
        return self._calibration_dir_override

    @calibration_dir.setter
    def calibration_dir(self, value: Path) -> None:
        """Setter to allow parent class to set calibration_dir, but we ignore it and use local."""
        # Ignore the value and keep using our local calibration directory
        pass

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._motors_ft}

    @property
    def is_connected(self) -> bool:
        # Print each connected check
        status = self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

        return status

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
         
        self.bus.disable_torque()
        # Always prompt user for calibration choice
        user_input = input(
            f"Press ENTER for default calibration setup, or type 'c' and press ENTER to modify config: "
        )
        if user_input.strip().lower() == "c":
            # User wants to choose a calibration file
            if self.calibration:
                logger.info(f"Using provided calibration file associated with the id {self.id}")
                self.bus.write_calibration(self.calibration)
                return
            else:
                logger.warning(f"No calibration file found for id {self.id}. Running default calibration instead.")
        
        # Default calibration setup
        logger.info(f"\nRunning default calibration of {self}")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        # Load motor presets from config.yaml
        config = _load_motor_config()
        preset_name = config.get("motor_preset", "Gentle")
        presets = config.get("motor_presets", {})
        preset = presets.get(preset_name, {})
        defaults = preset.get("default", {
            "p_coefficient": 8,
            "i_coefficient": 0,
            "d_coefficient": 10,
            "torque_limit": 200
        })

        logger.info(f"Configuring motors with preset: {preset_name}")

        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                # Get per-joint settings, falling back to defaults
                joint_config = preset.get(motor, {})
                p_coeff = joint_config.get("p_coefficient", defaults.get("p_coefficient", 8))
                i_coeff = joint_config.get("i_coefficient", defaults.get("i_coefficient", 0))
                d_coeff = joint_config.get("d_coefficient", defaults.get("d_coefficient", 10))
                torque_limit = joint_config.get("torque_limit", defaults.get("torque_limit", 200))

                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, p_coeff)
                self.bus.write("I_Coefficient", motor, i_coeff)
                self.bus.write("D_Coefficient", motor, d_coeff)
                self.bus.write("Torque_Limit", motor, torque_limit)
                self.bus.write("Velocity_closed_loop_P_proportional_coefficient", motor, 50)
                self.bus.write("Velocity_closed_loop_I_integral_coefficient", motor, 200)

                motor_id = self.bus.motors[motor].id
                if motor_id == 1:
                    deadzone_value = 3
                else:
                    deadzone_value = 1
                self.bus.write("CW_Dead_Zone", motor, deadzone_value)
                self.bus.write("CCW_Dead_Zone", motor, deadzone_value)

                logger.debug(f"  {motor}: P={p_coeff}, I={i_coeff}, D={d_coeff}, Torque={torque_limit}")

    def apply_preset(self, preset_name: str = None) -> bool:
        """
        Apply a motor preset at runtime (without full reconfiguration).

        Args:
            preset_name: Name of preset to apply (Gentle, Normal, Sport).
                        If None, reads from config.yaml motor_preset value.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_connected:
            logger.error("Cannot apply preset: robot not connected")
            return False

        # Load config
        config = _load_motor_config()
        if preset_name is None:
            preset_name = config.get("motor_preset", "Gentle")

        presets = config.get("motor_presets", {})
        if preset_name not in presets:
            logger.error(f"Preset '{preset_name}' not found. Available: {list(presets.keys())}")
            return False

        preset = presets[preset_name]
        defaults = preset.get("default", {
            "p_coefficient": 8,
            "i_coefficient": 0,
            "d_coefficient": 10,
            "torque_limit": 200
        })

        logger.info(f"Applying motor preset: {preset_name}")

        try:
            for motor in self.bus.motors:
                # Get per-joint settings, falling back to defaults
                joint_config = preset.get(motor, {})
                p_coeff = joint_config.get("p_coefficient", defaults.get("p_coefficient", 8))
                i_coeff = joint_config.get("i_coefficient", defaults.get("i_coefficient", 0))
                d_coeff = joint_config.get("d_coefficient", defaults.get("d_coefficient", 10))
                torque_limit = joint_config.get("torque_limit", defaults.get("torque_limit", 200))

                self.bus.write("P_Coefficient", motor, p_coeff)
                self.bus.write("I_Coefficient", motor, i_coeff)
                self.bus.write("D_Coefficient", motor, d_coeff)
                self.bus.write("Torque_Limit", motor, torque_limit)

                logger.debug(f"  {motor}: P={p_coeff}, I={i_coeff}, D={d_coeff}, Torque={torque_limit}")

            return True
        except Exception as e:
            logger.error(f"Error applying preset: {e}")
            return False

    def get_available_presets(self) -> list:
        """Get list of available motor presets."""
        config = _load_motor_config()
        presets = config.get("motor_presets", {})
        return list(presets.keys())

    def enable_pushable_mode(self) -> bool:
        """
        Enable pushable mode - lamp becomes compliant and can be moved by hand.
        Uses Gentle preset for soft feel, and continuously updates goal position
        to match where user moves it.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_connected:
            logger.error("Cannot enable pushable mode: robot not connected")
            return False

        config = _load_motor_config()
        pushable_config = config.get("pushable_mode", {})
        preset_name = pushable_config.get("preset", "Gentle")

        logger.info(f"Enabling pushable mode with {preset_name} preset")

        # Apply the gentle preset for compliant feel
        success = self.apply_preset(preset_name)
        if success:
            self._pushable_mode = True
            # Store current position as the initial goal
            try:
                self._held_position = self.bus.sync_read("Present_Position")
            except Exception:
                self._held_position = None

        return success

    def update_goal_to_current_position(self) -> bool:
        """
        Read current position and set it as the new goal position.
        Call this periodically in pushable mode so the lamp doesn't fight back.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_connected:
            return False

        try:
            # Read where the servos actually are
            current_pos = self.bus.sync_read("Present_Position")
            # Set that as the new goal so they don't fight back
            self.bus.sync_write("Goal_Position", current_pos)
            self._held_position = current_pos
            return True
        except Exception as e:
            logger.debug(f"Error updating goal position: {e}")
            return False

    def disable_pushable_mode(self, return_to_idle: bool = None) -> bool:
        """
        Disable pushable mode - lamp returns to normal LLM-controlled operation.

        Args:
            return_to_idle: If True, return to idle pose. If None, uses config setting.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_connected:
            logger.error("Cannot disable pushable mode: robot not connected")
            return False

        config = _load_motor_config()
        pushable_config = config.get("pushable_mode", {})

        if return_to_idle is None:
            return_to_idle = pushable_config.get("return_to_idle", True)

        # Get the normal operating preset
        normal_preset = config.get("motor_preset", "Normal")

        logger.info(f"Disabling pushable mode, returning to {normal_preset} preset")

        # Apply normal preset
        success = self.apply_preset(normal_preset)
        if success:
            self._pushable_mode = False
            self._held_position = None

        return success

    def is_pushable_mode(self) -> bool:
        """Check if pushable mode is currently enabled."""
        return getattr(self, '_pushable_mode', False)

    def get_held_position(self) -> dict:
        """Get the current held position in pushable mode."""
        if not self.is_connected:
            return {}
        try:
            return self.bus.sync_read("Present_Position")
        except Exception:
            return {}

    def update_held_position(self) -> dict:
        """Update the held position to current position (call after user moves lamp)."""
        pos = self.get_held_position()
        self._held_position = pos
        return pos

    def setup_motors(self, min_voltage_limit: int = 45, max_voltage_limit: int = 80) -> None:
        """
        Setup motors with voltage limits.

        Args:
            min_voltage_limit: Minimum voltage limit (default 45 = 4.5V for 7.4V servos)
            max_voltage_limit: Maximum voltage limit (default 80 = 8.0V for 7.4V servos)
                              For 12V servos, use min=95 (9.5V), max=135 (13.5V)
        """
        # NOTE: Voltage limit setting is commented out - set these manually if needed
        # print(f"Setting voltage limits: Min={min_voltage_limit/10}V, Max={max_voltage_limit/10}V")

        # Get motors in ID order (1→5) for clearer setup
        motors_by_id = sorted(self.bus.motors.keys(), key=lambda m: self.bus.motors[m].id)
        total_motors = len(motors_by_id)

        print(f"\n{'='*60}")
        print("Motor ID Setup")
        print(f"{'='*60}")
        print("This will configure each motor, one at a time.")
        print("Connect ONLY the motor being configured to the controller.")
        print(f"{'='*60}")
        print("")
        print("LeLamp Motor Layout:")
        print("")
        print("                    [LAMP HEAD]")
        print("                         |")
        print("              ID5: wrist_pitch (nod)")
        print("                         |")
        print("              ID4: wrist_roll (twist)")
        print("                         |")
        print("              ID3: elbow_pitch (bend)")
        print("                         |")
        print("              ID2: base_pitch (tilt)")
        print("                         |")
        print("              ID1: base_yaw (rotate)")
        print("                         |")
        print("                      [BASE]")
        print("")
        print(f"{'='*60}\n")

        # Use /dev/tty for input to handle stdin redirection issues
        import sys
        try:
            tty = open('/dev/tty', 'r')
        except OSError:
            tty = sys.stdin

        for i, motor in enumerate(motors_by_id, 1):
            motor_id = self.bus.motors[motor].id
            print(f"\n[{i}/{total_motors}] Motor ID {motor_id}: {motor}")
            print("-" * 40)
            print(f"Connect ONLY the motor that will be ID {motor_id} ({motor}) and press Enter...", end='', flush=True)
            tty.readline()
            self.bus.setup_motor(motor)
            # WARNING: Setting voltage limits can damage motors if values don't match your servo specs
            # Uncomment and verify voltage values match your servos before using:
            # self.bus.write("Min_Voltage_Limit", motor, min_voltage_limit, normalize=False)
            # self.bus.write("Max_Voltage_Limit", motor, max_voltage_limit, normalize=False)
            print(f"✓ Motor '{motor}' configured with ID {motor_id}")

            # Offer to calibrate center position
            print("")
            print("  ⚠ CALIBRATE CENTER POSITION")
            print("  This will move the motor to center position (2048).")
            print("")
            print("  ⚡ WARNING: THE MOTOR WILL SPIN! ⚡")
            print("")
            print("  Make sure the motor is:")
            print("    - NOT attached to any LeLamp parts")
            print("    - Free to spin without obstruction")
            print("    - Clear of fingers and obstacles")
            print("")
            print("  Move motor to center position? [y/N]: ", end='', flush=True)
            response = tty.readline().strip().lower()

            if response == 'y':
                try:
                    # Move motor to center position (2048) using raw register writes
                    # We bypass self.bus.write() for Goal_Position because it requires calibration
                    print("  ⚡ Motor spinning to center position...")
                    import time

                    # Get the motor's ID for direct register access
                    motor_id = self.bus.motors[motor].id

                    # Use the underlying packet handler for raw writes
                    # Torque_Enable is at address 40 (1 byte)
                    # Goal_Position is at address 42 (2 bytes)
                    self.bus.port_handler.setPacketTimeoutMillis(1000)

                    # Enable torque (address 40, value 1)
                    comm_result, error = self.bus.packet_handler.write1ByteTxRx(
                        self.bus.port_handler, motor_id, 40, 1
                    )
                    time.sleep(0.05)

                    # Write goal position 2048 (address 42, 2 bytes)
                    comm_result, error = self.bus.packet_handler.write2ByteTxRx(
                        self.bus.port_handler, motor_id, 42, 2048
                    )

                    time.sleep(1)  # Wait for motor to reach position

                    # Disable torque (address 40, value 0)
                    comm_result, error = self.bus.packet_handler.write1ByteTxRx(
                        self.bus.port_handler, motor_id, 40, 0
                    )

                    print(f"  ✓ Motor moved to center position (2048)")
                except Exception as e:
                    print(f"  ✗ Center position failed: {e}")
            else:
                print("  → Skipped center calibration")

        if tty is not sys.stdin:
            tty.close()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        # Read LEDs
        start = time.perf_counter()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read LEDs: {dt_ms:.1f}ms")
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)


        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
