import os
import csv
import time
from typing import Any, List
from ..base import ServiceBase
from lelamp.follower import LeLampFollowerConfig, LeLampFollower

LAMP_ID = "lelamp"


class MotorsService(ServiceBase):
    def __init__(self, port: str, fps: int = 30):
        super().__init__("motors")
        self.port = port
        self.fps = fps
        self.robot_config = LeLampFollowerConfig(port=port, id=LAMP_ID)
        self.robot: LeLampFollower = None
        self.recordings_dir = os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
    
    def start(self):
        super().start()

        # Check if motors are enabled in config
        import lelamp.globals as g
        motors_config = g.CONFIG.get("motors", {})
        if not motors_config.get("enabled", True):
            self.logger.info("Motors disabled in config - motors service not started")
            return

        self.robot = LeLampFollower(self.robot_config)

        # Check if calibration file exists before connecting
        if not os.path.exists(self.robot.calibration_fpath):
            # Set global flag for post-assembly setup
            g.calibration_required = True
            g.calibration_path = self.robot.calibration_fpath
            self.logger.warning(f"Calibration required: {self.robot.calibration_fpath}")
            self.logger.info("Motors service will remain disabled until calibration is complete")
            self.logger.info(f"Complete setup wizard at WebUI or run: uv run -m lelamp.calibrate --port {self.port}")
            # Don't raise error - just return without connecting
            return

        self.robot.connect(calibrate=True)
        self.logger.info(f"Motors service connected to {self.port}")

    def stop(self, timeout: float = 5.0):
        if self.robot:
            self.robot.disconnect()
            self.robot = None
        super().stop(timeout)
    
    def handle_event(self, event_type: str, payload: Any):
        if event_type == "play":
            self._handle_play(payload)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _handle_play(self, recording_name: str):
        """Play a recording by name"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return

        csv_filename = f"{recording_name}.csv"
        csv_path = os.path.join(self.recordings_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            self.logger.error(f"Recording not found: {csv_path}")
            return
        
        try:
            with open(csv_path, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                actions = list(csv_reader)
            
            self.logger.info(f"Playing {len(actions)} actions from {recording_name}")
            
            for row in actions:
                t0 = time.perf_counter()
                
                # Extract action data (exclude timestamp column)
                action = {key: float(value) for key, value in row.items() if key != 'timestamp'}
                self.robot.send_action(action)
                
                # Use time.sleep instead of busy_wait to avoid blocking other threads
                sleep_time = 1.0 / self.fps - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.logger.info(f"Finished playing recording: {recording_name}")
            
        except Exception as e:
            self.logger.error(f"Error playing recording {recording_name}: {e}")
    
    def get_available_recordings(self) -> List[str]:
        """Get list of available recording names"""
        if not os.path.exists(self.recordings_dir):
            return []

        recordings = []

        for filename in os.listdir(self.recordings_dir):
            if filename.endswith(".csv"):
                recording_name = filename[:-4]  # Remove .csv
                recordings.append(recording_name)

        return sorted(recordings)