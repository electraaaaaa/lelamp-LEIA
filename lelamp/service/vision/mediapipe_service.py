"""
MediaPipeVisionService - Advanced face detection and tracking using Google MediaPipe
Provides face mesh with 468 landmarks, head pose estimation, and gaze tracking
"""

import cv2
import threading
import time
import os
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
import logging

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


@dataclass
class MediaPipeFaceData:
    """Enhanced face data from MediaPipe"""
    detected: bool
    position: tuple  # (x, y) normalized -1.0 to 1.0 (face center)
    size: float  # 0.0 to 1.0, bigger = closer
    head_pose: dict  # {'pitch': float, 'yaw': float, 'roll': float} in degrees
    timestamp: float


class MediaPipeVisionService:
    """
    Vision service using MediaPipe Face Mesh
    - More accurate face detection than Haar Cascade
    - Works at angles and partial occlusion
    - Provides head pose angles for better motor control
    - Supports face tracking mode with callback
    """

    def __init__(
        self,
        camera_index: int = 0,
        resolution: tuple = (320, 240),
        fps: int = 10,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is not installed. Install with: pip install mediapipe\n"
                "Note: Requires Python 3.8-3.12 (not 3.13+)"
            )

        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.logger = logging.getLogger("service.MediaPipeVisionService")

        # Camera
        self.cap = None
        self._camera_thread = None
        self._running = False

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Include iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Face data
        self.latest_face_data: Optional[MediaPipeFaceData] = None
        self._face_lock = threading.Lock()

        # Face tracking mode
        self._tracking_mode = False
        self._tracking_callback: Optional[Callable[[MediaPipeFaceData], None]] = None
        self._tracking_lock = threading.Lock()

    def start(self):
        """Start vision service"""
        if self._running:
            self.logger.warning("MediaPipe vision service already running")
            return

        self._running = True
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        self.logger.info("MediaPipe vision service started")

    def stop(self):
        """Stop vision service"""
        self._running = False
        if self._camera_thread:
            self._camera_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        if self.face_mesh:
            self.face_mesh.close()
        self.logger.info("MediaPipe vision service stopped")

    def _camera_loop(self):
        """Main camera capture and processing loop"""
        # Try to open camera
        for cam_idx in [self.camera_index, 0, 1, 2]:
            test_cap = cv2.VideoCapture(cam_idx)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    self.logger.info(f"Camera opened at index {cam_idx}")
                    self.cap = test_cap
                    break
                test_cap.release()
            else:
                test_cap.release()

        if self.cap is None:
            self.logger.error("No camera found!")
            return

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        frame_delay = 1.0 / self.fps

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # MediaPipe requires RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)

            # Extract face data
            face_data = self._process_mediapipe_results(results, frame.shape)

            # Update latest data
            with self._face_lock:
                self.latest_face_data = face_data

            # If tracking mode is enabled, notify callback
            if self._tracking_mode and self._tracking_callback and face_data.detected:
                try:
                    self._tracking_callback(face_data)
                except Exception as e:
                    self.logger.error(f"Error in tracking callback: {e}")

            time.sleep(frame_delay)

        # Cleanup
        if self.cap:
            self.cap.release()

    def _process_mediapipe_results(self, results, frame_shape) -> MediaPipeFaceData:
        """Process MediaPipe results and extract face data"""
        if not results.multi_face_landmarks:
            return MediaPipeFaceData(
                detected=False,
                position=(0.0, 0.0),
                size=0.0,
                head_pose={'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
                timestamp=time.time()
            )

        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        frame_h, frame_w = frame_shape[:2]

        # Get nose tip (landmark 1) as face center
        nose = face_landmarks.landmark[1]
        face_center_x = nose.x * frame_w
        face_center_y = nose.y * frame_h

        # Calculate normalized position (-1 to 1)
        pos_x = (face_center_x - frame_w / 2) / (frame_w / 2)
        pos_y = (face_center_y - frame_h / 2) / (frame_h / 2)

        # Calculate face size (distance between eyes as proxy)
        left_eye = face_landmarks.landmark[33]  # Left eye outer corner
        right_eye = face_landmarks.landmark[263]  # Right eye outer corner
        eye_distance = np.sqrt(
            ((left_eye.x - right_eye.x) * frame_w) ** 2 +
            ((left_eye.y - right_eye.y) * frame_h) ** 2
        )
        # Normalize size (typical eye distance is ~60-80 pixels at 320x240)
        size = min(1.0, eye_distance / 100.0)

        # Calculate head pose angles
        head_pose = self._calculate_head_pose(face_landmarks, frame_w, frame_h)

        return MediaPipeFaceData(
            detected=True,
            position=(pos_x, pos_y),
            size=size,
            head_pose=head_pose,
            timestamp=time.time()
        )

    def _calculate_head_pose(self, landmarks, frame_w, frame_h) -> dict:
        """
        Calculate head pose (pitch, yaw, roll) from facial landmarks
        Uses simplified 6-point model for speed
        """
        # Key points for head pose estimation
        # Nose tip, chin, left eye, right eye, left mouth, right mouth
        image_points = np.array([
            (landmarks.landmark[1].x * frame_w, landmarks.landmark[1].y * frame_h),    # Nose tip
            (landmarks.landmark[152].x * frame_w, landmarks.landmark[152].y * frame_h),  # Chin
            (landmarks.landmark[33].x * frame_w, landmarks.landmark[33].y * frame_h),   # Left eye
            (landmarks.landmark[263].x * frame_w, landmarks.landmark[263].y * frame_h),  # Right eye
            (landmarks.landmark[61].x * frame_w, landmarks.landmark[61].y * frame_h),   # Left mouth
            (landmarks.landmark[291].x * frame_w, landmarks.landmark[291].y * frame_h),  # Right mouth
        ], dtype="double")

        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -330.0, -65.0),   # Chin
            (-225.0, 170.0, -135.0),# Left eye
            (225.0, 170.0, -135.0), # Right eye
            (-150.0, -150.0, -125.0),# Left mouth
            (150.0, -150.0, -125.0) # Right mouth
        ])

        # Camera internals (approximate for small resolution)
        focal_length = frame_w
        center = (frame_w / 2, frame_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

        # Convert rotation vector to euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        return {
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll)
        }

    def get_face_data(self) -> Optional[MediaPipeFaceData]:
        """Get latest face tracking data"""
        with self._face_lock:
            return self.latest_face_data

    def enable_tracking_mode(self, callback: Callable[[MediaPipeFaceData], None]):
        """Enable face tracking mode with callback"""
        with self._tracking_lock:
            self._tracking_mode = True
            self._tracking_callback = callback
        self.logger.info("MediaPipe face tracking mode ENABLED")

    def disable_tracking_mode(self):
        """Disable face tracking mode"""
        with self._tracking_lock:
            self._tracking_mode = False
            self._tracking_callback = None
        self.logger.info("MediaPipe face tracking mode DISABLED")

    def is_tracking_enabled(self) -> bool:
        """Check if tracking mode is enabled"""
        with self._tracking_lock:
            return self._tracking_mode
