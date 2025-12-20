"""
VisionService - Face detection and tracking for LeLamp
Provides face detection using OpenCV and supports face tracking mode
Optionally publishes camera feed to LiveKit for agent vision
"""

import cv2
import threading
import time
import os
import asyncio
from typing import Optional, Dict, Callable, Union
from dataclasses import dataclass
import logging
import numpy as np

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    rtc = None


@dataclass
class FaceData:
    """Data structure for face tracking information"""
    detected: bool
    position: tuple  # (x, y) normalized -1.0 to 1.0
    size: float  # 0.0 to 1.0, bigger = closer
    timestamp: float


class VisionService:
    """
    Vision service for face detection and tracking
    - Runs face detection in background thread
    - Provides face tracking mode that can control motors
    - Publishes face data updates to subscribers
    """

    def __init__(
        self,
        camera_index: Union[int, str] = 0,
        resolution: tuple = (320, 240),
        fps: int = 10,
        livekit_room: Optional[rtc.Room] = None,
        publish_video: bool = False,
        publish_image: bool = False,
        image_fps: float = 1.0,
        max_frame_size: int = 512
    ):
        # camera_index can be int (index) or str (device path like /dev/video0)
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.logger = logging.getLogger("service.VisionService")

        # Camera
        self.cap = None
        self._camera_thread = None
        self._running = False

        # LiveKit video publishing (continuous stream)
        self.livekit_room = livekit_room
        self.publish_video = publish_video and LIVEKIT_AVAILABLE and livekit_room is not None
        self.video_source: Optional[rtc.VideoSource] = None
        self.video_track: Optional[rtc.LocalVideoTrack] = None
        self._video_frame_count = 0

        # LiveKit image publishing (periodic stills)
        self.publish_image = publish_image
        self.image_fps = image_fps
        self.max_frame_size = max_frame_size
        self._image_callback: Optional[Callable] = None
        self._last_image_time = 0.0
        self._image_frame_count = 0

        # Face detection
        cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade from {cascade_path}")

        # Face data
        self.latest_face_data: Optional[FaceData] = None
        self._face_lock = threading.Lock()

        # Face tracking mode
        self._tracking_mode = False
        self._tracking_callback: Optional[Callable[[FaceData], None]] = None
        self._tracking_lock = threading.Lock()

        # Motor tracking callback (for face following)
        self._motor_tracking_enabled = False
        self._motor_tracking_callback: Optional[Callable[[float, float, bool], None]] = None

        # First face detection tracking (for theme sound)
        self._face_detected_once = False
        self._last_face_detected = False

    def start(self):
        """Start vision service"""
        if self._running:
            self.logger.warning("Vision service already running")
            return

        # Make sure any old thread is cleaned up
        if self._camera_thread and self._camera_thread.is_alive():
            self.logger.warning("Old camera thread still alive, waiting...")
            self._camera_thread.join(timeout=2.0)

        self._running = True
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        self.logger.info("Vision service started")

    def set_image_callback(self, callback: Callable[[bytes], None]):
        """
        Set callback for periodic image frames
        Callback receives base64-encoded JPEG image data
        """
        self._image_callback = callback
        self.logger.info(f"Image callback registered, will send at {self.image_fps} fps")

    async def setup_livekit_publishing(self):
        """Setup LiveKit video publishing (call after room is connected)"""
        if not self.publish_video or not self.livekit_room:
            return

        await self._setup_livekit_video_publishing()

    def stop(self):
        """Stop vision service"""
        self._running = False
        if self._camera_thread:
            self._camera_thread.join(timeout=2.0)
            self._camera_thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.logger.info("Vision service stopped")

    def _camera_loop(self):
        """Main camera capture and processing loop"""
        # Try to open camera
        # If camera_index is a string (device path), only try that path
        # If camera_index is an int, fall back to indices 0, 1, 2
        if isinstance(self.camera_index, str):
            candidates = [self.camera_index]
        else:
            candidates = [self.camera_index, 0, 1, 2]

        for cam_idx in candidates:
            test_cap = cv2.VideoCapture(cam_idx)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    self.logger.info(f"Camera opened at {cam_idx}")
                    self.cap = test_cap
                    break
                test_cap.release()
            else:
                test_cap.release()

        if self.cap is None:
            self.logger.error(f"No camera found! Tried: {candidates}")
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

            # Publish video frame to LiveKit if enabled
            self._publish_video_frame(frame)

            # Publish image frame periodically if enabled
            self._publish_image_frame(frame)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

            # Process face data
            face_data = self._process_faces(faces, frame.shape)

            # Update latest data
            with self._face_lock:
                self.latest_face_data = face_data

            # Play face detect sound on first detection (or when face reappears after absence)
            if face_data.detected and not self._last_face_detected:
                if not self._face_detected_once:
                    # First time ever detecting a face since service start
                    self._face_detected_once = True
                    try:
                        from lelamp.service.theme import get_theme_service, ThemeSound
                        theme = get_theme_service()
                        if theme:
                            theme.play(ThemeSound.FACE_DETECT)
                            self.logger.info("First face detected - played theme sound")
                    except Exception as e:
                        self.logger.warning(f"Could not play face detect sound: {e}")
            self._last_face_detected = face_data.detected

            # If tracking mode is enabled, notify callback
            if self._tracking_mode and self._tracking_callback and face_data.detected:
                try:
                    self._tracking_callback(face_data)
                except Exception as e:
                    self.logger.error(f"Error in tracking callback: {e}")

            # Motor tracking - send position to animation service
            if self._motor_tracking_enabled and self._motor_tracking_callback:
                try:
                    x, y = face_data.position
                    self._motor_tracking_callback(x, y, face_data.detected)
                except Exception as e:
                    self.logger.error(f"Error in motor tracking callback: {e}")

            time.sleep(frame_delay)

        # Cleanup
        if self.cap:
            self.cap.release()

    def _process_faces(self, faces, frame_shape) -> FaceData:
        """Process detected faces and return face data"""
        if len(faces) == 0:
            return FaceData(
                detected=False,
                position=(0.0, 0.0),
                size=0.0,
                timestamp=time.time()
            )

        # Get largest face (closest)
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]

        frame_h, frame_w = frame_shape[:2]

        # Calculate normalized position (-1 to 1)
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        pos_x = (face_center_x - frame_w // 2) / (frame_w / 2)
        pos_y = (face_center_y - frame_h // 2) / (frame_h / 2)

        # Calculate normalized size (0-1)
        size = (w * h) / (frame_w * frame_h)

        return FaceData(
            detected=True,
            position=(pos_x, pos_y),
            size=size,
            timestamp=time.time()
        )

    def get_face_data(self) -> Optional[FaceData]:
        """Get latest face tracking data"""
        with self._face_lock:
            return self.latest_face_data

    def enable_tracking_mode(self, callback: Callable[[FaceData], None]):
        """
        Enable face tracking mode with a callback
        The callback will be called with face data when a face is detected
        """
        with self._tracking_lock:
            self._tracking_mode = True
            self._tracking_callback = callback
        self.logger.info("Face tracking mode ENABLED")

    def disable_tracking_mode(self):
        """Disable face tracking mode"""
        with self._tracking_lock:
            self._tracking_mode = False
            self._tracking_callback = None
        self.logger.info("Face tracking mode DISABLED")

    def is_tracking_enabled(self) -> bool:
        """Check if tracking mode is enabled"""
        with self._tracking_lock:
            return self._tracking_mode

    def enable_motor_tracking(self, callback: Callable[[float, float, bool], None]):
        """
        Enable motor tracking mode - lamp follows detected faces.

        Args:
            callback: Function called with (x, y, detected) where x/y are -1 to 1
        """
        self._motor_tracking_callback = callback
        self._motor_tracking_enabled = True
        self.logger.info("Motor face tracking ENABLED")

    def disable_motor_tracking(self):
        """Disable motor tracking mode"""
        self._motor_tracking_enabled = False
        self._motor_tracking_callback = None
        self.logger.info("Motor face tracking DISABLED")

    def is_motor_tracking_enabled(self) -> bool:
        """Check if motor tracking is enabled"""
        return self._motor_tracking_enabled

    async def _setup_livekit_video_publishing(self):
        """Setup LiveKit video track publishing (called in async context)"""
        try:
            self.logger.info("Setting up LiveKit camera publishing...")

            # Wait for camera to open (with timeout)
            max_wait = 5.0  # Maximum 5 seconds
            elapsed = 0.0
            while elapsed < max_wait:
                if self.cap and self.cap.isOpened():
                    self.logger.info(f"Camera ready after {elapsed:.1f}s")
                    break
                await asyncio.sleep(0.1)
                elapsed += 0.1

            if not self.cap or not self.cap.isOpened():
                self.logger.error("Cannot publish to LiveKit: camera not opened after 5s timeout")
                return

            # Get actual resolution
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"Camera resolution: {width}x{height}")

            # Create video source and track
            self.video_source = rtc.VideoSource(width, height)
            self.logger.info("Created VideoSource")

            self.video_track = rtc.LocalVideoTrack.create_video_track(
                "robot_camera",
                self.video_source
            )
            self.logger.info("Created LocalVideoTrack: robot_camera")

            # Publish to room
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            publication = await self.livekit_room.local_participant.publish_track(self.video_track, options)
            self.logger.info(f"✓ Published camera track to LiveKit room: {width}x{height}, publication SID: {publication.sid}")
        except Exception as e:
            self.logger.error(f"Failed to setup LiveKit publishing: {e}", exc_info=True)
            self.publish_video = False

    def _publish_image_frame(self, frame):
        """Publish image frame periodically via callback (called from camera thread)"""
        if not self.publish_image or not self._image_callback:
            return

        current_time = time.time()
        time_since_last = current_time - self._last_image_time

        # Check if enough time has passed based on fps
        if time_since_last < (1.0 / self.image_fps):
            return

        self._last_image_time = current_time

        try:
            import base64
            import io
            from PIL import Image

            # Resize frame if needed
            height, width = frame.shape[:2]
            if max(width, height) > self.max_frame_size:
                scale = self.max_frame_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image and encode as JPEG
            pil_image = Image.fromarray(frame_rgb)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            jpeg_bytes = buffer.getvalue()

            # Encode to base64
            base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')

            # Call the callback with base64 data
            self._image_callback(base64_image)

            # Log every 10 images
            self._image_frame_count += 1
            if self._image_frame_count % 10 == 0:
                self.logger.info(f"Sent {self._image_frame_count} image frames to OpenAI")

        except Exception as e:
            self.logger.error(f"Error publishing image frame: {e}")

    def _publish_video_frame(self, frame):
        """Publish video frame to LiveKit (called from camera thread)"""
        if not self.publish_video or not self.video_source:
            return

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to ARGB for LiveKit
            height, width = frame_rgb.shape[:2]
            argb = np.zeros((height, width, 4), dtype=np.uint8)
            argb[:, :, 0] = 255  # Alpha channel
            argb[:, :, 1:4] = frame_rgb  # RGB

            # Create video frame
            video_frame = rtc.VideoFrame(
                width=width,
                height=height,
                type=rtc.VideoBufferType.RGBA,
                data=argb.tobytes()
            )

            # Publish frame (non-blocking)
            self.video_source.capture_frame(video_frame)

            # Log every 100 frames to verify publishing is working
            self._video_frame_count += 1
            if self._video_frame_count % 100 == 0:
                self.logger.info(f"Published {self._video_frame_count} video frames to LiveKit")
        except Exception as e:
            self.logger.error(f"Error publishing video frame to LiveKit: {e}")
