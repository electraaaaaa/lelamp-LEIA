"""
MediaPipeVisionService - Advanced face detection and tracking using Google MediaPipe
Provides face mesh with 468 landmarks, head pose estimation, and gaze tracking
"""
from typing import Optional, Callable, List
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


@dataclass
class MediaPipeHandData:
    detected: bool
    handedness: str       # "Left" or "Right"
    position: tuple       # (x, y) wrist
    gesture: str          # "None", "Fist", "Open_Palm", "Victory", "Pointer"
    fingers_up: list      # [thumb to little finger] (bool)
    timestamp: float
    is_pinching: bool = False      # 是否捏合
    pinch_distance: float = 0.0    # 捏合的距离 (0.0 - 1.0)，用于模拟量控制
    landmarks: List[tuple] = None


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

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, # one hand and faster
            model_complexity=0, # 0=fast, 1=accurate
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Face data
        self.latest_face_data: Optional[MediaPipeFaceData] = None
        self.latest_hand_data: Optional[MediaPipeHandData] = None
        self._face_lock = threading.Lock()
        self._hand_lock = threading.Lock()

        # Tracking mode
        self._tracking_mode = False
        self._tracking_callback: Optional[Callable[[MediaPipeFaceData], None]] = None
        self._hand_callback = None
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
        if self.hands:
            self.hands.close()
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
            self.debug_frame = frame.copy()
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

            hand_results = self.hands.process(rgb_frame)
            hand_data = self._process_hand_results(hand_results, frame.shape)

            with self._hand_lock:
                self.latest_hand_data = hand_data

            if self._hand_callback and hand_data.detected:
                self._hand_callback(hand_data)

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

    def _process_hand_results(self, results, frame_shape) -> MediaPipeHandData:
        if not results.multi_hand_landmarks:
            return MediaPipeHandData(
                detected=False,
                handedness="None",
                position=(0.0, 0.0),
                gesture="None",
                fingers_up=[0,0,0,0,0],
                timestamp=time.time()
            )

        # 获取第一只手
        hand_landmarks = results.multi_hand_landmarks[0]

        # --- 新增：提取所有21个点 ---
        # 我们存储归一化坐标(0.0-1.0)，方便画图时根据图像尺寸缩放
        all_landmarks = []
        for lm in hand_landmarks.landmark:
            all_landmarks.append((lm.x, lm.y))

        # 获取左右手信息 (Label: "Left", "Right")
        handedness_info = results.multi_handedness[0].classification[0].label

        frame_h, frame_w = frame_shape[:2]

        # 1. 提取手腕位置 (Landmark 0)
        wrist = hand_landmarks.landmark[0]
        pos_x = (wrist.x * frame_w - frame_w / 2) / (frame_w / 2)
        pos_y = (wrist.y * frame_h - frame_h / 2) / (frame_h / 2)

        # 2. 判断手指状态 (伸直/弯曲)
        fingers_up = self._count_fingers(hand_landmarks)

        thumb_tip = hand_landmarks.landmark[4]  # 拇指指尖
        index_tip = hand_landmarks.landmark[8]  # 食指指尖

        # 计算像素距离 (Pixel Distance)
        # 我们使用像素距离而不是归一化距离，因为像素距离更符合直觉 (如 < 40px)
        dx = (thumb_tip.x - index_tip.x) * frame_w
        dy = (thumb_tip.y - index_tip.y) * frame_h
        distance_px = np.sqrt(dx**2 + dy**2)

        # 计算归一化距离 (用于模拟量控制，如 0.0~0.2)
        # 这里简单除以画面宽度作为一个参考
        norm_distance = distance_px / frame_w

        # 判定阈值：根据分辨率调整。320x240下大约30px，640x480下大约60px
        # 也可以使用相对于手掌大小的动态阈值，但固定阈值最简单有效
        PINCH_THRESHOLD_PX = 40
        is_pinching = distance_px < PINCH_THRESHOLD_PX# and fingers_up == [1,1,0,0,0]

        gesture = None
        if is_pinching:
            gesture = "Pinch"

        return MediaPipeHandData(
            detected=True,
            handedness=handedness_info,
            position=(pos_x, pos_y),
            gesture=gesture,
            fingers_up=fingers_up,
            timestamp=time.time(),
            landmarks=all_landmarks
        )

    def _count_fingers(self, landmarks):
        """判断5根手指是否伸直，返回 [1,0,1,1,1] 格式"""
        fingers = []

        # 关键点索引
        # 拇指: 4, 食指: 8, 中指: 12, 无名指: 16, 小指: 20
        # 关节索引: 拇指(3), 其他(6, 10, 14, 18)

        # 1. 拇指处理 (比较x坐标，因为拇指是侧向运动)
        # 注意：左右手拇指判断方向相反，这里做一个简化版通用判断
        # 更好的方式是根据 handedness 判断，这里简单判断指尖是否远离手掌中心(9号点)
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        pinky_mcp = landmarks.landmark[17]

        # 简单的距离判断：如果拇指指尖离小指根部 比 拇指关节离小指根部 远，则视为伸直
        dist_tip = np.sqrt((thumb_tip.x - pinky_mcp.x)**2 + (thumb_tip.y - pinky_mcp.y)**2)
        dist_ip = np.sqrt((thumb_ip.x - pinky_mcp.x)**2 + (thumb_ip.y - pinky_mcp.y)**2)
        fingers.append(1 if dist_tip > dist_ip else 0)

        # 2. 其他四指 (比较y坐标，指尖需要在关节上方)
        # 注意：OpenCV坐标系y轴向下，所以“上方”意味着y值更小
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18] # 近指关节

        for tip, pip in zip(tips, pips):
            if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

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

    def get_hand_data(self) -> Optional[MediaPipeHandData]:
        with self._hand_lock:
            return self.latest_hand_data

    def enable_tracking_mode(self, callback: Callable[[MediaPipeFaceData], None]):
        """Enable face tracking mode with callback"""
        with self._tracking_lock:
            self._tracking_mode = True
            self._tracking_callback = callback
        self.logger.info("MediaPipe face tracking mode ENABLED")

    def set_hand_callback(self, callback):
        self._hand_callback = callback

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

if __name__ == "__main__":
    import cv2
    import numpy as np
    import time
    from queue import Queue

    # 假设你的服务代码保存在 vision_service.py 中
    # from vision_service import MediaPipeVisionService, MediaPipeFaceData, MediaPipeHandData

    class VisionVisualizer:
        """视觉可视化工具类：负责将数据画在图像上"""
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),      # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),    # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),    # 小指
            (5, 9), (9, 13), (13, 17)                 # 手掌横向连接 (可选)
        ]
        @staticmethod
        def draw_hand_skeleton(image, hand_data):
            """绘制精美的手部骨架"""
            if not hand_data or not hand_data.detected or not hand_data.landmarks:
                return

            h, w = image.shape[:2]
            points = []

            # 1. 将归一化坐标转换为像素坐标
            for lm in hand_data.landmarks:
                px, py = int(lm[0] * w), int(lm[1] * h)
                points.append((px, py))

            # 2. 绘制骨骼连线
            # 根据是否捏合改变骨架颜色 (捏合时变红，平时为黄色)
            bone_color = (0, 0, 255) if hand_data.is_pinching else (0, 255, 255)

            for start_idx, end_idx in VisionVisualizer.HAND_CONNECTIONS:
                pt1 = points[start_idx]
                pt2 = points[end_idx]
                cv2.line(image, pt1, pt2, bone_color, 2, cv2.LINE_AA)

            # 3. 绘制关键点 (关节)
            for i, (px, py) in enumerate(points):
                # 指尖 (4, 8, 12, 16, 20) 画大一点的圆
                if i in [4, 8, 12, 16, 20]:
                    radius = 6
                    color = (0, 255, 0) # 绿色指尖
                else:
                    radius = 4
                    color = (0, 0, 255) # 红色关节

                cv2.circle(image, (px, py), radius, color, -1)
                # 可选：绘制点号用于调试
                # cv2.putText(image, str(i), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

            # 4. 特别标注捏合线 (拇指与食指)
            if hand_data.is_pinching:
                thumb_tip = points[4]
                index_tip = points[8]
                # 画一条显眼的线连接两指
                cv2.line(image, thumb_tip, index_tip, (255, 0, 255), 4)
                # 画中心点
                cx, cy = (thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2
                cv2.circle(image, (cx, cy), 8, (255, 0, 255), -1)

        @staticmethod
        def draw_face_info(image, face_data):
            if not face_data or not face_data.detected:
                return

            h, w = image.shape[:2]
            cx, cy = int((face_data.position[0] + 1) / 2 * w), int((face_data.position[1] + 1) / 2 * h)

            # 1. 画人脸中心十字
            cv2.line(image, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
            cv2.line(image, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

            # 2. 显示头部姿态数据 (Pitch, Yaw, Roll)
            pose = face_data.head_pose
            text = f"Face: P:{pose['pitch']:.1f} Y:{pose['yaw']:.1f} R:{pose['roll']:.1f}"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 3. 简单的姿态指示条 (Yaw - 左右转头)
            bar_x = 50
            bar_y = 60
            bar_w = 100
            # 绘制背景条
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (100, 100, 100), -1)
            # 绘制游标 (映射 yaw -45 到 45 度)
            yaw_norm = np.clip(pose['yaw'], -45, 45)
            cursor_x = int(bar_x + bar_w/2 + (yaw_norm / 45) * (bar_w/2))
            cv2.circle(image, (cursor_x, bar_y + 5), 5, (0, 255, 255), -1)
            cv2.putText(image, "Yaw", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        @staticmethod
        def draw_hand_info(image, hand_data):
            if not hand_data or not hand_data.detected:
                return

            h, w = image.shape[:2]
            wx, wy = int((hand_data.position[0] + 1) / 2 * w), int((hand_data.position[1] + 1) / 2 * h)

            # 1. 标记手腕位置
            color = (255, 0, 0) if hand_data.handedness == "Right" else (0, 0, 255)
            cv2.circle(image, (wx, wy), 8, color, -1)
            cv2.putText(image, hand_data.handedness[0], (wx-5, wy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 2. 显示手势名称
            cv2.putText(image, f"Gesture: {hand_data.gesture}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 3. 捏合状态可视化 (Pinch Visualization)
            if hand_data.is_pinching:
                # 在右上角显示巨大的红色警告
                cv2.putText(image, "PINCHING!", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                # 显示捏合力度条
                cv2.rectangle(image, (w - 30, 100), (w - 10, 200), (50, 50, 50), -1)
                # distance 越小，捏得越紧，填充越高
                # 假设 pinch_distance 0.0 ~ 0.2
                fill_h = int((1.0 - (hand_data.pinch_distance * 5)) * 100)
                fill_h = np.clip(fill_h, 0, 100)
                cv2.rectangle(image, (w - 30, 200 - fill_h), (w - 10, 200), (0, 0, 255), -1)
    import cv2
    import threading
    import time
    from queue import Queue
    from datetime import datetime

    # 引入上面的 Visualizer 类
    # from visualizer import VisionVisualizer

    def run_unit_test():
        # 1. 配置参数
        RESOLUTION = (640, 480)
        FPS = 20
        OUTPUT_FILE = f"test_output_{datetime.now().strftime('%H%M%S')}.avi"

        # 2. 初始化服务
        # 注意：我们使用一个 trick，通过主线程的 VideoCapture 还是让服务自己管理？
        # 为了测试原汁原味的 Service，让 Service 自己管理摄像头。
        # 但为了获取图像，我们需要 Service 暴露出 latest_frame。
        # 如果不想改 Service 源码，我们在主循环里利用 OpenCV 的摄像头独占特性有点难。
        # *** 最佳方案：修改 Service 或 继承 ***

        class DebugVisionService(MediaPipeVisionService):
            """继承原服务，增加获取当前帧的方法用于Debug"""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._current_frame = None
                self._frame_lock = threading.Lock()

            # 重写 _camera_loop 的一部分是很困难的，
            # 所以我们在外部通过 callback 获取数据，
            # 但我们需要 frame。
            # 既然是单元测试，我们假设你已经在 Service 类中添加了 self.latest_frame
            # 或者我们在这里使用一种 Hack 方式：

            # 实际上，最简单的测试方法是：不要用 Service 的内部摄像头，
            # 而是测试程序读取摄像头，把图传给 Service 处理。
            # 但你的 Service 是设计为自己管理摄像头的。

            # 让我们使用 "Queue" 模式，并假设我们在 callback 里没法拿到 frame。
            # 这是一个常见的设计痛点。
            # 在此测试中，我们让 Visualizer 画在一个 空白/黑色 背景上，
            # 或者我们稍微修改 Service 逻辑。
            pass

        # ==========================================
        # 实际测试逻辑
        # ==========================================
        print("初始化服务...")
        service = MediaPipeVisionService(camera_index=0, resolution=RESOLUTION, fps=FPS)

        # 准备视频保存
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, RESOLUTION)

        # 线程通信队列
        # 格式: (face_data, hand_data)
        data_queue = Queue(maxsize=10)

        # 定义回调函数：当有新数据时触发
        def tracking_callback(face_data, hand_data=None):
            # 注意：原来的 callback 签名可能只支持一个参数
            # 我们这里为了演示，假设我们在 Service 里改成了支持双数据回调
            # 或者我们分别注册两个 callback，放入同一个队列
            pass

        # 分别注册回调
        def on_face(data):
            # 这是一个简单的同步机制：
            # 我们用一个全局字典暂存数据，每帧由主线程合成
            pass

        # --- 更好的主循环方案 ---
        # 不完全依赖 callback 做绘图，而是用 callback 更新状态，主线程 loop 绘图

        service.start()

        # 给摄像头一点预热时间
        time.sleep(2.0)

        print(f"开始录制，结果将保存至 {OUTPUT_FILE}")
        print("按 'q' 退出测试")

        try:
            while True:
                loop_start = time.time()

                # 1. 核心 Hacker 步骤：获取 Service 内部的 Frame
                # 由于 Python 的动态特性，我们可以直接访问 Service 实例的 cap
                # 但 cap.read() 不是线程安全的。
                # 最稳妥的方式：Service 应该有一个 get_debug_frame() 方法。

                # 这里演示如果你的 Service 没有提供 frame 访问接口，
                # 我们只能画在黑板上，或者侵入式获取。
                # 假设你在 VisionService 中添加了: self.latest_debug_frame = frame

                # --- 模拟获取帧 (如果 Service 中没有 exposed frame，这里会报错) ---
                # 请在 MediaPipeVisionService 的 _camera_loop 中添加:
                # self.latest_debug_frame = frame.copy()

                frame = None
                if hasattr(service, 'latest_face_data'): # 这是一个不完美的检查
                    # 我们尝试直接访问 service._camera_thread 中的变量是不安全的
                    pass

                # 为了让这个测试跑起来，我们创建一个可视化背景
                canvas = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)

                # 2. 获取数据 (线程安全读取)
                face_data = service.get_face_data()
                hand_data = service.get_hand_data() # 需要确保 Service 有这个方法

                # 3. 绘制
                VisionVisualizer.draw_face_info(canvas, face_data)
                VisionVisualizer.draw_hand_info(canvas, hand_data)
                VisionVisualizer.draw_hand_skeleton(canvas, hand_data)

                # 4. 如果你想看摄像头的图，
                # 必须修改 Service 代码，增加 `self.debug_frame = frame`
                # 假设我们已经修改了，代码如下：
                if hasattr(service, 'debug_frame') and service.debug_frame is not None:
                    # 使用摄像头原图覆盖黑色背景
                    with service._face_lock: # 借用一下锁
                        canvas = service.debug_frame.copy()
                    VisionVisualizer.draw_face_info(canvas, face_data)
                    VisionVisualizer.draw_hand_info(canvas, hand_data)
                    VisionVisualizer.draw_hand_skeleton(canvas, hand_data)
                else:
                    cv2.putText(canvas, "Original Frame Not Available", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(canvas, "(Modify Service to expose self.debug_frame)", (10, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                # 5. 显示和保存
                # cv2.imshow("Vision Service Unit Test", canvas)
                out.write(canvas)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # 维持 FPS
                elapsed = time.time() - loop_start
                wait = max(0, (1.0/FPS) - elapsed)
                time.sleep(wait)

        except KeyboardInterrupt:
            pass
        finally:
            service.stop()
            out.release()
            cv2.destroyAllWindows()
            print("测试结束。")
    run_unit_test()