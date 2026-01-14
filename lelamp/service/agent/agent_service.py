import sounddevice as sd
import logging
import threading
from typing import Callable, Optional

from RPi.GPIO import output
from deepgram import AsyncDeepgramClient
import asyncio
from deepgram.core.events import EventType

# 配置日志
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("Agent_Service")

class FluxListener:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(FluxListener, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化基础状态，确保不会重复初始化"""
        if getattr(self, "_initialized", False):
            return

        self.dg_client: Optional[AsyncDeepgramClient] = None
        self.dg_connection = None
        self.input_stream = None
        self.is_running = False
        self._loop = None
        self._connection_task = None

        # 回调函数容器 (外部注入)
        self.on_speech_start: Optional[Callable] = None
        self.on_transcript_update: Optional[Callable] = None
        self.on_turn_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        self._transcript_buffer = []
        self._initialized = True

    def initialize(self,
                   api_key: str,
                   on_speech_start: Callable = None,
                   on_turn_complete: Callable = None,
                   on_transcript_update: Callable = None,
                   device_index: int = None):
        """
        配置 Deepgram 和回调函数
        """
        self.dg_client = AsyncDeepgramClient(api_key=api_key)
        self.on_speech_start = on_speech_start
        self.on_turn_complete = on_turn_complete
        self.on_transcript_update = on_transcript_update
        self.device_index = device_index

        logger.info("FluxListener initialized (Singleton).")

    async def start(self):
        """启动 Deepgram 连接和麦克风流"""
        if self.is_running:
            logger.warning("Listener is already running.")
            return

            # 创建或获取事件循环
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._loop.run_forever, daemon=True).start()

        # 启动 Sounddevice 输入流
        try:
            self.input_stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=16000,
                dtype="int16",
                callback=self._audio_callback,
                blocksize=1024
            )
            self.input_stream.start()
            self.is_running = True
            logger.info("Microphone listening started...")

        except Exception as e:
            logger.error(f"Microphone Error: {e}")
            self.stop()
        await self._run_connection()

    async def _run_connection(self):
        """保持连接活跃的异步方法"""
        try:
            options = {
                "model": "flux-general-en",
                "encoding": "linear16",
                "sample_rate": "16000",
                "eot_timeout_ms": 1000,
            }

            # 保持连接打开
            async with self.dg_client.listen.v2.connect(**options) as connection:
                self.dg_connection = connection
                self._register_events(connection)
                await connection.start_listening()

                # 保持连接活跃，直到 stop() 被调用
                while self.is_running:
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Deepgram Connection Error: {e}")
            if self.on_error:
                self.on_error(e)

    def stop(self):
        """停止监听并释放资源"""
        self.is_running = False

        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        logger.info("FluxListener stopped.")

    def _audio_callback(self, indata, frames, time, status):
        """SoundDevice 的回调：将音频推送到 Deepgram"""
        if status:
            logger.warning(f"Audio status: {status}")
        # print(indata)

        if self.dg_connection and self.is_running:
            # print("sent")
            # 在事件循环中发送音频数据
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(
                        self.dg_connection.send_media(bytes(indata))
                    )
                )
        else:
            print(self.dg_connection, self.is_running)

    def _dispatch_callback(self, callback, *args):
        """[新增] 辅助函数：安全地从同步回调中调用异步或同步函数"""
        if not callback:
            return

        if asyncio.iscoroutinefunction(callback):
            # 如果是异步函数，且循环正在运行，则线程安全地提交任务
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(callback(*args), self._loop)
            else:
                logger.error("Event loop is not running, cannot schedule async callback")
        else:
            # 如果是普通同步函数，直接调用
            callback(*args)

    def _register_events(self, connection):
        """绑定 Deepgram 内部事件到外部回调"""

        def on_message(message):
            if hasattr(message, 'type'):
                if message.type == "TurnInfo":
                    if hasattr(message, 'event') and message.event=='EndOfTurn':
                        transcript = message.transcript
                        self._dispatch_callback(self.on_turn_complete, transcript)

            else:
                print(message)
        def on_speech_started():
            """Flux 核心：打断信号"""
            logger.info(">> User started speaking (Barge-in)")
            if self.on_speech_start:
                self.on_speech_start()

        def on_utterance_end():
            """Flux 核心：话轮结束"""
            if self._transcript_buffer:
                full_text = " ".join(self._transcript_buffer).strip()
                self._transcript_buffer = []

                logger.info(f"End of turn detected. Text: {full_text}")

                if self.on_turn_complete and full_text:
                    self.on_turn_complete(full_text)

                    # 注册事件处理器
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.OPEN, lambda _: logger.info("Connection opened"))
        connection.on(EventType.CLOSE, lambda _: logger.info("Connection closed"))
        connection.on(EventType.ERROR, lambda error: logger.error(f"Error: {error}"))


import os
import queue
from dotenv import load_dotenv

# 加载 API Key
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

import os

from groq import Groq
from lelamp.service.agent.tools import Tool

# class LLM:
#     def __init__(self):
#         self.is_speaking = False
#         self.response_queue = queue.Queue()
#         self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
#         self.chat_history = [
#             {
#                 "role": "system",
#                 "content": ""
#             }
#         ]
#         self.agent = Agent()
#
#     def handle_interruption(self):
#         """
#         Interruption callback
#         """
#         pass  # Not Implemented
#
#     async def handle_user_input(self, text):
#         """
#         Full sentence process callback
#         """
#         try:
#             logger.info(f"LLM received from user, requesting: {text}'")
#             self.chat_history.append({"role": "user", "content": text})
#             chat_completion = self.client.chat.completions.create(
#                 messages=self.chat_history,
#                 model="llama-3.3-70b-versatile",
#                 tools=Tool.tools_schema[:3],
#                 tool_choice="required",
#
#             )
#             message = chat_completion.choices[0].message
#             logger.info(f"LLM response received: {message}")
#         except Exception as e:
#             logger.critical(f"Error{e}")
#         # Check if tools called
#         if message.tool_calls:
#             logger.info(f"LLM Calling {len(message.tool_calls) }tools...")
#
#             # 4. 依次执行工具
#             for tool_call in message.tool_calls:
#                 func_name = tool_call.function.name
#                 args = tool_call.function.arguments
#                 call_id = tool_call.id
#
#                 logger.info(f"   -> Execute: {func_name}({args})")
#                 result = await Tool.execute(func_name, args, self.agent) or "Success"
#                 logger.info(f"   <- Result: {result}")
#
#                 # 5. 将结果作为 role='tool' 存入历史
#                 self.chat_history.append({
#                     "role": "tool",
#                     "tool_call_id": call_id,
#                     "name": func_name,
#                     "content": result
#                 })
#
#             # 6. 第二轮调用：把工具结果发回给 LLM，获取最终回复
#             logger.info("Second request")
#             final_completion = self.client.chat.completions.create(
#                 messages=self.chat_history,
#                 model="llama-3.3-70b-versatile",
#                 # 第二轮通常不需要再强制 tool_choice，除非是多步复杂任务
#             )
#             final_response = final_completion.choices[0].message.content
#             self.chat_history.append({"role": "assistant", "content": final_response})
#             logger.info(f"Final response: {message.content}")
#             return final_response
#
#         else:
#             # No tool call, output
#             logger.info(f"Final response: {message.content}")
#             return message.content
#
#     def handle_transcript_update(self, text, is_final):
#         pass  # Not Implemented or legacy
#
#     def synthesize_and_play(self, text):
#         pass  # Not Implemented

# --- 主程序 ---
# async def init_agent_service():
#     # 1. 实例化业务逻辑
#     bot = LLM()
#
#     # 2. 获取单例的听觉模块
#     listener = FluxListener()
#
#     # 3. 初始化并注入依赖 (Dependency Injection)
#     # 关键点：将 bot 的方法传给 listener，实现解耦
#     listener.initialize(
#         api_key=DEEPGRAM_API_KEY,
#         on_speech_start=bot.handle_interruption,       # 绑定打断逻辑
#         on_turn_complete=bot.handle_user_input,        # 绑定对话逻辑
#         on_transcript_update=bot.handle_transcript_update # 绑定 UI 逻辑
#     )
#
#     # 4. 启动
#     print("系统启动中... (按 Ctrl+C 退出)")
#     await listener.start()

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
import lelamp.globals as g
from lelamp.service.theme import ThemeSound
class Agent(
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
    def __init__(self):
        super().__init__()
        """
        Initialize LeLamp agent.

        Services are already initialized by server.py and available in globals.
        The agent consumes these services and adds AI-specific behavior.

        Args:
            config: Configuration dict. If None, uses global CONFIG.
        """
        # Use provided config or fall back to global
        config = g.CONFIG
        print(config)
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

        # Agent-specific service setup
        # self._setup_workflow_service()
        # self._setup_spotify_service()
        # self._setup_alarm_service()

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
    def _mark_activity(self):
        """Mark that activity occurred (for idle timeout)."""
        import time
        self.last_activity_time = time.time()

    def _set_system_volume(self, volume_percent: int):
        """Set system playback volume."""
        import subprocess
        try:
            subprocess.run(
                ["amixer", "sset", "PCM", f"{volume_percent}%"],
                capture_output=True, timeout=5
            )
        except Exception:
            pass

    def _set_system_microphone_volume(self, volume_percent: int):
        """Set system microphone/capture volume."""
        import subprocess
        try:
            for control in ['Capture', 'ADC', 'ADC PCM']:
                subprocess.run(
                    ["amixer", "sset", control, f"{volume_percent}%"],
                    capture_output=True, timeout=5
                )
        except Exception:
            pass

import asyncio
import websockets
import json
import base64
import sounddevice as sd
import os
import queue
async def init_agent_service():
    bot = LLM()
    await bot.start()

class LLM:
    def __init__(self):
        # 配置
        self.API_KEY = os.getenv("OPENAI_API_KEY")  # 或者直接填入 "sk-..."
        # 使用最新的 Realtime 模型
        self.URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"#-2024-10-01"

        # 音频配置 (OpenAI Realtime API 标准为 24kHz PCM16 Mono)
        self.SAMPLE_RATE = 24000
        self.CHANNELS = 1
        self.DTYPE = 'int16'
        self.CHUNK_SIZE = 1024  # 每次读取的音频帧大小

        # 线程安全的队列，用于在音频回调和 asyncio 之间传输数据
        self.input_queue = asyncio.Queue()
        self.output_queue = queue.Queue()

        self.agent = Agent()

    def _fix_tools_format(self, original_tools):
        """将 Chat Completion 格式的工具转换为 Realtime API 格式"""
        fixed_tools = []
        for t in original_tools:
            # 如果是旧格式 {"type": "function", "function": {"name":...}}
            if "function" in t:
                func_def = t["function"]
                fixed_tools.append({
                    "type": "function",
                    "name": func_def.get("name"),
                    "description": func_def.get("description", ""),
                    "parameters": func_def.get("parameters", {})
                })
            # 如果已经是新格式（有 name 在顶层），直接使用
            elif "name" in t:
                fixed_tools.append(t)
        return fixed_tools

    def input_callback(self, indata, frames, time, status):
        """麦克风录音回调：将录到的原始音频数据放入队列"""
        # if status:
        #     print(status)
        # 将 numpy array 转换为 bytes
        audio_bytes = indata.tobytes()
        # 注意：这里我们不能直接用 await，因为这是在非 async 线程中运行
        # 我们使用 asyncio.run_coroutine_threadsafe 或者简单的 loop.call_soon_threadsafe
        # 为了简单，我们这里使用 asyncio.Queue 的 put_nowait (如果是在同一个 loop)
        # 但由于这是跨线程，标准的 queue 配合 async 包装通常更稳健，
        # 或者直接在主循环中处理。这里为了代码简洁，我们假定 loop 在运行。
        try:
            self.loop.call_soon_threadsafe(self.input_queue.put_nowait, audio_bytes)
        except Exception as e:
            pass

    async def send_audio(self, websocket):
        """持续从麦克风队列读取数据并发送给 OpenAI"""
        while True:
            audio_bytes = await self.input_queue.get()
            # Base64 编码
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

            # 发送 input_audio_buffer.append 事件
            event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            await websocket.send(json.dumps(event))

    async def receive(self, websocket):
        """持续接收 OpenAI 的响应并放入播放队列"""
        async for message in websocket:
            event = json.loads(message)
            event_type = event.get("type")
            current_stream_type = None
            # print(message)

            # 打印部分日志以便调试
            # --- 1. 用户语音转录结果 (User Input) ---
            # 需要在 session 中开启 input_audio_transcription 才会收到此事件
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    if current_stream_type: print("") # 换行
                    print(f"[用户语音转录]: {transcript}")
                    current_stream_type = None

            # --- 2. AI 文本流式输出 (AI Response) ---
            # 因为 modalities=["text"]，所以这里监听 response.text.delta 而不是 audio
            elif event_type == "response.text.delta":
                delta = event.get("delta", "")
                if current_stream_type != "text":
                    print(f"[AI Response]: ", end="")
                    current_stream_type = "text"
                print(delta, end="", flush=True)

            # --- 3. 响应结束 ---
            elif event_type == "response.done":
                if current_stream_type == "text":
                    print(f"") # 结束颜色
                    current_stream_type = None

            elif event['type'] == "response.function_call_arguments.done":
                # AI 已经生成了完整的函数调用参数
                call_id = event["call_id"]
                name = event["name"]
                arguments = event["arguments"]

                print(f"\n[System] AI 请求调用工具: {name}({arguments})")
                output_str = await Tool.execute(name, arguments, self.agent)
                # 1. 创建一个由工具输出组成的新对话项 (Item)
                item_create_event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,  # 必须对应之前的 call_id
                        "output": output_str
                    }
                }
                await websocket.send(json.dumps(item_create_event))

                # 2. 告诉 AI："结果给你了，现在请根据这个结果回复我" (触发 response.create)
                response_create_event = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text"],
                    }
                }
                await websocket.send(json.dumps(response_create_event))

    async def start(self):
        self.loop = asyncio.get_running_loop()

        headers = {
            "Authorization": "Bearer " + self.API_KEY,
            "OpenAI-Beta": "realtime=v1"
        }

        print("正在连接 OpenAI Realtime API...")
        async with websockets.connect(self.URL, additional_headers=headers) as websocket:
            system_prompt = "You always think and response in English no matter what language the user speaks. You're a cute lamp who DON'T speak, and the words you say are NOT visible to the user. Use your language as thinking process in your brain. The only way you can interact with the user is calling functions. Use the functions to response the user. For example, when you feel said you should look down and turn led to cold color. When you're happy you should turn your led to warm color and play positive animation. When you're excited you should make your led bright and colorful, and call multiple actions to be hyperactive. For another example, when the user ask you the weather, you can query thr weather first and user warm yellow to represent sunny or cold blue to represent rainy."
            # 1. 初始化会话 (可选：设置声音、VAD模式等)
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "voice": "alloy",  # 可选: alloy, ash, ballad, coral, echo, sage, shimmer, verse
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",  # 开启服务端语音活动检测（自动打断、自动回复）
                    },
                    "tools": self._fix_tools_format(Tool.tools_schema),
                    "tool_choice": "auto",
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        },
                    "instructions": system_prompt.strip()
                }
            }
            await websocket.send(json.dumps(session_update))

            input_stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                callback=self.input_callback,
                blocksize=self.CHUNK_SIZE
        )

            with input_stream:
                # 3. 并发运行发送和接收任务
                send_task = asyncio.create_task(self.send_audio(websocket))
                receive_task = asyncio.create_task(self.receive(websocket))

                try:
                    await asyncio.gather(send_task, receive_task)
                except KeyboardInterrupt:
                    print("停止对话...")

if __name__ == "__main__":
    asyncio.run(init_agent_service())