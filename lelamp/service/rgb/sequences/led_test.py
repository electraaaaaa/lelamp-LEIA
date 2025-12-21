"""LED Test - Simple single LED chase for hardware testing"""

import time
from typing import Optional, Tuple
from . import register_animation


@register_animation(
    name="led_test",
    description="Simple single LED chase from 0 to 93. For hardware testing."
)
def led_test(controller, color: Optional[Tuple[int, int, int]] = None, duration: float = 10.0):
    """Single LED at a time, chasing from 0 to 93."""
    if color is None:
        color = (255, 255, 255)  # White default

    led_count = controller.led_count
    delay = 0.05  # 50ms per LED

    while not controller._stop_animation.is_set():
        for i in range(led_count):
            if controller._stop_animation.is_set():
                break

            # All off except current LED
            frame = [(0, 0, 0)] * led_count
            frame[i] = color
            controller._update_frame(frame)
            time.sleep(delay)

    # Turn off
    controller._update_frame([(0, 0, 0)] * led_count)
