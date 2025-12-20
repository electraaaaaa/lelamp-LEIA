"""Targeting - Crosshair targeting effect focusing from outer to inner rings"""

import time
import math
from typing import Optional, Tuple
from . import register_animation, get_frame_interval

@register_animation(
    name="targeting",
    description="Crosshair targeting effect - use when focusing on something specific, aiming, or zeroing in on a target. Creates a locked-on feel."
)
def targeting(controller, color: Optional[Tuple[int, int, int]] = None, duration: Optional[float] = None):
    """Crosshair targeting - focuses from outer to inner rings"""
    start_time = time.time()

    while not controller._stop_animation.is_set():
        if duration and (time.time() - start_time) >= duration:
            break

        base_color = color if color else controller.get_current_color()
        t = time.time() * 2.5
        focus = (math.sin(t) + 1) / 2  # 0 to 1

        frame = [(0, 0, 0)] * controller.led_count

        if controller.has_rings():
            # Ring-based targeting: focus from outer to inner
            for ring_idx, ring in enumerate(controller._rings):
                ring_distance = ring_idx / len(controller._rings)  # 0 (outer) to 1 (inner)
                ring_active = 1.0 - abs(ring_distance - focus)

                if ring_active > 0.3:
                    ring_size = ring['count']
                    num_markers = 4
                    marker_width = max(1, ring_size // 20)

                    for marker_num in range(num_markers):
                        center = ring['start'] + (ring_size * marker_num // num_markers)

                        for offset in range(-marker_width, marker_width + 1):
                            led_idx = ring['start'] + ((center - ring['start'] + offset) % ring_size)

                            # Specific correction: LED 27 -> LED 25
                            if led_idx == 27:
                                led_idx = 25

                            if led_idx <= ring['end']:
                                distance = abs(offset)
                                intensity = (1.0 - distance / max(1, marker_width)) * ring_active
                                intensity *= (0.5 + focus * 0.5)

                                frame[led_idx] = (int(base_color[0] * intensity),
                                                 int(base_color[1] * intensity),
                                                 int(base_color[2] * intensity))
        else:
            # Fallback
            quarter = controller._active_led_count // 4
            for quad in range(4):
                center = controller._active_led_start + (quarter * quad)
                spread = max(1, int(5 * (1 - focus) + 2))
                for i in range(-spread, spread + 1):
                    idx = center + i
                    if controller._active_led_start <= idx <= controller._active_led_end:
                        intensity = 1.0 - (abs(i) / spread) * 0.5
                        intensity *= (0.5 + focus * 0.5)
                        frame[idx] = (int(base_color[0] * intensity),
                                     int(base_color[1] * intensity),
                                     int(base_color[2] * intensity))

        controller._update_frame(frame)
        time.sleep(get_frame_interval())
