"""
RGB LED driver using lgpio bitbanging.

Works on Raspberry Pi 5 without kernel modules by directly
controlling GPIO pins using the lgpio library.

WIRING:
- Connect LED data line to GPIO 10 (or any available GPIO)
- Connect LED ground to Pi ground
- Connect LED power to appropriate power supply (3.3V-5V for WS2812)
- IMPORTANT: If using external power, connect external GND to Pi GND

REQUIREMENTS:
- pip install lgpio

NOTE: Bitbanging timing is less precise than hardware SPI/PWM,
but works reliably for most WS2812/SK6812 LEDs.
"""

import time
from typing import List, Tuple
from .base import RGBDriver


class LgpioDriver(RGBDriver):
    """
    RGB driver using lgpio for GPIO bitbanging.

    This driver directly controls GPIO pins using the lgpio library,
    which works on Pi 5 without requiring kernel modules or SPI setup.

    Advantages:
        - Works on Pi 5 without kernel module
        - Works on any GPIO pin
        - No SPI/PWM configuration needed

    Disadvantages:
        - Timing is software-based (less precise)
        - CPU-intensive during rendering
        - May have occasional glitches under heavy system load
    """

    def __init__(
        self,
        led_count: int,
        led_pin: int = 10,
        led_brightness: int = 255,
        pixel_order: str = "GRB",
    ):
        """
        Initialize lgpio driver.

        Args:
            led_count: Number of LEDs in the strip
            led_pin: GPIO pin number (BCM numbering)
            led_brightness: Brightness 0-255
            pixel_order: Color order (GRB for most WS2812, RGB for some)
        """
        super().__init__(led_count)

        self._pin = led_pin
        self._brightness = led_brightness
        self._pixel_order = pixel_order.upper()

        self._handle = None
        self._chip = None

    def _find_gpio_chip(self) -> int:
        """Find the correct GPIO chip for this Pi."""
        import lgpio

        # Pi 5 typically uses gpiochip4, older Pis use gpiochip0
        for chip in [4, 0, 1, 2, 3]:
            try:
                h = lgpio.gpiochip_open(chip)
                lgpio.gpiochip_close(h)
                return chip
            except Exception:
                continue
        return 0  # Default fallback

    def initialize(self) -> bool:
        """Initialize the GPIO pin for output."""
        try:
            import lgpio

            self._chip = self._find_gpio_chip()
            self._handle = lgpio.gpiochip_open(self._chip)

            # Set pin as output, initially low
            lgpio.gpio_claim_output(self._handle, self._pin, 0)

            self._initialized = True
            self.logger.info(
                f"lgpio RGB driver initialized: {self.led_count} LEDs on GPIO {self._pin} (chip {self._chip})"
            )
            return True

        except ImportError as e:
            self.logger.error(
                f"lgpio library not installed: {e}. "
                "Install with: pip install lgpio"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize lgpio driver: {e}")
            return False

    def _send_byte(self, byte: int) -> None:
        """Send a single byte using WS2812B timing."""
        import lgpio

        h = self._handle
        pin = self._pin

        for i in range(7, -1, -1):
            if (byte >> i) & 1:
                # 1 bit: longer high pulse (~0.8us high, ~0.45us low)
                lgpio.gpio_write(h, pin, 1)
                lgpio.gpio_write(h, pin, 1)
                lgpio.gpio_write(h, pin, 1)
                lgpio.gpio_write(h, pin, 0)
            else:
                # 0 bit: shorter high pulse (~0.4us high, ~0.85us low)
                lgpio.gpio_write(h, pin, 1)
                lgpio.gpio_write(h, pin, 0)
                lgpio.gpio_write(h, pin, 0)
                lgpio.gpio_write(h, pin, 0)

    def _send_color(self, r: int, g: int, b: int) -> None:
        """Send RGB color in the correct order."""
        # Apply brightness
        scale = self._brightness / 255.0
        r = int(r * scale)
        g = int(g * scale)
        b = int(b * scale)

        if self._pixel_order == "GRB":
            self._send_byte(g)
            self._send_byte(r)
            self._send_byte(b)
        elif self._pixel_order == "RGB":
            self._send_byte(r)
            self._send_byte(g)
            self._send_byte(b)
        elif self._pixel_order == "BGR":
            self._send_byte(b)
            self._send_byte(g)
            self._send_byte(r)
        else:  # Default GRB
            self._send_byte(g)
            self._send_byte(r)
            self._send_byte(b)

    def render(self, frame: List[Tuple[int, int, int]]) -> None:
        """Write frame to LED strip via GPIO bitbanging."""
        if not self._initialized or self._handle is None:
            return

        import lgpio

        try:
            # Send all pixel data
            for i, (r, g, b) in enumerate(frame):
                if i < self.led_count:
                    self._send_color(r, g, b)

            # Reset pulse (hold low for >50us)
            lgpio.gpio_write(self._handle, self._pin, 0)
            time.sleep(0.0001)  # 100us reset

        except Exception as e:
            self.logger.error(f"Error rendering frame: {e}")

    def set_brightness(self, brightness: int) -> None:
        """Set software brightness (0-255)."""
        self._brightness = max(0, min(255, brightness))
        self.logger.debug(f"Brightness set to {self._brightness}")

    def cleanup(self) -> None:
        """Turn off LEDs and release GPIO."""
        if self._handle is not None:
            import lgpio

            try:
                # Turn off all LEDs
                for _ in range(self.led_count):
                    self._send_color(0, 0, 0)
                lgpio.gpio_write(self._handle, self._pin, 0)

                # Release GPIO
                lgpio.gpio_free(self._handle, self._pin)
                lgpio.gpiochip_close(self._handle)
            except Exception:
                pass

            self._handle = None

        self._initialized = False
        self.logger.info("lgpio RGB driver cleaned up")
