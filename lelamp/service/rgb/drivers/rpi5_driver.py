"""
RGB LED driver for Raspberry Pi 5.

Uses rpi_ws281x library with kernel module support.
The Pi 5 uses the RP1 southbridge chip for GPIO, requiring different
initialization than earlier Pi models.

SETUP REQUIRED FOR Pi 5:
1. Build and install the ws2811 kernel module from source
2. Add device tree overlay: sudo dtoverlay ws2811-pwm
3. Install rpi-ws281x: pip install rpi-ws281x
4. See: https://github.com/jgarff/rpi_ws281x/wiki/Raspberry-Pi-5-Support

ALTERNATIVE: Use SPI method (GPIO 10) which doesn't require kernel module
"""

from typing import List, Tuple
from .base import RGBDriver


class Rpi5Driver(RGBDriver):
    """
    RGB driver for Raspberry Pi 5 using rpi_ws281x with kernel module.

    The Raspberry Pi 5 uses the RP1 I/O controller chip which requires
    a kernel module for WS281x LED control. This driver uses the updated
    rpi_ws281x library (>= 6.0.0) which supports the Pi 5.

    Setup requirements:
        1. Install kernel module and device tree overlay
        2. Install rpi-ws281x >= 6.0.0 from pip
        3. Run with sudo for hardware access

    Supported pins (via RP1):
        - GPIO 12 (PWM0) - default
        - GPIO 13 (PWM1)
        - GPIO 18 (PWM0)
        - GPIO 19 (PWM1)
        - GPIO 10 (SPI MOSI) - alternative method

    Note: The 2GB Pi 5 variant may require additional configuration
    as its revision code differs from 4GB/8GB models.
    """

    def __init__(
        self,
        led_count: int,
        led_pin: int = 12,
        led_freq_hz: int = 800000,
        led_dma: int = 10,
        led_brightness: int = 255,
        led_invert: bool = False,
        led_channel: int = 0,
    ):
        super().__init__(led_count)

        self.led_pin = led_pin
        self.led_freq_hz = led_freq_hz
        self.led_dma = led_dma
        self.led_invert = led_invert
        self.led_channel = led_channel
        self._brightness = led_brightness

        self._strip = None

    def _check_kernel_module(self) -> bool:
        """Check if the ws2811 kernel module is loaded."""
        try:
            with open("/proc/modules", "r") as f:
                modules = f.read()
                return "ws2811" in modules
        except Exception:
            return False

    def _check_device_tree_overlay(self) -> bool:
        """Check if the ws2811 device tree overlay is loaded."""
        try:
            import subprocess
            result = subprocess.run(
                ["dtoverlay", "-l"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "ws2811" in result.stdout
        except Exception:
            # Can't check, assume it might be okay
            return True

    def initialize(self) -> bool:
        """Initialize the PixelStrip hardware for Pi 5."""
        # Check for kernel module (Pi 5 requirement)
        if not self._check_kernel_module():
            self.logger.warning(
                "ws2811 kernel module not detected. "
                "Pi 5 requires the kernel module for LED control. "
                "See: https://github.com/jgarff/rpi_ws281x/wiki/Raspberry-Pi-5-Support"
            )
            # Continue anyway - module might be built-in or loaded differently

        try:
            from rpi_ws281x import PixelStrip, ws

            # Check library version supports Pi 5
            # Version 6.0+ has Pi 5 support
            try:
                version = getattr(ws, '__version__', '5.0.0')
                major = int(version.split('.')[0])
                if major < 6:
                    self.logger.warning(
                        f"rpi_ws281x version {version} may not support Pi 5. "
                        "Consider upgrading to >= 6.0.0"
                    )
            except Exception:
                pass  # Can't determine version, continue anyway

            self._strip = PixelStrip(
                self.led_count,
                self.led_pin,
                self.led_freq_hz,
                self.led_dma,
                self.led_invert,
                self._brightness,
                self.led_channel,
            )
            self._strip.begin()
            self._initialized = True
            self.logger.info(
                f"RPi5 RGB driver initialized: {self.led_count} LEDs on GPIO {self.led_pin}"
            )
            return True

        except ImportError as e:
            self.logger.error(
                f"rpi_ws281x library not installed: {e}. "
                "Install with: pip install rpi-ws281x"
            )
            return False
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "kernel" in error_msg or "module" in error_msg:
                self.logger.error(
                    f"Kernel module error: {e}. "
                    "Pi 5 requires ws2811 kernel module. "
                    "Run: sudo dtoverlay ws2811-pwm"
                )
            elif "sp" in error_msg or "permission" in error_msg:
                self.logger.error(
                    f"Permission error: {e}. "
                    "Make sure to run with sudo."
                )
            else:
                self.logger.error(f"Failed to initialize LED strip: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error initializing LED strip: {e}")
            return False

    def render(self, frame: List[Tuple[int, int, int]]) -> None:
        """Write frame to LED strip."""
        if not self._initialized or self._strip is None:
            return

        try:
            from rpi_ws281x import Color

            # Set all pixels
            for i, (r, g, b) in enumerate(frame):
                if i < self.led_count:
                    self._strip.setPixelColor(i, Color(r, g, b))

            # Push to hardware
            self._strip.show()

        except Exception as e:
            self.logger.error(f"Error rendering frame: {e}")

    def set_brightness(self, brightness: int) -> None:
        """Set hardware brightness (0-255)."""
        self._brightness = max(0, min(255, brightness))

        if self._strip is not None:
            self._strip.setBrightness(self._brightness)
            self._strip.show()
            self.logger.debug(f"Brightness set to {self._brightness}")

    def cleanup(self) -> None:
        """Turn off LEDs and release resources."""
        if self._strip is not None:
            # Turn off all LEDs
            self.clear()
            self._strip = None

        self._initialized = False
        self.logger.info("RPi5 RGB driver cleaned up")
