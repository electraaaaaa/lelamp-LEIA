"""
RGB LED driver using SPI (NeoPixel SPI).

Uses Adafruit's NeoPixel SPI library which communicates via the SPI bus.
This method works on Raspberry Pi 5 without requiring kernel modules.

WIRING:
- Connect LED data line to GPIO 10 (SPI MOSI)
- Connect LED ground to Pi ground
- Connect LED power to appropriate power supply (5V for WS2812)

REQUIREMENTS:
- SPI must be enabled (raspi-config -> Interface Options -> SPI)
- pip install adafruit-circuitpython-neopixel-spi

NOTE: SPI method may have slight timing differences from PWM method.
Most WS2812/SK6812 LEDs work fine, but some clones may have issues.
"""

from typing import List, Tuple
from .base import RGBDriver


class SpiDriver(RGBDriver):
    """
    RGB driver using SPI bus for LED control.

    This driver uses the Adafruit NeoPixel SPI library which sends
    WS2812 data over the SPI MOSI pin. This works on Pi 5 without
    needing the ws2811 kernel module.

    Advantages:
        - Works on Pi 5 without kernel module
        - Doesn't conflict with audio (PWM can interfere)
        - More portable across Pi versions

    Disadvantages:
        - Only works on SPI pins (GPIO 10 for MOSI)
        - May have timing issues with some LED clones
        - Requires SPI to be enabled
    """

    def __init__(
        self,
        led_count: int,
        led_brightness: int = 255,
        pixel_order: str = "GRB",
        spi_bus: int = 0,
        auto_write: bool = False,
    ):
        """
        Initialize SPI driver.

        Args:
            led_count: Number of LEDs in the strip
            led_brightness: Brightness 0-255
            pixel_order: Color order (GRB for most WS2812, RGB for some)
            spi_bus: SPI bus number (0 or 1)
            auto_write: If True, write to LEDs on every pixel change
        """
        super().__init__(led_count)

        self._brightness = led_brightness
        self._pixel_order = pixel_order
        self._spi_bus = spi_bus
        self._auto_write = auto_write

        self._pixels = None
        self._spi = None

    def _check_spi_available(self) -> bool:
        """Check if SPI device is available."""
        import os
        import glob

        # Pi 5 uses spidev10.x, Pi 4 uses spidev0.x
        # Check for any SPI device
        spi_devices = glob.glob("/dev/spidev*.0")
        if not spi_devices:
            self.logger.error(
                "No SPI device found. "
                "Enable SPI in raspi-config -> Interface Options -> SPI"
            )
            return False

        self.logger.debug(f"Found SPI devices: {spi_devices}")
        return True

    def initialize(self) -> bool:
        """Initialize the NeoPixel SPI hardware."""
        if not self._check_spi_available():
            return False

        try:
            import board
            import neopixel_spi as neopixel

            # Get pixel order constant
            if self._pixel_order.upper() == "RGB":
                order = neopixel.RGB
            elif self._pixel_order.upper() == "RGBW":
                order = neopixel.RGBW
            elif self._pixel_order.upper() == "GRBW":
                order = neopixel.GRBW
            else:
                order = neopixel.GRB  # Default for WS2812

            # Get SPI bus
            if self._spi_bus == 0:
                spi = board.SPI()
            else:
                # Alternative SPI bus (if available)
                try:
                    spi = board.SPI1()
                except AttributeError:
                    self.logger.warning("SPI1 not available, using SPI0")
                    spi = board.SPI()

            self._spi = spi

            # Create NeoPixel object
            self._pixels = neopixel.NeoPixel_SPI(
                spi,
                self.led_count,
                brightness=self._brightness / 255.0,
                auto_write=self._auto_write,
                pixel_order=order,
            )

            self._initialized = True
            self.logger.info(
                f"SPI RGB driver initialized: {self.led_count} LEDs on SPI{self._spi_bus}"
            )
            return True

        except ImportError as e:
            self.logger.error(
                f"NeoPixel SPI library not installed: {e}. "
                "Install with: pip install adafruit-circuitpython-neopixel-spi"
            )
            return False
        except PermissionError as e:
            self.logger.error(
                f"Permission denied accessing SPI: {e}. "
                "Add user to 'spi' group: sudo usermod -aG spi $USER"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize SPI driver: {e}")
            return False

    def render(self, frame: List[Tuple[int, int, int]]) -> None:
        """Write frame to LED strip via SPI."""
        if not self._initialized or self._pixels is None:
            return

        try:
            # Set all pixels
            for i, (r, g, b) in enumerate(frame):
                if i < self.led_count:
                    self._pixels[i] = (r, g, b)

            # Push to hardware
            self._pixels.show()

        except Exception as e:
            self.logger.error(f"Error rendering frame: {e}")

    def set_brightness(self, brightness: int) -> None:
        """Set hardware brightness (0-255)."""
        self._brightness = max(0, min(255, brightness))

        if self._pixels is not None:
            # NeoPixel SPI uses 0.0-1.0 brightness
            self._pixels.brightness = self._brightness / 255.0
            self.logger.debug(f"Brightness set to {self._brightness}")

    def cleanup(self) -> None:
        """Turn off LEDs and release resources."""
        if self._pixels is not None:
            # Turn off all LEDs
            self._pixels.fill((0, 0, 0))
            self._pixels.show()

            # Deinit
            try:
                self._pixels.deinit()
            except Exception:
                pass

            self._pixels = None

        self._initialized = False
        self.logger.info("SPI RGB driver cleaned up")
