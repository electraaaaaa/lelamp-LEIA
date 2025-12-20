"""
RGB LED Driver Factory

Provides automatic driver selection based on Raspberry Pi version.

Supported drivers:
- lgpio: Bitbanging driver using lgpio (Pi 5, any GPIO, no kernel module)
- rpi4: PWM-based driver for Pi 3/4 (uses rpi_ws281x)
- rpi5: PWM-based driver for Pi 5 (requires kernel module)
- spi: SPI-based driver for Pi 5 (uses neopixel_spi, GPIO 10)
- simulator: No-op driver for development/testing
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import RGBDriver

logger = logging.getLogger(__name__)


def get_driver(
    led_count: int,
    led_pin: int = 12,
    led_freq_hz: int = 800000,
    led_dma: int = 10,
    led_brightness: int = 255,
    led_invert: bool = False,
    led_channel: int = 0,
    force_driver: Optional[str] = None,
    pixel_order: str = "GRB",
) -> "RGBDriver":
    """
    Factory function to get the appropriate RGB driver for this Pi.

    Args:
        led_count: Number of LEDs in the strip
        led_pin: GPIO pin (default 12 for PWM0, use 10 for SPI on Pi5)
        led_freq_hz: LED signal frequency (800kHz for WS2812)
        led_dma: DMA channel (10 for Pi4, may differ for Pi5)
        led_brightness: Hardware brightness (0-255)
        led_invert: Invert signal (for level shifters)
        led_channel: PWM channel
        force_driver: Override auto-detection ("lgpio", "rpi4", "rpi5", "spi", "simulator")
        pixel_order: Color order for LEDs (GRB, RGB, RGBW, GRBW)

    Returns:
        Appropriate RGBDriver instance for the platform
    """
    from lelamp.user_data import get_pi_version, get_memory_mb

    # Allow forcing a specific driver
    if force_driver:
        driver_name = force_driver.lower()
    else:
        pi_version = get_pi_version()
        memory_mb = get_memory_mb()

        logger.info(f"Detected Pi version: {pi_version}, Memory: {memory_mb}MB")

        if pi_version == 5:
            # On Pi 5, prefer SPI driver if using GPIO 10 (SPI MOSI pin)
            # This avoids the need for kernel module installation
            if led_pin == 10:
                driver_name = "spi"
            else:
                driver_name = "rpi5"
        elif pi_version in (3, 4):
            driver_name = "rpi4"
        else:
            # Non-Pi or unknown - use simulator
            logger.warning("Unknown platform, using simulator driver")
            driver_name = "simulator"

    # Import and instantiate the appropriate driver
    if driver_name == "spi":
        # On Pi 5, try drivers in order: lgpio → PIO → SPI → rpi5
        # lgpio is preferred as it works without extra kernel modules or libraries
        try:
            from .lgpio_driver import LgpioDriver
            driver = LgpioDriver(
                led_count=led_count,
                led_pin=led_pin,
                led_brightness=led_brightness,
                pixel_order=pixel_order,
            )
            # Test initialization before returning
            if driver.initialize():
                logger.info("Using lgpio RGB driver (GPIO bitbanging)")
                return driver
            else:
                logger.warning("lgpio driver failed to initialize, trying fallback")
                driver.cleanup()
        except Exception as e:
            logger.warning(f"lgpio driver not available: {e}")

        # Try PIO driver as fallback
        try:
            from .pi5_pio_driver import Pi5PioDriver
            driver = Pi5PioDriver(
                led_count=led_count,
                led_pin=led_pin,
                led_brightness=led_brightness,
                pixel_order=pixel_order,
                auto_write=False,
            )
            if driver.initialize():
                logger.info("Using Pi5 PIO RGB driver (GPIO 10)")
                return driver
            else:
                logger.warning("Pi5 PIO driver failed to initialize, trying fallback")
                driver.cleanup()
        except Exception as e:
            logger.warning(f"Pi5 PIO driver not available: {e}")

        # Try SPI as fallback
        try:
            from .spi_driver import SpiDriver
            driver = SpiDriver(
                led_count=led_count,
                led_brightness=led_brightness,
                pixel_order=pixel_order,
                spi_bus=0,
                auto_write=False,
            )
            if driver.initialize():
                logger.info("Using SPI RGB driver (neopixel_spi on GPIO 10)")
                return driver
            else:
                logger.warning("SPI driver failed to initialize, trying fallback")
                driver.cleanup()
        except Exception as e:
            logger.warning(f"SPI driver not available: {e}")

        # Fall back to rpi5 driver
        driver_name = "rpi5"

    if driver_name == "lgpio":
        try:
            from .lgpio_driver import LgpioDriver
            logger.info("Using lgpio RGB driver (GPIO bitbanging)")
            return LgpioDriver(
                led_count=led_count,
                led_pin=led_pin,
                led_brightness=led_brightness,
                pixel_order=pixel_order,
            )
        except ImportError as e:
            logger.error(f"Failed to import lgpio driver: {e}")
            logger.warning("Falling back to simulator driver")
            driver_name = "simulator"

    if driver_name == "rpi5":
        try:
            from .rpi5_driver import Rpi5Driver
            logger.info("Using RPi5 RGB driver (kernel module)")
            return Rpi5Driver(
                led_count=led_count,
                led_pin=led_pin,
                led_freq_hz=led_freq_hz,
                led_dma=led_dma,
                led_brightness=led_brightness,
                led_invert=led_invert,
                led_channel=led_channel,
            )
        except ImportError as e:
            logger.error(f"Failed to import RPi5 driver: {e}")
            logger.warning("Falling back to simulator driver")
            driver_name = "simulator"

    if driver_name == "rpi4":
        try:
            from .rpi4_driver import Rpi4Driver
            logger.info("Using RPi4 RGB driver (PWM/DMA)")
            return Rpi4Driver(
                led_count=led_count,
                led_pin=led_pin,
                led_freq_hz=led_freq_hz,
                led_dma=led_dma,
                led_brightness=led_brightness,
                led_invert=led_invert,
                led_channel=led_channel,
            )
        except ImportError as e:
            logger.error(f"Failed to import RPi4 driver: {e}")
            logger.warning("Falling back to simulator driver")
            driver_name = "simulator"

    # Fallback to simulator
    from .simulator_driver import SimulatorDriver
    logger.info("Using simulator RGB driver (no hardware)")
    return SimulatorDriver(led_count=led_count)


__all__ = ["get_driver"]
