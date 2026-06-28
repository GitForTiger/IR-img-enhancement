import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IR_Enhancement_Pipeline")

try :
    from .model import SatelliteColorizer
    from .utils import GlobalSceneNormalizer

    _all_ = [
        "SatelliteColorizer",
        "GlobalSceneNormalizer",
    ]

    logger.info("Core components successfully exposed to package root.")

except ImportError as e:
    logger.error(f"Failed to initialize package dependencies: {e}")