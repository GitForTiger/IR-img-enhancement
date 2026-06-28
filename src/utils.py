import logging
import numpy as np
import rasterio
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class GlobalSceneNormalizer:
    """
    Handles global scene statistics calculation and normalization 
    for raw scientific satellite telemetry.
    """
    def __init__(self, vmin: Optional[float] = None, vmax: Optional[float] = None):
        self.vmin = vmin
        self.vmax = vmax
        
        if self.vmin is not None and self.vmax is not None:
            logger.info(f"Initialized Normalizer with static bounds: [{self.vmin}, {self.vmax}]")
        else:
            logger.info("Initialized Normalizer in dynamic mode (bounds uncalibrated).")

    def fit(self, file_path: str) -> Tuple[float, float]:
        """
        Reads a full TIFF scene to calculate global minimum and maximum values,
        ignoring empty background pixels (0) to prevent skewed statistics.
        
        Args:
            file_path (str): The local path to the full-scale satellite .tif file.
            
        Returns:
            Tuple[float, float]: The calculated global minimum and maximum.
        """
        try:
            with rasterio.open(file_path) as src:
                arr = src.read(1)
                # Exclude zero padding often found in rotated satellite swathes
                valid_pixels = arr[arr > 0] 
                
                if len(valid_pixels) == 0:
                    self.vmin, self.vmax = arr.min(), arr.max()
                else:
                    self.vmin, self.vmax = valid_pixels.min(), valid_pixels.max()
                
            logger.info(f"Successfully calibrated global bounds: [{self.vmin:.4f}, {self.vmax:.4f}] from {file_path}")
            return self.vmin, self.vmax
            
        except Exception as e:
            logger.error(f"Failed to read or parse global statistics from {file_path}. Error: {e}")
            raise

    def transform(self, image_array: np.ndarray) -> np.ndarray:
        """
        Normalizes an input numpy array to [-1, 1] using the calibrated bounds,
        which is the mathematical range expected by the Pix2Pix Tanh activation layer.
        
        Args:
            image_array (np.ndarray): The raw scientific pixel array.
            
        Returns:
            np.ndarray: A scaled float32 array ready for neural network inference.
        """
        if self.vmin is None or self.vmax is None:
            logger.warning("Normalizer bounds uncalibrated. Falling back to local array min/max. This may cause color shifting!")
            _min, _max = image_array.min(), image_array.max()
        else:
            _min, _max = self.vmin, self.vmax

        range_val = _max - _min
        
        # Prevent division by zero if an image is completely flat/blank
        if range_val == 0:
            logger.warning("Min and Max values are identical. Returning zeroed array.")
            return np.zeros_like(image_array, dtype=np.float32)

        # Apply global scaling to [-1, 1]
        normalized = ((image_array - _min) / range_val) * 2 - 1
        return normalized.astype(np.float32)

    def fit_transform(self, image_array: np.ndarray) -> np.ndarray:
        """
        Convenience method to calculate bounds and normalize in one pass 
        for dynamic user uploads.
        """
        self.vmin, self.vmax = image_array.min(), image_array.max()
        return self.transform(image_array)


class ImagePostProcessor:
    """
    Utility class to handle post-inference conversions, formatting output tensors 
    back into human-readable visual formats.
    """
    @staticmethod
    def deprocess_to_rgb(normalized_array: np.ndarray) -> np.ndarray:
        """
        Converts a GAN-output [-1, 1] array to a [0, 1] float array for matplotlib/streamlit.
        """
        deprocessed = (normalized_array + 1) / 2.0
        return np.clip(deprocessed, 0, 1)

    @staticmethod
    def deprocess_to_uint8(normalized_array: np.ndarray) -> np.ndarray:
        """
        Converts a GAN-output [-1, 1] array to a [0, 255] uint8 array for saving to disk.
        """
        rgb = ImagePostProcessor.deprocess_to_rgb(normalized_array)
        return (rgb * 255).astype(np.uint8)