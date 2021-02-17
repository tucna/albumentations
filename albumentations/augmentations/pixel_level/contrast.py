import random
import warnings
import numpy as np

from typing import Optional, Union, Callable, Sequence, Tuple, Any

from . import functional as F
from ...core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = ["Equalize", "RandomBrightnessContrast", "RandomBrightness", "RandomContrast", "CLAHE"]


class Equalize(ImageOnlyTransform):
    """Equalize the image histogram.

    Args:
        mode (str): {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable.
            Function signature must include `image` argument.
        mask_params: Params for mask function.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        mode: str = "cv",
        by_channels: bool = True,
        mask: Optional[Union[np.ndarray, Callable[..., np.ndarray]]] = None,
        mask_params: Sequence[str] = (),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        modes = ["cv", "pil"]
        if mode not in modes:
            raise ValueError(f"Unsupported equalization mode. Supports: {modes}. Got: {mode}")

        super().__init__(always_apply, p)
        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, image: np.ndarray, mask: np.ndarray = None, **params) -> np.ndarray:  # type: ignore
        return F.equalize(image, mode=self.mode, by_channels=self.by_channels, mask=mask)

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        if not callable(self.mask):
            return {"mask": self.mask}

        return {"mask": self.mask(**params)}

    @property
    def targets_as_params(self) -> Sequence[str]:
        return ["image"] + list(self.mask_params)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("mode", "by_channels")


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        brightness_limit: Union[float, Tuple[float, float]] = 0.2,
        contrast_limit: Union[float, Tuple[float, float]] = 0.2,
        brightness_by_max: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply(self, img: np.ndarray, alpha: float = 1.0, beta: float = 0.0, **params) -> np.ndarray:
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)

    def get_params(self) -> dict:
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("brightness_limit", "contrast_limit", "brightness_by_max")


class RandomBrightness(RandomBrightnessContrast):
    """Randomly change brightness of the input image.

    Args:
        limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, limit: Union[float, Tuple[float, float]] = 0.2, always_apply: bool = False, p: float = 0.5):
        super().__init__(brightness_limit=limit, contrast_limit=0, always_apply=always_apply, p=p)
        warnings.warn("This class has been deprecated. Please use RandomBrightnessContrast", DeprecationWarning)

    def get_transform_init_args(self) -> dict:
        return {"limit": self.brightness_limit}


class RandomContrast(RandomBrightnessContrast):
    """Randomly change contrast of the input image.

    Args:
        limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, limit: Union[float, Tuple[float, float]] = 0.2, always_apply: bool = False, p: float = 0.5):
        super().__init__(brightness_limit=0, contrast_limit=limit, always_apply=always_apply, p=p)
        warnings.warn("This class has been deprecated. Please use RandomBrightnessContrast", DeprecationWarning)

    def get_transform_init_args(self) -> dict:
        return {"limit": self.contrast_limit}


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        clip_limit: Union[float, Tuple[float, float]] = 4.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, img: np.ndarray, clip_limit: float = 2, **params) -> np.ndarray:
        return F.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self) -> dict:
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("clip_limit", "tile_grid_size")
