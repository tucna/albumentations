import numpy as np

from typing import Union, Sequence, Tuple

from . import functional as F
from ...core.transforms_interface import ImageOnlyTransform


__all__ = ["Normalize"]


class Normalize(ImageOnlyTransform):
    """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.

    Args:
        mean (float, list of float): mean values. By default equal to ImageNet values. Default: (0.485, 0.456, 0.406)
        std  (float, list of float): std values. By default equal to ImageNet values. Default: (0.229, 0.224, 0.225)
        max_pixel_value (float): maximum possible pixel value. Default: 255

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        mean: Union[float, np.ndarray, Sequence[float]] = (0.485, 0.456, 0.406),
        std: Union[float, np.ndarray, Sequence[float]] = (0.229, 0.224, 0.225),
        max_pixel_value: float = 255.0,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image: np.ndarray, **params) -> np.ndarray:  # type: ignore
        return F.normalize(image, self.mean, self.std, self.max_pixel_value)

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("mean", "std", "max_pixel_value")
