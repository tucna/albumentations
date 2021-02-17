import random
import warnings
import numpy as np

from enum import IntEnum
from typing import Union, Tuple

from . import functional as F
from ...core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = [
    "JpegCompression",
    "ImageCompression",
    "Posterize",
]


class ImageCompression(ImageOnlyTransform):
    """Decrease Jpeg, WebP compression of an image.

    Args:
        quality_lower (int): lower bound on the image quality.
            Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper (int): upper bound on the image quality.
            Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type (ImageCompressionType): should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
            Default: ImageCompressionType.JPEG

    Targets:
        image

    Image types:
        uint8, float32
    """

    class ImageCompressionType(IntEnum):
        JPEG = 0
        WEBP = 1

    def __init__(
        self,
        quality_lower: int = 99,
        quality_upper: int = 100,
        compression_type: Union[int, ImageCompressionType] = ImageCompressionType.JPEG,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        self.compression_type = ImageCompression.ImageCompressionType(compression_type)
        low_thresh_quality_assert = 0

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            low_thresh_quality_assert = 1

        if not low_thresh_quality_assert <= quality_lower <= 100:
            raise ValueError(f"Invalid quality_lower. Got: {quality_lower}")
        if not low_thresh_quality_assert <= quality_upper <= 100:
            raise ValueError(f"Invalid quality_upper. Got: {quality_upper}")

        self.quality_lower: int = quality_lower
        self.quality_upper: int = quality_upper

    def apply(  # type: ignore
        self, image: np.ndarray, quality: int = 100, image_type: str = ".jpg", **params
    ) -> np.ndarray:
        return F.image_compression(image, quality, image_type)

    def get_params(self) -> dict:
        image_type = ".jpg"

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            image_type = ".webp"

        return {"quality": random.randint(self.quality_lower, self.quality_upper), "image_type": image_type}

    def get_transform_init_args(self) -> dict:
        return {
            "quality_lower": self.quality_lower,
            "quality_upper": self.quality_upper,
            "compression_type": self.compression_type.value,
        }


class JpegCompression(ImageCompression):
    """Decrease Jpeg compression of an image.

    Args:
        quality_lower (int): lower bound on the jpeg quality. Should be in [0, 100] range
        quality_upper (int): upper bound on the jpeg quality. Should be in [0, 100] range

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, quality_lower: int = 99, quality_upper: int = 100, always_apply: bool = False, p: float = 0.5):
        super().__init__(
            quality_lower=quality_lower,
            quality_upper=quality_upper,
            compression_type=ImageCompression.ImageCompressionType.JPEG,
            always_apply=always_apply,
            p=p,
        )
        warnings.warn("This class has been deprecated. Please use ImageCompression", DeprecationWarning)

    def get_transform_init_args(self) -> dict:
        return {"quality_lower": self.quality_lower, "quality_upper": self.quality_upper}


class Posterize(ImageOnlyTransform):
    """Reduce the number of bits for each color channel.

    Args:
        num_bits ((int, int) or int,
                  or list of ints [r, g, b],
                  or list of ints [[r1, r1], [g1, g2], [b1, b2]]): number of high bits.
            If num_bits is a single value, the range will be [num_bits, num_bits].
            Must be in range [0, 8]. Default: 4.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        num_bits: Union[
            int, Tuple[int, int], Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        ] = 4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        if isinstance(num_bits, (list, tuple)):
            if len(num_bits) == 3:
                self.num_bits = [to_tuple(i, 0) for i in num_bits]
            else:
                self.num_bits = to_tuple(num_bits, 0)
        else:
            self.num_bits = to_tuple(num_bits, num_bits)

    def apply(self, image: np.ndarray, num_bits: Union[int, np.ndarray] = 1, **params) -> np.ndarray:  # type: ignore
        return F.posterize(image, num_bits)

    def get_params(self) -> dict:
        if len(self.num_bits) == 3:
            return {"num_bits": [random.randint(i[0], i[1]) for i in self.num_bits]}
        return {"num_bits": random.randint(self.num_bits[0], self.num_bits[1])}

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("num_bits",)
