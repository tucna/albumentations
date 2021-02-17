import random
import numpy as np

from typing import Union, Tuple, Sequence

from . import functional as F
from ...core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = ["HueSaturationValue", "Solarize", "RGBShift", "ChannelDropout"]

NumberOrRange = Union[int, float, Union[Tuple[int, int], Tuple[float, float]]]


class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

    Args:
        hue_shift_limit ((int, int) or int): range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        sat_shift_limit ((int, int) or int): range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        val_shift_limit ((int, int) or int): range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        hue_shift_limit: NumberOrRange = 20,
        sat_shift_limit: NumberOrRange = 30,
        val_shift_limit: NumberOrRange = 20,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(HueSaturationValue, self).__init__(always_apply, p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(  # type: ignore
        self,
        image: np.ndarray,
        hue_shift: NumberOrRange = 0,
        sat_shift: NumberOrRange = 0,
        val_shift: NumberOrRange = 0,
        **params
    ) -> np.ndarray:
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self) -> dict:
        return {
            "hue_shift": random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
            "sat_shift": random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
            "val_shift": random.uniform(self.val_shift_limit[0], self.val_shift_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")


class Solarize(ImageOnlyTransform):
    """Invert all pixel values above a threshold.

    Args:
        threshold ((int, int) or int, or (float, float) or float): range for solarizing threshold.
        If threshold is a single value, the range will be [threshold, threshold]. Default: 128.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        any
    """

    def __init__(self, threshold: NumberOrRange = 128, always_apply: bool = False, p: float = 0.5):
        super(Solarize, self).__init__(always_apply, p)

        if isinstance(threshold, (int, float)):
            self.threshold = to_tuple(threshold, low=threshold)
        else:
            self.threshold = to_tuple(threshold, low=0)

    def apply(self, image: np.ndarray, threshold: Union[int, float] = 0, **params) -> np.ndarray:  # type: ignore
        return F.solarize(image, threshold)

    def get_params(self) -> dict:
        return {"threshold": random.uniform(self.threshold[0], self.threshold[1])}

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("threshold",)


class RGBShift(ImageOnlyTransform):
    """Randomly shift values for each channel of the input RGB image.

    Args:
        r_shift_limit ((int, int) or int): range for changing values for the red channel. If r_shift_limit is a single
            int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
        g_shift_limit ((int, int) or int): range for changing values for the green channel. If g_shift_limit is a
            single int, the range  will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
        b_shift_limit ((int, int) or int): range for changing values for the blue channel. If b_shift_limit is a single
            int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        r_shift_limit: NumberOrRange = 20,
        g_shift_limit: NumberOrRange = 20,
        b_shift_limit: NumberOrRange = 20,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(RGBShift, self).__init__(always_apply, p)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(  # type: ignore
        self,
        image: np.ndarray,
        r_shift: Union[int, float] = 0,
        g_shift: Union[int, float] = 0,
        b_shift: Union[int, float] = 0,
        **params
    ) -> np.ndarray:
        return F.shift_rgb(image, r_shift, g_shift, b_shift)

    def get_params(self) -> dict:
        return {
            "r_shift": random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
            "g_shift": random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
            "b_shift": random.uniform(self.b_shift_limit[0], self.b_shift_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("r_shift_limit", "g_shift_limit", "b_shift_limit")


class ChannelDropout(ImageOnlyTransform):
    """Randomly Drop Channels in the input Image.

    Args:
        channel_drop_range (int, int): range from which we choose the number of channels to drop.
        fill_value (int, float): pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, unit32, float32
    """

    def __init__(
        self,
        channel_drop_range: Tuple[int, int] = (1, 1),
        fill_value: Union[int, float] = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(ChannelDropout, self).__init__(always_apply, p)

        self.channel_drop_range = channel_drop_range

        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]

        if not 1 <= self.min_channels <= self.max_channels:
            raise ValueError("Invalid channel_drop_range. Got: {}".format(channel_drop_range))

        self.fill_value = fill_value

    def apply(
        self, img: np.ndarray, channels_to_drop: Union[int, Sequence[int], np.ndarray] = (0,), **params
    ) -> np.ndarray:
        return F.channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        img = params["image"]

        num_channels = img.shape[-1]

        if len(img.shape) == 2 or num_channels == 1:
            raise NotImplementedError("Images has one channel. ChannelDropout is not defined.")

        if self.max_channels >= num_channels:
            raise ValueError("Can not drop all channels in ChannelDropout.")

        num_drop_channels = random.randint(self.min_channels, self.max_channels)

        channels_to_drop = random.sample(range(num_channels), k=num_drop_channels)

        return {"channels_to_drop": channels_to_drop}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("channel_drop_range", "fill_value")

    @property
    def targets_as_params(self) -> Sequence[str]:
        return ["image"]
