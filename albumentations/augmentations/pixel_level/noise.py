import random
import numpy as np

from typing import Sequence, Tuple, Union

from . import functional as F
from ...core.transforms_interface import ImageOnlyTransform

__all__ = ["GaussNoise", "ISONoise"]


class GaussNoise(ImageOnlyTransform):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        var_limit: Union[float, Tuple[float, float]] = (10.0, 50.0),
        mean: float = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )

        self.mean = mean

    def apply(self, img: np.ndarray, gauss: np.ndarray = None, **params) -> np.ndarray:
        return F.gauss_noise(img, gauss=gauss)

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

        gauss = random_state.normal(self.mean, sigma, image.shape)
        return {"gauss": gauss}

    @property
    def targets_as_params(self) -> Sequence[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("var_limit",)


class ISONoise(ImageOnlyTransform):
    """
    Apply camera sensor noise.

    Args:
        color_shift (float, float): variance range for color hue change.
            Measured as a fraction of 360 degree Hue angle in HLS colorspace.
        intensity ((float, float): Multiplicative factor that control strength
            of color and luminace noise.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        color_shift: Tuple[float, float] = (0.01, 0.05),
        intensity: Tuple[float, float] = (0.1, 0.5),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.intensity = intensity
        self.color_shift = color_shift

    def apply(
        self, img: np.ndarray, color_shift: float = 0.05, intensity: float = 1.0, random_state: int = None, **params
    ) -> np.ndarray:
        return F.iso_noise(img, color_shift, intensity, np.random.RandomState(random_state))

    def get_params(self) -> dict:
        return {
            "color_shift": random.uniform(self.color_shift[0], self.color_shift[1]),
            "intensity": random.uniform(self.intensity[0], self.intensity[1]),
            "random_state": random.randint(0, 65536),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("intensity", "color_shift")
