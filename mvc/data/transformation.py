"""
@Time    : 2020/12/14 15:20
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : transformation.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import List, Union, Dict, Any

import numpy as np
from scipy.interpolate import CubicSpline


def random_curve(length, sigma=0.2, knots=4):
    xx = np.arange(0, length, (length - 1) / (knots + 1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=knots + 2)
    x_range = np.arange(length)
    interpolator = CubicSpline(xx, yy)

    return interpolator(x_range)


class Transformation(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def apply(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        """

        Parameters
        ----------
        x : (*, length)
        """
        pass


class Jittering(Transformation):
    def __init__(self, loc: float = 0.0, sigma: float = 1.0):
        super(Jittering, self).__init__()

        self.loc = loc
        self.sigma = sigma

    def apply(self, x: np.ndarray):
        return x + self.sigma * np.random.randn(*x.shape) + self.loc

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            return self.apply(x)
        else:
            x['mid'] = self.apply(x['mid'])
            return x


class HorizontalFlipping(Transformation):
    def __init__(self, randomize: bool = True):
        super(HorizontalFlipping, self).__init__()

        self.randomize = randomize

    def apply(self, x: np.ndarray):
        if self.randomize:
            return np.flip(x, axis=-1) if np.random.randint(low=0, high=2) == 1 else x
        else:
            return np.flip(x, axis=-1)

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            return self.apply(x)
        else:
            x['mid'] = self.apply(x['mid'])
            return x


class VerticalFlipping(Transformation):
    def __init__(self, randomize: bool = True):
        super(VerticalFlipping, self).__init__()

        self.randomize = randomize

    def apply(self, x: np.ndarray):
        if self.randomize:
            return -x if np.random.randint(low=0, high=2) == 1 else x
        else:
            return -x

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            return self.apply(x)
        else:
            x['mid'] = self.apply(x['mid'])
            return x


class MagnitudeWarping(Transformation):
    def __init__(self, sigma: float = 1.0, knots: int = 4):
        super(MagnitudeWarping, self).__init__()

        self.sigma = sigma
        self.knots = knots

    def apply(self, x: np.ndarray):
        return x * random_curve(x.shape[-1], self.sigma, self.knots)

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            return self.apply(x)
        else:
            x['mid'] = self.apply(x['mid'])
            return x


class TimeWarping(Transformation):
    def __init__(self, sigma: float = 1.0, knots: int = 4):
        super(TimeWarping, self).__init__()

        self.sigma = sigma
        self.knots = knots

    def apply(self, x: np.ndarray):
        length = x.shape[-1]
        out = []
        for i in range(x.shape[-2]):
            timestamps = random_curve(length, self.sigma, self.knots)
            timestamps_cumsum = np.cumsum(timestamps)
            scale = (length - 1) / timestamps_cumsum[-1]
            timestamps_new = timestamps_cumsum * scale
            x_range = np.arange(length)
            out.append(np.interp(x_range, timestamps_new, x[i]))

        return np.stack(out)

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            return self.apply(x)
        else:
            x['mid'] = self.apply(x['mid'])
            return x


class Scaling(Transformation):
    def __init__(self, scale_factor: float = 0.2, direction: str = 'both', randomize: bool = True):
        super(Scaling, self).__init__()

        assert direction in ['both', 'increase', 'decrease']

        self.scale_factor = scale_factor
        self.direction = direction
        self.randomize = randomize

    def apply(self, x: np.ndarray):
        if self.direction == 'both':
            if self.randomize:
                return x * (1.0 + self.scale_factor * np.random.randint(low=-1, high=2))
            else:
                raise ValueError('`randomize` must be `True` when `direction` is `both`!')
        elif self.direction == 'increase':
            if self.randomize:
                return x * (1.0 + self.scale_factor * np.random.randint(low=0, high=2))
            else:
                return x * (1.0 + self.scale_factor)
        elif self.direction == 'decrease':
            if self.randomize:
                return x * (1.0 + self.scale_factor * np.random.randint(low=-1, high=1))
            else:
                return x * (1.0 - self.scale_factor)
        else:
            raise ValueError

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            return self.apply(x)
        else:
            x['mid'] = self.apply(x['mid'])
            return x


class RandomCropping(Transformation):
    def __init__(self, size: int, fill: float = 0.0, padding_mode: str = 'constant'):
        super(RandomCropping, self).__init__()

        assert padding_mode in ['constant', 'edge']

        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def apply(self, x: np.ndarray):
        start_idx = np.random.randint(low=0, high=x.shape[-1] - self.size)
        if self.padding_mode == 'constant':
            out = np.full_like(x, fill_value=self.fill, dtype=x.dtype)
            out[:, start_idx: start_idx + self.size] = x[:, start_idx: start_idx + self.size]
        elif self.padding_mode == 'edge':
            out = np.zeros_like(x, dtype=x.dtype)
            out[:, start_idx: start_idx + self.size] = x[:, start_idx: start_idx + self.size]
            if start_idx + self.size < x.shape[-1]:
                out[:, start_idx + self.size:] = x[:, start_idx + self.size: start_idx + self.size + 1]
            if start_idx > 0:
                out[:, 0: start_idx] = x[:, start_idx: start_idx + 1]
        else:
            raise ValueError

        return out

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            return self.apply(x)
        else:
            x['mid'] = self.apply(x['mid'])
            return x


class ChannelShuffling(Transformation):
    def __init__(self):
        super(ChannelShuffling, self).__init__()

    def apply(self, x: np.ndarray):
        channel_indices = np.arange(x.shape[-2])
        np.random.shuffle(channel_indices)

        return x[channel_indices, :]

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            return self.apply(x)
        else:
            x['mid'] = self.apply(x['mid'])
            return x


class Perturbation(Transformation):
    def __init__(self, min_perturbation, max_perturbation):
        super(Perturbation, self).__init__()

        self.min_perturbation = min_perturbation
        self.max_perturbation = max_perturbation

    def apply(self, x: np.ndarray):
        pass

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            raise ValueError('Only support for Dict input!')
        else:
            assert self.max_perturbation <= x['head'].shape[-1]
            num_perturbation = np.random.randint(self.min_perturbation, self.max_perturbation + 1)
            sign = np.random.choice([-1, 1])
            if sign == -1:
                out = np.concatenate([x['head'][:, -num_perturbation:], x['mid'][:, :-num_perturbation]], axis=-1)
            else:
                out = np.concatenate([x['mid'][:, num_perturbation:], x['tail'][:, :num_perturbation]], axis=-1)

            x['mid'] = out
            return x


class TwoCropsTransform(Transformation):
    def __init__(self, transform: Transformation):
        super(TwoCropsTransform, self).__init__()

        self.transform = transform

    def apply(self, x: np.ndarray):
        pass

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        return [self.transform(x), self.transform(x)]


class Compose(Transformation):
    def __init__(self, trans: List[Transformation]):
        super(Transformation, self).__init__()

        self.trans = trans

    def apply(self, x: np.ndarray):
        pass

    def __call__(self, x: np.ndarray):
        out = x
        for transformation in self.trans:
            out = transformation(out)

        return out
