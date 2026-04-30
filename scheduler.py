"""Inverse-temperature schedules for discrete action sampling."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class TemperatureScheduler(ABC):
    @abstractmethod
    def get_inverse_temperature(self) -> float:
        """Return the beta value used for the next rollout."""

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Advance the schedule after PPO has consumed the rollout."""


class FixedScheduler(TemperatureScheduler):
    def __init__(self, value: float):
        self.current_temperature = value

    def get_inverse_temperature(self) -> float:
        return self.current_temperature

    def update(self, **kwargs) -> None:
        del kwargs


class LinearScheduler(TemperatureScheduler):
    def __init__(self, start: float, delta: float, minimum=None, maximum=None):
        self.start = start
        self.delta = delta
        self.minimum = minimum
        self.maximum = maximum
        self.current_temperature = start

    def get_inverse_temperature(self) -> float:
        return self.current_temperature

    def update(self, **kwargs) -> None:
        self.current_temperature += self.delta
        if self.minimum is not None:
            self.current_temperature = max(self.minimum, self.current_temperature)
        if self.maximum is not None:
            self.current_temperature = min(self.maximum, self.current_temperature)


DefaultScheduler = FixedScheduler


class CosineScheduler(TemperatureScheduler):
    def __init__(self, minimum: float, maximum: float, frequency: int):
        self.minimum = minimum
        self.maximum = maximum
        self.frequency = frequency
        self.current_iter = 0
        self.current_temperature = (maximum + minimum) / 2

    def get_inverse_temperature(self) -> float:
        return self.current_temperature

    def update(self, **kwargs) -> None:
        self.current_iter += 1
        self.current_temperature = (self.maximum + self.minimum) / 2 - (
            self.maximum - self.minimum
        ) / 2 * math.cos(2 * math.pi * self.current_iter / self.frequency)
