"""
weather_engine.py
-----------------
WeatherEngine : applies CARLA 0.9.15 weather presets and schedules
dynamic transitions mid-episode.

Compatible with CARLA 0.9.15 on Windows.
CARLA 0.9.15 added scattering_intensity and mie_scattering_scale to
WeatherParameters — we deliberately do NOT set these so the code remains
backward-compatible with 0.9.14 as well.
"""

import carla
import logging
from config import WEATHER_PARAMS

logger = logging.getLogger(__name__)


class WeatherEngine:
    """
    Wraps CARLA weather management.

    Parameters
    ----------
    world            : carla.World
    schedule         : list[str] | None
        Weather state names to cycle through automatically.
        Each element must be a key in WEATHER_PARAMS.
    ticks_per_state  : int
        Simulation ticks to hold each state before transitioning.
    """

    VALID_STATES = set(WEATHER_PARAMS.keys())

    def __init__(self, world, schedule=None, ticks_per_state=200):
        self._world           = world
        self._schedule        = schedule or []
        self._ticks_per_state = ticks_per_state
        self._schedule_index  = 0
        self._tick_counter    = 0
        self._current_state   = None

        if self._schedule:
            self.set_weather(self._schedule[0])

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def set_weather(self, state):
        """Immediately apply a named weather state to the CARLA world."""
        if state not in self.VALID_STATES:
            raise ValueError(
                "Unknown weather state '{}'. Valid: {}".format(
                    state, sorted(self.VALID_STATES)
                )
            )
        p = WEATHER_PARAMS[state]
        weather = carla.WeatherParameters(
            cloudiness             = p["cloudiness"],
            precipitation          = p["precipitation"],
            precipitation_deposits = p["precipitation_deposits"],
            wind_intensity         = p["wind_intensity"],
            sun_azimuth_angle      = p["sun_azimuth_angle"],
            sun_altitude_angle     = p["sun_altitude_angle"],
            fog_density            = p["fog_density"],
            fog_distance           = p["fog_distance"],
            wetness                = p["wetness"],
        )
        self._world.set_weather(weather)
        self._current_state = state
        logger.info("Weather -> %s", state)

    def step(self):
        """
        Advance the dynamic schedule by one tick.
        Call inside the main simulation loop when using a schedule.

        Returns
        -------
        str : current weather state name
        """
        if not self._schedule:
            return self._current_state or "unknown"

        self._tick_counter += 1
        if self._tick_counter >= self._ticks_per_state:
            self._tick_counter   = 0
            self._schedule_index = (self._schedule_index + 1) % len(self._schedule)
            self.set_weather(self._schedule[self._schedule_index])

        return self._current_state

    @property
    def current_state(self):
        return self._current_state or "unknown"

    @property
    def fog_density(self):
        if self._current_state is None:
            return 0.0
        return WEATHER_PARAMS[self._current_state].get("fog_density", 0.0)
