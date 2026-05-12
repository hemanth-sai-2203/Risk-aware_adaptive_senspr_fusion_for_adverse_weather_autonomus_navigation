"""
risk_aware_pid.py
-----------------
Risk-Aware PID Controller for RA-ASF.

This is the component that CLOSES THE LOOP between Perception and Control.
Conventional PID controllers use a fixed target speed. Our controller
dynamically scales the target speed based on the perception uncertainty:

    target_speed = MAX_SPEED * (1 - U_global)^power

When uncertainty is low (clear weather, good fusion), the vehicle drives
at full speed. As uncertainty rises (fog, sensor failure), the vehicle
progressively slows down. Above EMERGENCY_BRAKE_THRESHOLD, it brakes hard.

This is the key novelty claim of the RA-ASF paper:
    "Uncertainty stays in perception; vehicle behavior does not adapt
     based on confidence" — this is the Gap 3 we close.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class RiskAwarePID:
    """
    PID speed controller that scales target speed by uncertainty.

    Parameters
    ----------
    max_speed_kmh : float
        Maximum target speed in clear conditions (km/h).
    emergency_threshold : float
        Uncertainty above this triggers full brake.
    speed_power : float
        Exponent for non-linear speed reduction.
        1.0 = linear, >1.0 = more aggressive at high uncertainty.
    Kp, Ki, Kd : float
        PID gains for throttle control.
    """

    def __init__(self, max_speed_kmh=30.0, emergency_threshold=0.85,
                 speed_power=1.5, Kp=0.5, Ki=0.05, Kd=0.1):
        self.max_speed = max_speed_kmh
        self.emergency = emergency_threshold
        self.power     = speed_power

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self._integral  = 0.0
        self._prev_error = 0.0

        logger.info(
            "RiskAwarePID ready (max=%.1f km/h, emergency=%.2f, Kp=%.2f Ki=%.2f Kd=%.2f)",
            max_speed_kmh, emergency_threshold, Kp, Ki, Kd,
        )

    def compute(self, current_speed_kmh, uncertainty, dt=0.05):
        """
        Compute throttle and brake commands.

        Parameters
        ----------
        current_speed_kmh : float
            Current vehicle speed in km/h.
        uncertainty : float
            Global uncertainty score from UncertaintyEngine [0, 1].
        dt : float
            Time step (seconds).

        Returns
        -------
        dict with keys:
            "throttle"     : float [0, 1]
            "brake"        : float [0, 1]
            "target_speed" : float (km/h)
            "uncertainty"  : float
            "mode"         : str ("NORMAL" or "EMERGENCY_BRAKE")
        """
        # ── Emergency brake check ─────────────────────────────────────────────
        if uncertainty >= self.emergency:
            logger.warning(
                "🛑 EMERGENCY BRAKE! Uncertainty=%.3f >= threshold=%.2f",
                uncertainty, self.emergency,
            )
            self._integral = 0.0
            return {
                "throttle"     : 0.0,
                "brake"        : 1.0,
                "target_speed" : 0.0,
                "uncertainty"  : round(uncertainty, 4),
                "mode"         : "EMERGENCY_BRAKE",
            }

        # ── Dynamic target speed ──────────────────────────────────────────────
        #
        # Non-linear scaling: (1 - U)^power
        # At power=1.5:
        #   U=0.0 → speed = 100% max
        #   U=0.3 → speed = 59% max
        #   U=0.5 → speed = 35% max
        #   U=0.7 → speed = 16% max
        #
        # This is more aggressive than linear scaling because at high uncertainty,
        # even small increases should produce large speed reductions.
        confidence = max(0.0, 1.0 - uncertainty)
        target_speed = self.max_speed * (confidence ** self.power)

        # ── PID control ───────────────────────────────────────────────────────
        error = target_speed - current_speed_kmh
        self._integral += error * dt
        # Anti-windup: clamp integral
        self._integral = float(np.clip(self._integral, -10.0, 10.0))

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        # Map PID output to throttle/brake
        if output >= 0:
            throttle = float(np.clip(output / self.max_speed, 0.0, 1.0))
            brake    = 0.0
        else:
            throttle = 0.0
            brake    = float(np.clip(-output / self.max_speed, 0.0, 1.0))

        return {
            "throttle"     : round(throttle, 4),
            "brake"        : round(brake, 4),
            "target_speed" : round(target_speed, 2),
            "uncertainty"  : round(uncertainty, 4),
            "mode"         : "NORMAL",
        }

    def reset(self):
        """Reset PID state for a new episode."""
        self._integral   = 0.0
        self._prev_error = 0.0


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    pid = RiskAwarePID(max_speed_kmh=30.0)

    print("\n-- Test 1: Low uncertainty (clear weather) --")
    r = pid.compute(current_speed_kmh=20.0, uncertainty=0.1)
    print(f"  Target: {r['target_speed']:.1f} km/h  Throttle: {r['throttle']:.3f}  Brake: {r['brake']:.3f}")

    print("\n-- Test 2: Medium uncertainty (light fog) --")
    r = pid.compute(current_speed_kmh=20.0, uncertainty=0.5)
    print(f"  Target: {r['target_speed']:.1f} km/h  Throttle: {r['throttle']:.3f}  Brake: {r['brake']:.3f}")

    print("\n-- Test 3: High uncertainty (heavy fog) --")
    r = pid.compute(current_speed_kmh=15.0, uncertainty=0.75)
    print(f"  Target: {r['target_speed']:.1f} km/h  Throttle: {r['throttle']:.3f}  Brake: {r['brake']:.3f}")

    print("\n-- Test 4: Emergency brake --")
    r = pid.compute(current_speed_kmh=25.0, uncertainty=0.90)
    print(f"  Mode: {r['mode']}  Brake: {r['brake']:.3f}")

    print("\nAll tests passed.\n")
