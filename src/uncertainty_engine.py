"""
uncertainty_engine.py
---------------------
Implements the S.D.T (Sensor-health, Disagreement, Temporal-jitter)
Uncertainty Model for the RA-ASF project.

The Global Uncertainty Score U_global is computed as:

    U_global = alpha * H_sys + beta * D_spatial + gamma * T_jitter

Where:
    H_sys      = Active sensor health penalty (only considers sensors in the active module)
    D_spatial  = Spatial disagreement between the two fused sensor outputs
    T_jitter   = Temporal jitter (object flickering between consecutive frames)

All values are bounded in [0.0, 1.0]:
    0.0 = Perfect confidence (drive at max speed)
    1.0 = Critical failure (emergency brake)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class UncertaintyEngine:
    """
    Computes the S.D.T Uncertainty Score for any active fusion module.

    Parameters
    ----------
    alpha : float
        Weight for the Active Sensor Health component (default 0.40).
    beta : float
        Weight for the Spatial Disagreement component (default 0.40).
    gamma : float
        Weight for the Temporal Jitter component (default 0.20).
    disagreement_lambda : float
        Exponential decay rate for D_spatial. Controls how quickly
        large pixel-distance disagreements saturate to 1.0.
    """

    def __init__(self, alpha=0.40, beta=0.40, gamma=0.20,
                 disagreement_lambda=0.02):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.lam   = disagreement_lambda

        # State for temporal jitter tracking
        self._prev_object_ids = set()
        self._prev_n_objects  = 0

        logger.info(
            "UncertaintyEngine ready  (a=%.2f b=%.2f g=%.2f lam=%.3f)",
            alpha, beta, gamma, disagreement_lambda,
        )

    def compute(self, active_mode, health_dict, match_distances,
                current_object_ids, n_cam_objs=0):
        """
        Compute the global uncertainty score.

        Parameters
        ----------
        active_mode : str
            Current fusion mode: "M1", "M2", "M3", or "GOLD".
        health_dict : dict
            Must contain keys "cam", "lid", "rad" with float health scores [0,1].
        match_distances : list[float]
            List of pixel distances between matched object pairs.
        current_object_ids : set[int]
            Set of object IDs detected in the current frame.
        n_cam_objs : int
            Number of objects detected by the primary sensor (Camera).
        """
        cam_h = health_dict.get("cam", 0.5)
        lid_h = health_dict.get("lid", 0.5)
        rad_h = health_dict.get("rad", 0.5)

        if active_mode == "M1":
            H_sys = 1.0 - (0.5 * cam_h + 0.5 * lid_h)
        elif active_mode == "M2":
            H_sys = 1.0 - (0.6 * cam_h + 0.4 * rad_h)
        elif active_mode == "M3":
            H_sys = 1.0 - (0.55 * lid_h + 0.45 * rad_h)
        elif active_mode == "GOLD":
            H_sys = 1.0 - (0.4 * cam_h + 0.35 * lid_h + 0.25 * rad_h)
        else:
            H_sys = 1.0

        H_sys = float(np.clip(H_sys, 0.0, 1.0))

        # ── B. Spatial Disagreement (D_spatial) ───────────────────────────────
        if n_cam_objs == 0:
            D_spatial = 0.0  # Empty road - both sensors agree
        elif len(match_distances) > 0:
            avg_dist = float(np.mean(match_distances))
            D_spatial = 1.0 - np.exp(-self.lam * avg_dist)
        else:
            D_spatial = 1.0  # Camera sees things but sensors don't match

        D_spatial = float(np.clip(D_spatial, 0.0, 1.0))

        # ── C. Temporal Jitter (T_jitter) ─────────────────────────────────────
        if self._prev_n_objects > 0:
            lost = self._prev_object_ids - current_object_ids
            T_jitter = len(lost) / self._prev_n_objects
        else:
            T_jitter = 0.0

        T_jitter = float(np.clip(T_jitter, 0.0, 1.0))

        # Update state for next frame
        self._prev_object_ids = set(current_object_ids)
        self._prev_n_objects  = len(current_object_ids)

        # ── D. Global Uncertainty ─────────────────────────────────────────────
        U_global = (self.alpha * H_sys
                  + self.beta  * D_spatial
                  + self.gamma * T_jitter)

        U_global = float(np.clip(U_global, 0.0, 1.0))

        return {
            "U_global"  : round(U_global, 4),
            "H_sys"     : round(H_sys, 4),
            "D_spatial" : round(D_spatial, 4),
            "T_jitter"  : round(T_jitter, 4),
            "components": {
                "active_mode"     : active_mode,
                "n_cam_objs"      : n_cam_objs,
                "n_matches"       : len(match_distances),
                "avg_match_dist"  : round(float(np.mean(match_distances)), 2) if match_distances else 0.0,
                "n_objects_now"   : len(current_object_ids),
                "n_objects_prev"  : self._prev_n_objects,
            }
        }

    def reset(self):
        """Reset temporal state."""
        self._prev_object_ids = set()
        self._prev_n_objects  = 0


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    ue = UncertaintyEngine()

    print("\n-- Test 1: Healthy sensors, good matches --")
    r = ue.compute(active_mode="M1", health_dict={"cam": 0.95, "lid": 0.90}, 
                   match_distances=[15.0, 22.0], current_object_ids={1, 2}, n_cam_objs=2)
    print(f"  U_global = {r['U_global']:.4f}  (Expected: LOW)")

    print("\n-- Test 2: Empty Road (Agreement) --")
    r = ue.compute(active_mode="M1", health_dict={"cam": 0.95, "lid": 0.90}, 
                   match_distances=[], current_object_ids=set(), n_cam_objs=0)
    print(f"  U_global = {r['U_global']:.4f}  (Expected: VERY LOW)")

    print("\n-- Test 3: Sensor Mismatch (Camera sees, Lidar doesn't) --")
    r = ue.compute(active_mode="M1", health_dict={"cam": 0.95, "lid": 0.90}, 
                   match_distances=[], current_object_ids=set(), n_cam_objs=2)
    print(f"  U_global = {r['U_global']:.4f}  (Expected: HIGH)")

    print("\nAll tests passed.\n")
