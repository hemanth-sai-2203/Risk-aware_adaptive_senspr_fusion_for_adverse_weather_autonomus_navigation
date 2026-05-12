"""
npc_manager.py
--------------
Spawns and destroys NPC vehicles and pedestrians in CARLA.
Called by data_collector.py before collection starts.

Design:
  - Spawns vehicles on random spawn points (excluding ego vehicle's point)
  - Spawns pedestrians on random sidewalk points
  - All NPCs use autopilot / AI controller
  - destroy_all() cleans up cleanly before the next weather state

Python 3.7 | Windows | CARLA 0.9.15
"""

import logging
import random
import time

import carla

logger = logging.getLogger(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
N_VEHICLES    = 40    # increased number of NPC vehicles
N_PEDESTRIANS = 20    # increased number of NPC pedestrians

# Vehicle blueprints to use (common road vehicles only — no bikes or emergency)
VEHICLE_FILTERS = [
    "vehicle.audi.*",
    "vehicle.bmw.*",
    "vehicle.chevrolet.*",
    "vehicle.citroen.*",
    "vehicle.dodge.*",
    "vehicle.ford.*",
    "vehicle.jeep.*",
    "vehicle.lincoln.*",
    "vehicle.mercedes.*",
    "vehicle.mini.*",
    "vehicle.mustang.*",
    "vehicle.nissan.*",
    "vehicle.seat.*",
    "vehicle.tesla.cybertruck",
    "vehicle.toyota.*",
    "vehicle.volkswagen.*",
]


class NpcManager:
    """
    Manages NPC vehicles and pedestrians for data collection.

    Usage
    -----
    npc = NpcManager(client, world, tm_port=8000)
    npc.spawn_all(ego_vehicle)
    # ... collect data ...
    npc.destroy_all()
    """

    def __init__(self, client, world, tm_port=8000):
        self._client   = client
        self._world    = world
        self._tm_port  = tm_port
        self._vehicles = []
        self._walkers  = []
        self._walker_controllers = []

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def spawn_all(self, ego_vehicle):
        """
        Spawn NPC vehicles and pedestrians.
        ego_vehicle is passed so its spawn point is excluded.
        """
        self._spawn_vehicles(ego_vehicle)
        self._spawn_pedestrians()
        logger.info(
            "NPCs ready: %d vehicles + %d pedestrians",
            len(self._vehicles), len(self._walkers),
        )

    def destroy_all(self):
        """Destroy all NPCs. Call before changing weather or ending episode."""
        # Stop walker AI controllers first
        for ctrl in self._walker_controllers:
            try:
                ctrl.stop()
            except Exception:
                pass

        # Batch destroy for speed
        actors_to_destroy = (
            self._walker_controllers
            + self._walkers
            + self._vehicles
        )
        if actors_to_destroy:
            self._client.apply_batch([
                carla.command.DestroyActor(a) for a in actors_to_destroy
                if a is not None
            ])

        self._vehicles.clear()
        self._walkers.clear()
        self._walker_controllers.clear()
        logger.info("All NPCs destroyed.")

    @property
    def vehicle_count(self):
        return len(self._vehicles)

    @property
    def pedestrian_count(self):
        return len(self._walkers)

    # ── PRIVATE: VEHICLES ─────────────────────────────────────────────────────

    def _spawn_vehicles(self, ego_vehicle):
        """Spawn NPC vehicles using batch commands for speed."""
        lib          = self._world.get_blueprint_library()
        spawn_points = self._world.get_map().get_spawn_points()

        if not spawn_points:
            logger.warning("No spawn points — cannot spawn NPC vehicles.")
            return

        # Exclude the ego vehicle's current spawn point
        ego_loc = ego_vehicle.get_location()
        spawn_points = [
            sp for sp in spawn_points
            if sp.location.distance(ego_loc) > 10.0
        ]
        random.shuffle(spawn_points)

        # Collect usable blueprints
        vehicle_bps = []
        for filt in VEHICLE_FILTERS:
            vehicle_bps.extend(lib.filter(filt))
        if not vehicle_bps:
            vehicle_bps = list(lib.filter("vehicle.*"))

        # Remove bikes and motorcycles (too narrow — bad bounding boxes)
        vehicle_bps = [
            bp for bp in vehicle_bps
            if int(bp.get_attribute("number_of_wheels")) >= 4
        ]

        # Build batch spawn commands
        batch    = []
        n_target = min(N_VEHICLES, len(spawn_points))

        for i in range(n_target):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(
                    bp.get_attribute("color").recommended_values
                ))
            batch.append(
                carla.command.SpawnActor(bp, spawn_points[i])
                .then(carla.command.SetAutopilot(
                    carla.command.FutureActor, True, self._tm_port
                ))
            )

        # Execute batch
        results = self._client.apply_batch_sync(batch, True)

        tm = self._client.get_trafficmanager(self._tm_port)
        for result in results:
            if result.error:
                continue
            actor = self._world.get_actor(result.actor_id)
            if actor is None:
                continue
            self._vehicles.append(actor)

            # NPC behaviour settings
            tm.auto_lane_change(actor, True)
            tm.ignore_lights_percentage(actor, 0)       # NPCs obey lights
            tm.distance_to_leading_vehicle(actor, 1.5)
            # Vary NPC speeds ±30% for realism
            speed_var = random.uniform(-30, 30)
            tm.vehicle_percentage_speed_difference(actor, speed_var)

        logger.info("Spawned %d / %d NPC vehicles", len(self._vehicles), n_target)

    # ── PRIVATE: PEDESTRIANS ──────────────────────────────────────────────────

    def _spawn_pedestrians(self):
        """Spawn pedestrians and their AI controllers."""
        lib = self._world.get_blueprint_library()

        # Get random sidewalk spawn points
        spawn_points = []
        for _ in range(N_PEDESTRIANS * 3):   # try 3x more points than needed
            loc = self._world.get_random_location_from_navigation()
            if loc is not None:
                spawn_points.append(carla.Transform(loc))
            if len(spawn_points) >= N_PEDESTRIANS * 2:
                break

        random.shuffle(spawn_points)

        walker_bps = list(lib.filter("walker.pedestrian.*"))
        if not walker_bps:
            logger.warning("No pedestrian blueprints found — skipping pedestrians.")
            return

        controller_bp = lib.find("controller.ai.walker")

        # Spawn walkers
        walker_batch = []
        for i in range(min(N_PEDESTRIANS, len(spawn_points))):
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker_batch.append(carla.command.SpawnActor(bp, spawn_points[i]))

        walker_results = self._client.apply_batch_sync(walker_batch, True)

        # Spawn AI controllers for each walker
        ctrl_batch = []
        valid_walkers = []
        for result in walker_results:
            if result.error:
                continue
            actor = self._world.get_actor(result.actor_id)
            if actor is None:
                continue
            valid_walkers.append(actor)
            ctrl_batch.append(
                carla.command.SpawnActor(
                    controller_bp,
                    carla.Transform(),
                    actor,
                )
            )

        ctrl_results = self._client.apply_batch_sync(ctrl_batch, True)

        for i, result in enumerate(ctrl_results):
            if result.error:
                continue
            ctrl = self._world.get_actor(result.actor_id)
            if ctrl is None:
                continue
            self._walker_controllers.append(ctrl)
            self._walkers.append(valid_walkers[i])

        # Tick world once so controllers initialise before we start them
        self._world.tick()

        # Start each controller walking to a random destination
        for ctrl in self._walker_controllers:
            try:
                ctrl.start()
                ctrl.go_to_location(
                    self._world.get_random_location_from_navigation()
                )
                ctrl.set_max_speed(random.uniform(1.0, 2.0))
            except Exception as exc:
                logger.debug("Walker controller start failed: %s", exc)

        logger.info("Spawned %d pedestrians", len(self._walkers))
