"""
Benchmark Scenarios

Standardized test cases inspired by RiMEA and ISO 20414 for validating
pedestrian simulation models. These allow reproducible evaluation of
movement models, collision avoidance, and evacuation dynamics.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from engine.simulation_engine import SimulationEngine, AgentType, HumanRole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers for simple test cases
# ---------------------------------------------------------------------------

def create_corridor_geometry(length: float = 20.0, width: float = 3.0) -> Dict:
    """
    Create a simple corridor geometry with two spaces (start and end).
    Returns a minimal BIM-like spatial dict for testing.
    """
    return {
        "spaces": {
            "start": {"center": (0.0, width / 2, 0.0), "area": width * 5.0, "category": "corridor", "level": "0"},
            "corridor": {"center": (length / 2, width / 2, 0.0), "area": length * width, "category": "corridor", "level": "0"},
            "end": {"center": (length, width / 2, 0.0), "area": width * 5.0, "category": "corridor", "level": "0"},
        },
        "walls": [
            {"bounds": ((-1, 0, 0), (length + 1, 0.1, 2.0))},  # bottom wall
            {"bounds": ((-1, width - 0.1, 0), (length + 1, width, 2.0))},  # top wall
        ],
        "connections": [
            ("start", "corridor", "corridor"),
            ("corridor", "end", "corridor"),
        ]
    }


def create_room_with_exit_geometry(room_width: float = 10.0, room_depth: float = 10.0,
                                   door_width: float = 1.0, door_position: float = 5.0) -> Dict:
    """Create a room with a single exit (door)."""
    return {
        "spaces": {
            "room": {"center": (room_width / 2, room_depth / 2, 0.0), "area": room_width * room_depth,
                     "category": "public", "level": "0"},
            "outside": {"center": (room_width + 3.0, door_position, 0.0), "area": 20.0,
                        "category": "public", "level": "0"},
        },
        "walls": [
            {"bounds": ((0, 0, 0), (room_width, 0.1, 2.0))},  # bottom
            {"bounds": ((0, room_depth - 0.1, 0), (room_width, room_depth, 2.0))},  # top
            {"bounds": ((0, 0, 0), (0.1, room_depth, 2.0))},  # left
            {"bounds": ((room_width - 0.1, 0, 0), (room_width, door_position - door_width / 2, 2.0))},  # right-bottom
            {"bounds": ((room_width - 0.1, door_position + door_width / 2, 0),
                        (room_width, room_depth, 2.0))},  # right-top
        ],
        "connections": [
            ("room", "outside", "door"),
        ]
    }


def create_bottleneck_geometry(corridor_width: float = 3.0, bottleneck_width: float = 1.0,
                                length_before: float = 10.0, length_after: float = 10.0) -> Dict:
    """Create a corridor that narrows to a bottleneck."""
    total_length = length_before + 3.0 + length_after
    return {
        "spaces": {
            "before": {"center": (length_before / 2, corridor_width / 2, 0.0),
                       "area": length_before * corridor_width, "category": "corridor", "level": "0"},
            "bottleneck": {"center": (length_before + 1.5, bottleneck_width / 2, 0.0),
                           "area": 3.0 * bottleneck_width, "category": "corridor", "level": "0"},
            "after": {"center": (length_before + 3.0 + length_after / 2, corridor_width / 2, 0.0),
                      "area": length_after * corridor_width, "category": "corridor", "level": "0"},
        },
        "walls": [
            # Before: top and bottom walls
            {"bounds": ((-1, 0, 0), (length_before + 3.0, 0.1, 2.0))},  # bottom
            {"bounds": ((-1, corridor_width - 0.1, 0), (length_before, corridor_width, 2.0))},  # top before
            {"bounds": ((length_before + 3.0, corridor_width - 0.1, 0), (total_length + 1, corridor_width, 2.0))},  # top after
            # Bottleneck walls
            {"bounds": ((length_before, bottleneck_width - 0.1, 0), (length_before + 3.0, corridor_width, 2.0))},  # top bottleneck
            {"bounds": ((length_before, 0, 0), (length_before + 3.0, corridor_width - bottleneck_width, 2.0))},  # bottom bottleneck
            # End walls
            {"bounds": ((total_length, 0, 0), (total_length + 0.1, corridor_width, 2.0))},  # far right
        ],
        "connections": [
            ("before", "bottleneck", "corridor"),
            ("bottleneck", "after", "corridor"),
        ]
    }


# ---------------------------------------------------------------------------
# Benchmark Scenarios
# ---------------------------------------------------------------------------

class BenchmarkScenarios:
    """Standardized benchmark scenarios for model validation."""

    @staticmethod
    def rimea_1_straight_corridor():
        """
        RiMEA Test 1: Straight corridor, single pedestrian.
        Expected: pedestrian walks at desired speed to the exit.
        """
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="RiMEA-1: Straight Corridor",
            description="Single pedestrian walking 20m corridor at 1.34 m/s",
            duration=60,
            time_step=0.1
        )
        engine.add_agent_profile(
            scenario=scenario,
            name="Pedestrian",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=1,
            base_speed=1.34,
            max_speed=1.34,
            size=0.25
        )
        engine.add_event(scenario, time=0, event_type="set_destination",
                        destination="end")
        return scenario

    @staticmethod
    def rimea_2_room_evacuation():
        """
        RiMEA Test 2: Room evacuation.
        Agents must find the exit and evacuate within a reasonable time.
        """
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="RiMEA-2: Room Evacuation",
            description="50 pedestrians evacuating a 10x10m room through a 1m door",
            duration=300,
            time_step=0.1
        )
        engine.add_agent_profile(
            scenario=scenario,
            name="Pedestrian",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=50,
            base_speed=1.34,
            size=0.25
        )
        engine.add_event(scenario, time=0, event_type="evacuate")
        return scenario

    @staticmethod
    def rimea_3_bottleneck():
        """
        RiMEA Test 3: Bottleneck.
        Measure flow through a bottleneck and compare to fundamental diagram.
        """
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="RiMEA-3: Bottleneck Flow",
            description="100 pedestrians through a 1m wide bottleneck",
            duration=300,
            time_step=0.1
        )
        engine.add_agent_profile(
            scenario=scenario,
            name="Pedestrian",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=100,
            base_speed=1.34,
            size=0.25
        )
        engine.add_event(scenario, time=0, event_type="set_destination",
                        destination="after")
        return scenario

    @staticmethod
    def rimea_4_counterflow():
        """
        RiMEA Test 4: Counterflow.
        Two groups moving in opposite directions should form lanes.
        """
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="RiMEA-4: Counterflow",
            description="Two groups of 20 pedestrians moving in opposite directions",
            duration=120,
            time_step=0.1
        )
        # Group A: left to right
        group_a = engine.add_agent_profile(
            scenario=scenario,
            name="Group A",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=20,
            base_speed=1.34,
            size=0.25
        )
        # Group B: right to left
        group_b = engine.add_agent_profile(
            scenario=scenario,
            name="Group B",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=20,
            base_speed=1.34,
            size=0.25
        )
        # Spawn positions would be handled by custom geometry
        engine.add_event(scenario, time=0, event_type="set_destination",
                        destination="end")
        return scenario

    @staticmethod
    def iso_20414_exit_speed_test():
        """
        ISO 20414: Test that evacuation speed decreases with density.
        """
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="ISO-20414: Density-Speed Test",
            description="Measure agent speed at different densities",
            duration=120,
            time_step=0.1
        )
        for density in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
            count = int(density * 20)  # 20 sqm area
            engine.add_agent_profile(
                scenario=scenario,
                name=f"Density-{density}",
                agent_type=AgentType.HUMAN,
                role=HumanRole.VISITOR,
                count=count,
                base_speed=1.34,
                size=0.25
            )
        engine.add_event(scenario, time=0, event_type="set_destination",
                        destination="end")
        return scenario

    @staticmethod
    def high_density_stability_test():
        """
        Test stability under high density (>5 persons/m²).
        Models should not produce overlapping or exploding velocities.
        """
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="High Density Stability",
            description="100 pedestrians in a 5x5m room (4 persons/m²)",
            duration=60,
            time_step=0.05
        )
        engine.add_agent_profile(
            scenario=scenario,
            name="Pedestrian",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=100,
            base_speed=1.34,
            size=0.25
        )
        engine.add_event(scenario, time=0, event_type="set_destination",
                        destination="outside")
        return scenario

    @staticmethod
    def queuing_behavior_test():
        """
        Test natural queue formation at a narrow door.
        """
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Queue Formation",
            description="20 pedestrians passing through a 0.8m door with flow limit",
            duration=180,
            time_step=0.1
        )
        engine.add_agent_profile(
            scenario=scenario,
            name="Pedestrian",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=20,
            base_speed=1.34,
            size=0.25,
            patience=0.8
        )
        engine.add_event(scenario, time=0, event_type="set_destination",
                        destination="outside")
        return scenario

    @staticmethod
    def list_all() -> List[str]:
        """List all available benchmark scenarios."""
        return [
            "RiMEA-1: Straight Corridor",
            "RiMEA-2: Room Evacuation",
            "RiMEA-3: Bottleneck Flow",
            "RiMEA-4: Counterflow",
            "ISO-20414: Density-Speed Test",
            "High Density Stability",
            "Queue Formation",
        ]

    @staticmethod
    def get(name: str):
        """Get a benchmark scenario by name."""
        mapping = {
            "RiMEA-1: Straight Corridor": BenchmarkScenarios.rimea_1_straight_corridor,
            "RiMEA-2: Room Evacuation": BenchmarkScenarios.rimea_2_room_evacuation,
            "RiMEA-3: Bottleneck Flow": BenchmarkScenarios.rimea_3_bottleneck,
            "RiMEA-4: Counterflow": BenchmarkScenarios.rimea_4_counterflow,
            "ISO-20414: Density-Speed Test": BenchmarkScenarios.iso_20414_exit_speed_test,
            "High Density Stability": BenchmarkScenarios.high_density_stability_test,
            "Queue Formation": BenchmarkScenarios.queuing_behavior_test,
        }
        factory = mapping.get(name)
        return factory() if factory else None
