# BIM-Agent Studio Simulation Engine Package
try:
    from .simulation_engine import (
        SimulationEngine, BIMSimulationModel, BIMAgent, AgentProfile,
        SimulationScenario, SimulationMetrics, AgentType, HumanRole, AgentState,
        ScenarioPresets, GroupBehavior,
        FED_RATE_COEFFICIENT, FED_SPEED_FACTOR,
        SMOKE_MAX_VISIBILITY, SMOKE_MIN_SPEED_RATIO,
    )
    from .pedestrian_models import (
        get_model, list_models, AgentMovementParams,
        SocialForceModel, CollisionFreeSpeedModel, AnticipationVelocityModel,
        GeneralizedCentrifugalForceModel, BasicModel, WarpDriverModel,  # NEW v1.3.0
        WallSegment, UniformGridSpatialIndex,                            # NEW v1.3.0
    )
    from .journey_system import (
        Journey, Stage, WaypointStage, WaitStage, FlowLimitStage,
        EvacuateStage, DirectSteeringStage, RepeatStage,
        BlockedPathStage,                                                # NEW v1.3.0
        StageResult, StageState,
    )
    from .benchmark_scenarios import BenchmarkScenarios

    __all__ = [
        # Engine core
        "SimulationEngine", "BIMSimulationModel", "BIMAgent", "AgentProfile",
        "SimulationScenario", "SimulationMetrics", "AgentType", "HumanRole",
        "AgentState", "ScenarioPresets",
        # v1.3.0 additions
        "GroupBehavior",
        "FED_RATE_COEFFICIENT", "FED_SPEED_FACTOR",
        "SMOKE_MAX_VISIBILITY", "SMOKE_MIN_SPEED_RATIO",
        # Pedestrian models
        "get_model", "list_models", "AgentMovementParams",
        "SocialForceModel", "CollisionFreeSpeedModel", "AnticipationVelocityModel",
        "GeneralizedCentrifugalForceModel", "BasicModel",
        "WarpDriverModel",          # NEW v1.3.0
        "WallSegment",
        "UniformGridSpatialIndex",  # NEW v1.3.0
        # Journey system
        "Journey", "Stage", "WaypointStage", "WaitStage", "FlowLimitStage",
        "EvacuateStage", "DirectSteeringStage", "RepeatStage",
        "BlockedPathStage",         # NEW v1.3.0
        "StageResult", "StageState",
        # Benchmarks
        "BenchmarkScenarios",
    ]
except ImportError:
    # Dependencies like mesa may not be available in all environments
    __all__ = []
