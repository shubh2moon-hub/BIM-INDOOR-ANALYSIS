#!/usr/bin/env python3
"""
BIM-Agent Studio - Basic Usage Example

This example demonstrates how to use the BIM-Agent Studio programmatically
without the GUI.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)

from core.bim_processor import BIMProcessor
from core.spatial_engine import SpatialIntelligenceEngine
from engine.simulation_engine import SimulationEngine, ScenarioPresets


def main():
    """Run a basic simulation example."""
    
    print("=" * 60)
    print("BIM-Agent Studio - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Load BIM Model
    print("\n[Step 1] Loading BIM Model...")
    processor = BIMProcessor()
    
    # Use a sample IFC file if provided as argument
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        ifc_path = sys.argv[1]
        print(f"Loading IFC: {ifc_path}")
        
        try:
            model = processor.load_ifc(ifc_path)
            print(f"  Model loaded: {model.name}")
            print(f"  Elements: {len(model.elements)}")
            print(f"  Spaces: {len(model.spaces)}")
            print(f"  Levels: {len(model.levels)}")
            
            # Print summary
            summary = processor.export_summary(model)
            print("\n" + summary[:500] + "...")
            
        except Exception as e:
            print(f"Error loading IFC: {e}")
            print("\nContinuing with demo mode (no model loaded)...")
            model = None
    else:
        print("No IFC file provided. Usage: python basic_usage.py <path_to.ifc>")
        print("\nContinuing with demo mode...")
        model = None
    
    # Step 2: Process Spatial Intelligence
    if model:
        print("\n[Step 2] Processing Spatial Intelligence...")
        spatial_engine = SpatialIntelligenceEngine()
        spatial_graph = spatial_engine.process_model(model)
        
        print(f"  Spaces (nodes): {len(spatial_graph.nodes)}")
        print(f"  Connections (edges): {len(spatial_graph.connections)}")
        
        # Analyze accessibility
        accessibility = spatial_engine.analyze_accessibility()
        print(f"\n  Accessibility Analysis:")
        print(f"    Total Spaces: {accessibility.get('total_spaces', 0)}")
        print(f"    Total Connections: {accessibility.get('total_connections', 0)}")
        print(f"    Avg Connections/Space: {accessibility.get('avg_connections_per_space', 0):.2f}")
        
        # Find bottlenecks
        bottlenecks = spatial_engine.get_critical_bottlenecks(5)
        if bottlenecks:
            print(f"\n  Critical Bottlenecks:")
            for b in bottlenecks:
                print(f"    {b['name']} (centrality: {b['betweenness']:.3f})")
        
        # Step 3: Create and Run Simulation
        print("\n[Step 3] Creating Simulation...")
        sim_engine = SimulationEngine()
        
        # Use evacuation preset for demo
        scenario = ScenarioPresets.evacuation_scenario()
        print(f"  Scenario: {scenario.name}")
        print(f"  Description: {scenario.description}")
        print(f"  Duration: {scenario.duration} steps")
        print(f"  Agent Profiles: {len(scenario.agent_profiles)}")
        print(f"  Events: {len(scenario.events)}")
        
        # Initialize simulation
        sim_model = sim_engine.initialize_simulation(model, spatial_engine, scenario)
        print(f"\n  Simulation initialized with {len(sim_model.schedule.agents)} agents")
        
        # Run simulation for a few steps
        print("\n[Step 4] Running Simulation...")
        steps_to_run = 50
        
        for step in range(steps_to_run):
            sim_engine.step()
            
            if step % 10 == 0:
                metrics = sim_model.get_current_metrics()
                print(f"  Step {step}: {metrics.agent_count} agents, "
                      f"{metrics.agents_moving} moving, "
                      f"avg speed: {metrics.avg_speed:.2f} m/s")
        
        # Step 5: Get Results
        print("\n[Step 5] Simulation Results...")
        results = sim_engine.get_results()
        
        print(f"  Total Steps: {len(results['metrics_history'])}")
        print(f"  Social Interactions: {results['total_interactions']}")
        print(f"  Evacuated Agents: {results['evacuated_agents']}")
        print(f"  Congestion Events: {len(results['congestion_events'])}")
        
        # Space occupancy
        print(f"\n  Space Occupancy (top 5):")
        occupancy = sim_model.get_space_occupancy()
        sorted_occupancy = sorted(
            occupancy.items(),
            key=lambda x: x[1]['agent_count'],
            reverse=True
        )[:5]
        
        for space_id, data in sorted_occupancy:
            print(f"    {data['space_name']}: {data['agent_count']} agents "
                  f"(density: {data['density']:.3f})")
        
        # Agent metrics
        print(f"\n  Agent Movement (sample):")
        for agent in list(sim_model.schedule.agents)[:5]:
            metrics = agent.get_metrics()
            print(f"    Agent {metrics['id']}: "
                  f"distance={metrics['traveled_distance']:.1f}m, "
                  f"time={metrics['travel_time']:.1f}s, "
                  f"state={metrics['state']}")
    
    else:
        print("\n[Note] No BIM model loaded. Skipping simulation steps.")
        print("       Run with an IFC file to see full functionality.")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
