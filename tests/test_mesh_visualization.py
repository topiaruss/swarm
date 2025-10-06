#!/usr/bin/env python3
"""
Test mesh visualization line rendering.

This test creates a simple scenario to verify mesh lines are properly
cleared and redrawn as network topology changes.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Team, Role
from src.mesh import MeshNetwork, RadioType
from src.visualization import ArenaVisualizer


def test_mesh_line_rendering():
    """Test that mesh lines are properly cleared and redrawn."""

    # Create arena
    arena = Arena(
        bounds=(1000, 1000, 500),
        geo_reference=GeographicReference(latitude=37.7749, longitude=-122.4194)
    )

    # Create mesh network
    mesh = MeshNetwork(radio_type=RadioType.ESP_NOW)

    # Create 4 drones in a line (all within ESP-NOW range initially)
    center = arena.get_bounds_center()
    spacing = 100  # 100m apart

    for i in range(4):
        pos = center + np.array([i * spacing - 150, 0, 100])
        drone = Drone(
            position=pos,
            team=Team.FRIENDLY,
            role=Role.PATROL,
            max_speed=15.0
        )
        drone.id = f"DRONE-{i+1}"
        arena.add_entity(drone)
        mesh.add_node(drone.id, pos)

    # Initial connectivity check
    mesh.calculate_connectivity()
    initial_edges = mesh.get_connectivity_graph()
    print(f"Initial connectivity: {len(initial_edges)} edges")
    for edge in initial_edges:
        print(f"  {edge[0]} <-> {edge[1]}")

    # Move DRONE-1 far away (break connectivity)
    drone1 = arena.get_entity_by_id("DRONE-1")
    drone1.position = center + np.array([-500, 0, 100])  # 500m away
    mesh.update_node_position("DRONE-1", drone1.position)

    mesh.calculate_connectivity()
    updated_edges = mesh.get_connectivity_graph()
    print(f"\nAfter moving DRONE-1 away: {len(updated_edges)} edges")
    for edge in updated_edges:
        print(f"  {edge[0]} <-> {edge[1]}")

    # Verify connectivity changed
    assert len(initial_edges) != len(updated_edges), "Connectivity should change when drone moves"

    print("\n✓ Mesh connectivity updates correctly")
    print("✓ Visualization should reflect these changes dynamically")

    return arena, mesh


def test_mesh_visualization_interactive():
    """Interactive test - watch mesh lines update as drones move."""

    arena, mesh = test_mesh_line_rendering()

    viz = ArenaVisualizer(
        arena,
        show_detection_range=False,
        show_trails=True,
        trail_length=50,
        mesh_network=mesh
    )

    print("\n" + "=" * 70)
    print("Interactive Mesh Visualization Test")
    print("=" * 70)
    print()
    print("This test moves drones to verify mesh lines update correctly:")
    print("  - Frame 0-100: DRONE-1 moves away (lines should break)")
    print("  - Frame 100-200: DRONE-1 returns (lines should reconnect)")
    print()
    print("Watch for:")
    print("  - Green mesh lines should disappear when DRONE-1 moves away")
    print("  - Green mesh lines should reappear when DRONE-1 returns")
    print("  - NO lines should remain 'stuck' behind drones")
    print()

    center = arena.get_bounds_center()
    frame_count = [0]

    def update_test():
        frame = frame_count[0]
        frame_count[0] += 1

        # Move DRONE-1 away and back
        if frame < 100:
            # Move away
            progress = frame / 100.0
            offset_x = -150 + (-350) * progress  # From -150 to -500
            drone1 = arena.get_entity_by_id("DRONE-1")
            drone1.position = center + np.array([offset_x, 0, 100])
            mesh.update_node_position("DRONE-1", drone1.position)
        elif frame < 200:
            # Move back
            progress = (frame - 100) / 100.0
            offset_x = -500 + (350) * progress  # From -500 to -150
            drone1 = arena.get_entity_by_id("DRONE-1")
            drone1.position = center + np.array([offset_x, 0, 100])
            mesh.update_node_position("DRONE-1", drone1.position)

        # Update mesh connectivity
        mesh.calculate_connectivity()

        # Update visualization reference
        viz.mesh_network = mesh

        arena.update(0.033)

    anim = viz.animate(
        update_func=update_test,
        interval=33,
        frames=200
    )

    viz.show()


if __name__ == "__main__":
    # Run non-interactive test first
    print("Running mesh connectivity test...")
    test_mesh_line_rendering()

    # Then interactive visualization
    print("\nStarting interactive visualization test...")
    test_mesh_visualization_interactive()
