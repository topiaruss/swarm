"""
3D visualization using Matplotlib.

Real-time plotting of arena, entities, and detection ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
from .arena import Arena
from .entities import Entity, Drone, Team, Transponder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import MeshNetwork


class ArenaVisualizer:
    """
    Real-time 3D visualization of the arena.

    Displays drones, trajectories, and detection ranges.
    """

    def __init__(
        self,
        arena: Arena,
        show_detection_range: bool = True,
        show_trails: bool = False,
        trail_length: int = 50,
        show_uncertainty: bool = False,
        mesh_network: Optional['MeshNetwork'] = None
    ):
        """
        Initialize visualizer.

        Args:
            arena: Arena to visualize
            show_detection_range: Whether to show detection range circles
            show_trails: Whether to show trajectory trails
            trail_length: Number of points in trail
            show_uncertainty: Whether to show position uncertainty ellipses
            mesh_network: Optional mesh network to visualize
        """
        self.arena = arena
        self.show_detection_range = show_detection_range
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.show_uncertainty = show_uncertainty
        self.mesh_network = mesh_network

        # Trail history: dict of entity.id -> list of positions
        self.trails = {}

        # Uncertainty data: dict of entity.id -> uncertainty radius
        self.uncertainties = {}

        # Set up the plot
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Configure view
        self._setup_plot()

    def _setup_plot(self):
        """Configure plot appearance."""
        bounds = self.arena.bounds

        # Set limits
        self.ax.set_xlim(0, bounds[0])
        self.ax.set_ylim(0, bounds[1])
        self.ax.set_zlim(0, bounds[2])

        # Labels
        self.ax.set_xlabel('X (East) [m]')
        self.ax.set_ylabel('Y (North) [m]')
        self.ax.set_zlabel('Z (Up) [m]')

        # Title
        self.ax.set_title('Drone Swarm Arena')

        # Grid
        self.ax.grid(True, alpha=0.3)

        # Set viewing angle
        self.ax.view_init(elev=20, azim=45)

    def _get_entity_color(self, entity: Entity) -> str:
        """Get color for entity based on team."""
        if entity.team == Team.FRIENDLY:
            return 'blue'
        elif entity.team == Team.ENEMY:
            return 'red'
        else:
            return 'gray'

    def _get_entity_marker(self, entity: Entity) -> str:
        """Get marker shape for entity."""
        if isinstance(entity, Drone):
            return 'o'  # Circle for drones
        elif isinstance(entity, Transponder):
            return '^'  # Triangle for transponders
        else:
            return 's'  # Square for other entities

    def update(self, frame: Optional[int] = None):
        """
        Update visualization.

        Args:
            frame: Animation frame number (for FuncAnimation)
        """
        self.ax.clear()
        self._setup_plot()

        active_entities = self.arena.get_active_entities()

        # Draw each entity
        for entity in active_entities:
            pos = entity.position
            color = self._get_entity_color(entity)
            marker = self._get_entity_marker(entity)

            # Draw entity
            self.ax.scatter(
                pos[0], pos[1], pos[2],
                c=color,
                marker=marker,
                s=100,
                alpha=0.8,
                edgecolors='black',
                linewidths=1
            )

            # Draw velocity vector
            if np.linalg.norm(entity.velocity) > 0.1:
                vel_scale = 10  # Scale for visualization
                self.ax.quiver(
                    pos[0], pos[1], pos[2],
                    entity.velocity[0], entity.velocity[1], entity.velocity[2],
                    length=vel_scale,
                    color=color,
                    alpha=0.5,
                    arrow_length_ratio=0.3
                )

            # Draw detection range (for drones)
            if self.show_detection_range and isinstance(entity, Drone):
                self._draw_detection_range(entity)

            # Draw position uncertainty (if available)
            if self.show_uncertainty and entity.id in self.uncertainties:
                self._draw_uncertainty(entity, self.uncertainties[entity.id])

            # Update and draw trail
            if self.show_trails:
                self._update_trail(entity)
                self._draw_trail(entity)

        # Draw mesh network connections
        if self.mesh_network:
            self._draw_mesh_connections()

        # Add legend
        self._add_legend(active_entities)

        # Add time display
        self.ax.text2D(
            0.02, 0.98,
            f'Time: {self.arena.time:.1f}s',
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    def _draw_detection_range(self, drone: Drone):
        """Draw detection range circle around a drone."""
        # Draw circle in XY plane at drone altitude
        theta = np.linspace(0, 2*np.pi, 50)
        radius = drone.detection_range

        x = drone.position[0] + radius * np.cos(theta)
        y = drone.position[1] + radius * np.sin(theta)
        z = np.full_like(x, drone.position[2])

        color = self._get_entity_color(drone)
        self.ax.plot(x, y, z, color=color, alpha=0.2, linestyle='--', linewidth=1)

    def _draw_uncertainty(self, entity: Entity, uncertainty: float):
        """Draw position uncertainty circle around an entity."""
        # Draw circle in XY plane at entity altitude
        theta = np.linspace(0, 2*np.pi, 30)

        x = entity.position[0] + uncertainty * np.cos(theta)
        y = entity.position[1] + uncertainty * np.sin(theta)
        z = np.full_like(x, entity.position[2])

        color = self._get_entity_color(entity)
        self.ax.plot(x, y, z, color=color, alpha=0.4, linestyle=':', linewidth=2)

    def update_uncertainty(self, entity_id: str, uncertainty: float):
        """Update uncertainty visualization for an entity."""
        self.uncertainties[entity_id] = uncertainty

    def _draw_mesh_connections(self):
        """Draw mesh network connections between nodes."""
        if not self.mesh_network:
            return

        edges = self.mesh_network.get_connectivity_graph()

        for node_id1, node_id2 in edges:
            if node_id1 in self.mesh_network.nodes and node_id2 in self.mesh_network.nodes:
                pos1 = self.mesh_network.nodes[node_id1].position
                pos2 = self.mesh_network.nodes[node_id2].position

                # Draw line between connected nodes
                self.ax.plot(
                    [pos1[0], pos2[0]],
                    [pos1[1], pos2[1]],
                    [pos1[2], pos2[2]],
                    color='lime',      # Bright green instead of cyan
                    alpha=0.6,         # More opaque
                    linewidth=2,       # Thicker
                    linestyle='-'      # Solid line instead of dotted
                )

    def _update_trail(self, entity: Entity):
        """Update position trail for an entity."""
        if entity.id not in self.trails:
            self.trails[entity.id] = []

        trail = self.trails[entity.id]
        trail.append(entity.position.copy())

        # Limit trail length
        if len(trail) > self.trail_length:
            trail.pop(0)

    def _draw_trail(self, entity: Entity):
        """Draw position trail for an entity."""
        if entity.id in self.trails and len(self.trails[entity.id]) > 1:
            trail = np.array(self.trails[entity.id])
            color = self._get_entity_color(entity)
            self.ax.plot(
                trail[:, 0], trail[:, 1], trail[:, 2],
                color=color,
                alpha=0.3,
                linewidth=1
            )

    def _add_legend(self, entities: List[Entity]):
        """Add legend to plot."""
        # Count entities by team
        friendly_count = sum(1 for e in entities if e.team == Team.FRIENDLY)
        enemy_count = sum(1 for e in entities if e.team == Team.ENEMY)

        legend_text = f'Friendly: {friendly_count}  Enemy: {enemy_count}'
        self.ax.text2D(
            0.98, 0.98,
            legend_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    def show(self):
        """Display the plot."""
        plt.show()

    def save_frame(self, filename: str):
        """Save current frame to file."""
        plt.savefig(filename, dpi=150, bbox_inches='tight')

    def animate(self, update_func, interval: int = 50, frames: int = 200):
        """
        Create animation.

        Args:
            update_func: Function to call each frame (should update arena state)
            interval: Milliseconds between frames
            frames: Number of frames to animate
        """
        def combined_update(frame):
            update_func()
            self.update(frame)

        anim = FuncAnimation(
            self.fig,
            combined_update,
            frames=frames,
            interval=interval,
            blit=False
        )
        return anim
