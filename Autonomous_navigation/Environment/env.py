import torch
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoLocator, MultipleLocator

class Environment:
    def __init__(self, device=None):
        # Automatically use GPU if available
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obstacles = [] # Stores SDF functions for LiDAR/Physics
        
        # --- NEW: Metadata for visualization ---
        # Stores [ (bottom_left_x, bottom_left_y, width, height), ... ]
        self._visual_rects = [] 
        self._current_max_abs_coord = 5.0 # Default starting zoom level
        # ----------------------------------------
        
        # Hyperparameters for LiDAR Raymarching
        self.max_ray_steps = 64
        self.hit_threshold = 0.01
        self.max_distance = 50.0  # Max LiDAR range

    def add_box_obstacle(self, center, size):
        """Standard 3D box."""
        c = torch.tensor(center, device=self.device, dtype=torch.float32)
        s = torch.tensor(size, device=self.device, dtype=torch.float32)
        
        def box_sdf(p):
            q = torch.abs(p - c) - s
            length_q = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
            fill_q = torch.clamp(torch.max(q, dim=-1)[0], max=0.0)
            return length_q + fill_q
            
        self.obstacles.append(box_sdf)
        
        # Update visualization metadata (convert center/half-extents to bottom-left/full-size)
        w, d = size[0]*2, size[1]*2
        bl_x = center[0] - size[0]
        bl_y = center[1] - size[1]
        self._visual_rects.append((bl_x, bl_y, w, d))
        
        # Update bounds tracker
        coords = [bl_x, bl_y, bl_x + w, bl_y + d]
        self._current_max_abs_coord = max(self._current_max_abs_coord, max(map(abs, coords)))

    def add_matrix_obstacle(self, matrix, cell_size, height, center=(0.0, 0.0)):
        """Builds 3D obstacles from a 2D grid/matrix optimized for speed."""
        matrix_tensor = torch.tensor(matrix, device=self.device, dtype=torch.float32)
        rows, cols = matrix_tensor.shape
        
        if isinstance(height, (int, float)):
            height_tensor = torch.full_like(matrix_tensor, fill_value=height)
        else:
            height_tensor = torch.tensor(height, device=self.device, dtype=torch.float32)
            
        solid_indices = torch.nonzero(matrix_tensor > 0)
        
        centers_list = []
        sizes_list = []
        
        for idx in solid_indices:
            r, c = idx[0].item(), idx[1].item()
            
            # Physical coordinates
            x = (c - cols / 2.0 + 0.5) * cell_size + center[0]
            y = (rows / 2.0 - r - 0.5) * cell_size + center[1]
            
            h = height_tensor[r, c].item()
            z = h / 2.0
            
            centers_list.append([x, y, z])
            half_size = cell_size / 2.0
            sizes_list.append([half_size, half_size, h / 2.0])
            
            # --- NEW: Visualization Metadata ---
            # Matplotlib rectangle needs bottom-left corner
            bl_x = x - half_size
            bl_y = y - half_size
            self._visual_rects.append((bl_x, bl_y, cell_size, cell_size))
            
            # Update bounds tracker based on corners of this cell
            coords = [bl_x, bl_y, bl_x + cell_size, bl_y + cell_size]
            self._current_max_abs_coord = max(self._current_max_abs_coord, max(map(abs, coords)))
            # -----------------------------------
            
        c_tensor = torch.tensor(centers_list, device=self.device).unsqueeze(0)
        s_tensor = torch.tensor(sizes_list, device=self.device).unsqueeze(0)
        
        # Batched SDF for performance
        def batched_grid_sdf(p):
            p_expanded = p.unsqueeze(1) 
            q = torch.abs(p_expanded - c_tensor) - s_tensor
            length_q = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
            fill_q = torch.clamp(torch.max(q, dim=-1)[0], max=0.0)
            distances = length_q + fill_q 
            min_dist, _ = torch.min(distances, dim=1)
            return min_dist
            
        self.obstacles.append(batched_grid_sdf)

    def _get_world_sdf(self, p):
        if not self.obstacles: return torch.full((p.shape[0],), 50.0, device=self.device)
        distances = [obs(p) for obs in self.obstacles]
        return torch.stack(distances, dim=0).min(dim=0)[0]

    def generate_lidar_cloud(self, origin, heading=0.0, num_points=40000):
        """Massively parallel LiDAR generation on GPU, rotated by car's heading (yaw)."""
        origin = torch.tensor(origin, device=self.device, dtype=torch.float32)
        idx = torch.arange(0, num_points, device=self.device)
        phi = torch.arccos(1 - 2 * (idx + 0.5) / num_points)
        theta = math.pi * (1 + 5**0.5) * idx
        
        # Base spherical directions (global frame)
        dx = torch.cos(theta) * torch.sin(phi)
        dy = torch.sin(theta) * torch.sin(phi)
        dz = torch.cos(phi)
        
        # Apply Yaw Rotation matrix around Z-axis based on car heading
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        
        rot_dx = cos_h * dx - sin_h * dy
        rot_dy = sin_h * dx + cos_h * dy
        rot_dz = dz
        
        dirs = torch.stack([rot_dx, rot_dy, rot_dz], dim=-1)
        t = torch.zeros(num_points, device=self.device)
        
        for _ in range(self.max_ray_steps):
            p = origin + dirs * t.unsqueeze(1) 
            d = self._get_world_sdf(p)
            t += d
            if torch.all((d < self.hit_threshold) | (t > self.max_distance)): break

        hit_mask = (d < self.hit_threshold) & (t < self.max_distance)
        return origin + dirs[hit_mask] * t[hit_mask].unsqueeze(1)

    def generate_2d_top_down_plot(self, goal_pos=None, figsize=(10, 10),loc = "center right"):
        """
        Generates a 2D Matplotlib figure of the environment top-down.
        Obstacles: Black
        Goal: Green circle (if provided)
        Bounds: Dynamic based on obstacles + 5m margin, forced square aspect.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 1. Set dynamic bounds based on required formula: max(abs(coords)) + 5
        limit = self._current_max_abs_coord + 5.0
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # CRITICAL for absolute scale compatibility: Force 1 unit X == 1 unit Y
        ax.set_aspect('equal', adjustable='box')
        legend = True
        # 2. Draw Obstacles (Black)
        for rect_data in self._visual_rects:
            x, y, w, h = rect_data
            # We use vector patches, not grid rasterization, for instant rendering
          
            rect = patches.Rectangle((x, y), w, h, linewidth=0, edgecolor='none', facecolor='black', zorder=10,label="Obstacles") if legend else patches.Rectangle((x, y), w, h, linewidth=0, edgecolor='none', facecolor='black', zorder=10)
            legend = False
            ax.add_patch(rect)
            
        # 3. Draw Goal (Green)
        if goal_pos is not None:
            # goal_pos assumes (x, y)
            goal_circle = plt.Circle((goal_pos[0], goal_pos[1]), 0.5, color='green', fill=True, zorder=20,label="Goal")
            ax.add_patch(goal_circle)
            # Add a subtle glow for visibility
            goal_glow = plt.Circle((goal_pos[0], goal_pos[1]), 0.8, color='green', alpha=0.3, fill=True, zorder=19)
            ax.add_patch(goal_glow)

        # 4. Styling (Grid and Tick Marks)
        ax.set_xlabel('X Meters', fontweight='bold')
        ax.set_ylabel('Y Meters', fontweight='bold')
        ax.set_title('Environment Top-Down View', fontweight='bold')
        
        # Set dark grid lines behind obstacles
        ax.grid(True, which='both', color='#999999', linestyle='-', alpha=0.5, zorder=0)
        
        # Configure tick marks to be relative to map size automatically
        # For small maps, marks every 1m or 5m. For huge maps, marks every 100m.
        ax.xaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_locator(AutoLocator())
        
        # Add minor grid lines for better scale perception
        ax.minorticks_on()
        # Ensure minor grid doesn't cluster too densely on large maps
        if limit > 20:
             # On large maps, don't show minor ticks, they cause clutter
             ax.tick_params(axis='both', which='minor', bottom=False, left=False)
        else:
             ax.grid(True, which='minor', color='#CCCCCC', linestyle=':', alpha=0.3, zorder=0)

        fig.tight_layout()
        fig.legend(loc=loc)
        # Returns the figure object so user can .show() or .savefig()
        return fig
