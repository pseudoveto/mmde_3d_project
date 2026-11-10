import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import numpy as np
import math
import torch
import os
import subprocess

class VizAgent:
    def __init__(self, env):
        """
        Dedicated visualization and animation engine for EnvAgent simulation environments.

        Args:
            env (EnvAgent): Instance of the EnvAgent class.
        """
        self.env = env
        # Custom red-to-green colormap: 0.0 = Green (Safe), 1.0 = Bright Red (Danger)
        self.cost_cmap = LinearSegmentedColormap.from_list('cost_ring_cmap', ['#00FF00', '#FF0000'])

    def render(self, current_pos, car_heading, cost_ring, goal_pos=None, ring_radius=1.5, ring_width=0.4, figsize=(10, 10)):
        """
        Plots a single static 2D top-down view with a cost ring centered around the UGV.
        """
        fig = self.env.generate_2d_top_down_plot(goal_pos=goal_pos, figsize=figsize)
        ax = fig.gca()

        x_ugv, y_ugv = current_pos[0], current_pos[1]

        ring_vals = cost_ring.cpu().numpy() if hasattr(cost_ring, 'cpu') else np.array(cost_ring)
        num_bins = len(ring_vals)
        bin_angle_step = (2 * math.pi) / num_bins

        wedge_patches = []
        facecolors = []
        for i, cost in enumerate(ring_vals):
            angle_start_rad = car_heading + (-math.pi + i * bin_angle_step)
            angle_end_rad = angle_start_rad + bin_angle_step

            wedge = patches.Wedge(
                center=(x_ugv, y_ugv),
                r=ring_radius + ring_width,
                width=ring_width,
                theta1=math.degrees(angle_start_rad),
                theta2=math.degrees(angle_end_rad)
            )
            wedge_patches.append(wedge)
            facecolors.append(self.cost_cmap(float(cost)))

        collection = PatchCollection(wedge_patches, facecolors=facecolors, edgecolors='none', alpha=0.85, zorder=25)
        ax.add_collection(collection)

        # Draw UGV Body & Direction Heading Arrow
        ugv_marker = plt.Circle((x_ugv, y_ugv), radius=0.4, color='blue', zorder=30, label='UGV Agent')
        ax.add_patch(ugv_marker)

        arrow_dx = 1.2 * math.cos(car_heading)
        arrow_dy = 1.2 * math.sin(car_heading)
        ax.arrow(x_ugv, y_ugv, arrow_dx, arrow_dy, head_width=0.3, head_length=0.3, fc='cyan', ec='black', zorder=31)

        # Cost ring gradient colorbar in lower-left corner
        sm = plt.cm.ScalarMappable(cmap=self.cost_cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        cax = fig.add_axes([0.15, 0.15, 0.20, 0.025])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('Cost Ring Intensity (0=Safe, 1=Danger)', fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        ax.legend(loc='lower right')
        return fig

    def animate_trajectory(self, model, start_pos, goal_pos, initial_heading=0.0,
                           max_steps=300, ring_radius=1.5, ring_width=0.4,
                           save_path="ugv_trajectory.gif", fps=15, figsize=(10, 10)):
        """
        Runs model inference on GPU and generates an optimized trajectory animation with canvas blitting.
        Updated to support BrainAgent (TD3 non-recurrent MLP architecture).
        """
        import os
        import subprocess
        import torch
        from matplotlib.collections import PatchCollection
        import matplotlib.animation as animation

        device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure actor is in eval mode on target device
        if hasattr(model, "actor"):
            model.actor.to(device).eval()
        elif hasattr(model, "to"):
            model.to(device).eval()

        # Initialize tracking state variables
        curr_x, curr_y = float(start_pos[0]), float(start_pos[1])
        curr_heading = float(initial_heading)

        history_x = [curr_x]
        history_y = [curr_y]
        history_heading = [curr_heading]
        history_cost_ring = []

        def get_cost_ring_from_env(pos, heading):
            if hasattr(self.env, 'get_cost_ring'):
                return self.env.get_cost_ring(pos, heading)
            elif hasattr(self.env, 'cost_ring'):
                return self.env.cost_ring
            # Fallback: compute 360-degree cost ring using 2D raymarching
            hits = self.env.generate_lidar_cloud([pos[0], pos[1], 0.5], heading=heading, num_points=1800)
            if len(hits) == 0:
                return np.zeros(360, dtype=np.float32)
            dists = torch.norm(hits[:, :2] - torch.tensor(pos, device=self.env.device), dim=1)
            norm_cost = torch.clamp(1.0 - (dists / self.env.max_distance), 0.0, 1.0)
            return norm_cost.cpu().numpy() if len(norm_cost) == 360 else np.zeros(360, dtype=np.float32)

        history_cost_ring.append(get_cost_ring_from_env([curr_x, curr_y], curr_heading))

        # 1. Rollout Trajectory via GPU Inference
        for step in range(max_steps):
            # Compute relative goal delta vector
            rel_goal_x = goal_pos[0] - curr_x
            rel_goal_y = goal_pos[1] - curr_y

            # Format input tensors expected by BrainAgent.select_target_batch(cost_ring, current_pos, goal_pos)
            c_ring_t = torch.tensor(history_cost_ring[-1], dtype=torch.float32, device=device).unsqueeze(0)  # Shape: [1, 360]
            c_pos_t = torch.tensor([[curr_x, curr_y, curr_heading]], dtype=torch.float32, device=device)       # Shape: [1, 3]
            g_pos_t = torch.tensor([[rel_goal_x, rel_goal_y, 0.0]], dtype=torch.float32, device=device)       # Shape: [1, 3]

            with torch.no_grad():
                action = model.select_target_batch(c_ring_t, c_pos_t, g_pos_t)

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().squeeze()

            # Advance kinematics step
            if hasattr(self.env, 'step'):
                next_state, reward, done, info = self.env.step(action)
                if isinstance(next_state, dict):
                    curr_x, curr_y = next_state.get('pos', [curr_x, curr_y])
                    curr_heading = next_state.get('heading', curr_heading)
                    c_ring = next_state.get('cost_ring', get_cost_ring_from_env([curr_x, curr_y], curr_heading))
                else:
                    v_cmd = action[0] if hasattr(action, '__len__') else action
                    w_cmd = action[1] if hasattr(action, '__len__') and len(action) > 1 else 0.0
                    dt = 0.1
                    curr_heading += w_cmd * dt
                    curr_x += v_cmd * math.cos(curr_heading) * dt
                    curr_y += v_cmd * math.sin(curr_heading) * dt
                    c_ring = get_cost_ring_from_env([curr_x, curr_y], curr_heading)
                    done = np.hypot(curr_x - goal_pos[0], curr_y - goal_pos[1]) < 0.5
            else:
                # Default kinematic update rule
                v_cmd = action[0] if hasattr(action, '__len__') else 0.5
                w_cmd = action[1] if hasattr(action, '__len__') and len(action) > 1 else 0.0
                dt = 0.1
                curr_heading += w_cmd * dt
                curr_x += v_cmd * math.cos(curr_heading) * dt
                curr_y += v_cmd * math.sin(curr_heading) * dt

                c_ring = get_cost_ring_from_env([curr_x, curr_y], curr_heading)
                done = np.hypot(curr_x - goal_pos[0], curr_y - goal_pos[1]) < 0.5

            history_x.append(curr_x)
            history_y.append(curr_y)
            history_heading.append(curr_heading)
            history_cost_ring.append(c_ring)

            if done:
                print(f"Goal reached or episode finished at step {step + 1}/{max_steps}.")
                break

        total_frames = len(history_x)

        # 2. Render Base Figure with EnvAgent Top-Down View
        fig = self.env.generate_2d_top_down_plot(goal_pos=goal_pos, figsize=figsize)
        ax = fig.gca()

        ax.scatter([start_pos[0]], [start_pos[1]], color='lime', marker='*', s=200, zorder=35, label='Start Position')
        ax.scatter([goal_pos[0]], [goal_pos[1]], color='red', marker='X', s=200, zorder=35, label='Goal Position')

        path_line, = ax.plot([], [], color='yellow', linestyle='--', linewidth=2.5, zorder=20, label='UGV Path')
        ugv_marker = patches.Circle((history_x[0], history_y[0]), radius=0.4, color='blue', zorder=30, label='UGV Body')
        ax.add_patch(ugv_marker)

        arrow_dx = 1.2 * math.cos(history_heading[0])
        arrow_dy = 1.2 * math.sin(history_heading[0])
        heading_arrow = ax.arrow(history_x[0], history_y[0], arrow_dx, arrow_dy,
                                head_width=0.3, head_length=0.3, fc='cyan', ec='black', zorder=31)

        ax.legend(loc='lower right', framealpha=0.9, facecolor='white', fontsize=9)

        # Colorbar Gradient Legend
        sm = plt.cm.ScalarMappable(cmap=self.cost_cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        cax = fig.add_axes([0.15, 0.15, 0.22, 0.025])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('Cost Ring Intensity (0=Safe, 1=Danger)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        num_bins = len(history_cost_ring[0]) if len(history_cost_ring) > 0 and len(history_cost_ring[0]) > 0 else 360
        bin_angle_step = (2 * math.pi) / num_bins

        initial_wedges = [
            patches.Wedge(center=(history_x[0], history_y[0]), r=ring_radius + ring_width, width=ring_width, theta1=0, theta2=0)
            for _ in range(num_bins)
        ]
        ring_collection = PatchCollection(initial_wedges, facecolors='none', edgecolors='none', alpha=0.85, zorder=25)
        ax.add_collection(ring_collection)

        # 3. High-Performance Blitted Canvas Update
        def update(frame):
            nonlocal heading_arrow
            x = history_x[frame]
            y = history_y[frame]
            heading = history_heading[frame]
            c_ring = history_cost_ring[frame]

            path_line.set_data(history_x[:frame+1], history_y[:frame+1])
            ugv_marker.set_center((x, y))

            heading_arrow.remove()
            dx = 1.2 * math.cos(heading)
            dy = 1.2 * math.sin(heading)
            heading_arrow = ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='cyan', ec='black', zorder=31)

            ring_vals = c_ring.cpu().numpy() if hasattr(c_ring, 'cpu') else np.array(c_ring)
            wedge_patches = [
                patches.Wedge(
                    center=(x, y),
                    r=ring_radius + ring_width,
                    width=ring_width,
                    theta1=math.degrees(heading + (-math.pi + i * bin_angle_step)),
                    theta2=math.degrees(heading + (-math.pi + (i + 1) * bin_angle_step))
                )
                for i in range(len(ring_vals))
            ]

            ring_collection.set_paths(wedge_patches)
            ring_collection.set_facecolors([self.cost_cmap(float(c)) for c in ring_vals])

            return path_line, ugv_marker, heading_arrow, ring_collection

        # 4. Save GIF using Blitting and FFmpeg/NVENC Acceleration
        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=True)

        print(f"Rendering animation ({total_frames} frames)...")

        if save_path.endswith('.gif'):
            temp_mp4 = "temp_render.mp4"
            writer = animation.FFMpegWriter(fps=fps, extra_args=['-c:v', 'h264_nvenc', '-pix_fmt', 'yuv420p'])
            try:
                ani.save(temp_mp4, writer=writer)
                cmd = f"ffmpeg -y -i {temp_mp4} -vf \"fps={fps},scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" {save_path}"
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(temp_mp4):
                    os.remove(temp_mp4)
            except Exception:
                writer = animation.PillowWriter(fps=fps)
                ani.save(save_path, writer=writer)
        else:
            try:
                writer = animation.FFMpegWriter(fps=fps, extra_args=['-c:v', 'h264_nvenc', '-pix_fmt', 'yuv420p'])
                ani.save(save_path, writer=writer)
            except Exception:
                writer = animation.FFMpegWriter(fps=fps)
                ani.save(save_path, writer=writer)

        plt.close(fig)
        print(f"Animation saved successfully to {save_path}!")
        return fig
