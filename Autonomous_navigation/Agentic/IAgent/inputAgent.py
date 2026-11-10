import torch
import math

class InAgent:
    def __init__(self, num_bins=360, radius=50.0, device=None):
        """
        Processes multi-modal LiDAR point clouds into a vehicle-centric 360-degree cost ring.

        Args:
            num_bins (int): Resolution of the cost ring (default 360 = 1 deg/bin).
            max_range (float): Maximum sensor range for distance normalization.
            device: PyTorch compute device (cuda/cpu).
        """
        self.num_bins = num_bins
        self.max_range = float(radius)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Pre-calculate bin conversion constants
        # Scale local angles from [-pi, pi) directly to bin index range [0, num_bins - 1]
        self.scale = self.num_bins / (2.0 * math.pi)

    @torch.no_grad()
    def compute_cost_ring(self, lidar_cloud, current_pos, car_heading=0.0):
        """
        Vectorized 360-degree cost ring generation executing entirely on GPU.
        Achieves sub-millisecond throughput via PyTorch scatter reductions.
        """
        cost_ring = torch.zeros(self.num_bins, device=self.device, dtype=torch.float32)

        if lidar_cloud is None or lidar_cloud.shape[0] == 0:
            return cost_ring

        # Ensure inputs are tensors on the target device
        if not isinstance(lidar_cloud, torch.Tensor):
            lidar_cloud = torch.as_tensor(lidar_cloud, device=self.device, dtype=torch.float32)

        if not isinstance(current_pos, torch.Tensor):
            curr_xy = torch.as_tensor(current_pos[:2], device=self.device, dtype=torch.float32)
        else:
            curr_xy = current_pos[:2].to(device=self.device, dtype=torch.float32)

        # 1. Relative displacement & Euclidean distances (2D plane)
        rel_points = lidar_cloud[:, :2] - curr_xy
        distances = torch.linalg.norm(rel_points, dim=-1)

        # Filter out points exceeding max_range
        valid_mask = distances <= self.max_range
        if not valid_mask.any():
            return cost_ring

        rel_points = rel_points[valid_mask]
        distances = distances[valid_mask]

        # 2. Local angle transformation [-pi, pi)
        global_angles = torch.atan2(rel_points[:, 1], rel_points[:, 0])

        # Heading subtraction and clean wrap to [-pi, pi)
        local_angles = torch.remainder(global_angles - car_heading + math.pi, 2.0 * math.pi) - math.pi

        # 3. Direct O(1) bin index assignment
        # Bin calculation assumes scale = num_bins / (2 * pi)
        bin_indices = ((local_angles + math.pi) * self.scale).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        # 4. Vectorized Scatter Reduction (Minimum distance per bin)
        min_distances = torch.full((self.num_bins,), self.max_range, device=self.device, dtype=torch.float32)
        min_distances.scatter_reduce_(0, bin_indices, distances, reduce='amin', include_self=True)

        # 5. Bounded cost normalization: closer obstacles -> higher cost (~1.0)
        cost_ring = torch.clamp(1.0 - (min_distances / self.max_range), 0.0, 1.0)

        return cost_ring
