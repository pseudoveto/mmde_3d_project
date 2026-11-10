import torch

class VectorizedGPUReplayBuffer:
    def __init__(self, max_size=200000, device="cuda"):
        self.max_size = max_size
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        # Pre-allocated persistent CUDA tensors
        self.cost_ring = torch.zeros((max_size, 360), dtype=torch.float32, device=self.device)
        self.curr_pos = torch.zeros((max_size, 3), dtype=torch.float32, device=self.device)
        self.goal_pos = torch.zeros((max_size, 3), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((max_size, 3), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)
        self.next_cost_ring = torch.zeros((max_size, 360), dtype=torch.float32, device=self.device)
        self.next_curr_pos = torch.zeros((max_size, 3), dtype=torch.float32, device=self.device)
        self.next_goal_pos = torch.zeros((max_size, 3), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)

    def push_batch(self, cost_ring, curr_pos, goal_pos, action, reward, next_cost_ring, next_curr_pos, next_goal_pos, done):
        batch_size = cost_ring.shape[0]

        # Handle wrap-around indices for buffer overflow
        indices = (torch.arange(self.ptr, self.ptr + batch_size, device=self.device) % self.max_size).long()

        self.cost_ring[indices] = cost_ring
        self.curr_pos[indices] = curr_pos
        self.goal_pos[indices] = goal_pos
        self.action[indices] = action
        self.reward[indices] = reward.unsqueeze(-1) if reward.dim() == 1 else reward
        self.next_cost_ring[indices] = next_cost_ring
        self.next_curr_pos[indices] = next_curr_pos
        self.next_goal_pos[indices] = next_goal_pos
        self.done[indices] = done.unsqueeze(-1) if done.dim() == 1 else done

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size=256):
        indices = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return (
            self.cost_ring[indices],
            self.curr_pos[indices],
            self.goal_pos[indices],
            self.action[indices],
            self.reward[indices],
            self.next_cost_ring[indices],
            self.next_curr_pos[indices],
            self.next_goal_pos[indices],
            self.done[indices]
        )

    def __len__(self):
        return self.size
