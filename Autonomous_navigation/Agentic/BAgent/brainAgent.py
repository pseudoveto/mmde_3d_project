import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=32):
        super().__init__()
        # Fixed random matrix for high-frequency mapping
        self.B = nn.Parameter(torch.randn(in_dim, embed_dim // 2) * 10.0, requires_grad=False)

    def forward(self, x):
        # x: (B, in_dim)
        x_proj = 2 * torch.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TD3Actor(nn.Module):
    def __init__(self, pool_kernel=6):
        super().__init__()
        self.pool_kernel = pool_kernel
        self.fourier = FourierEncoder(in_dim=3, embed_dim=32)

        # 360 ring features pooled (360 // 6 = 60) + 32 fourier current pos + 32 fourier goal pos = 124
        self.fc1 = nn.Linear(60 + 32 + 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 3) # 3D action space (v_x, v_y, omega)

    def forward(self, cost_ring, current_pos, goal_pos):
        # Safely enforce 3D shape: (Batch, Channels, Length)
        if cost_ring.dim() == 2:
            cost_ring = cost_ring.unsqueeze(1)
        elif cost_ring.dim() == 4:
            cost_ring = cost_ring.squeeze(1)

        ring_60 = F.max_pool1d(cost_ring, kernel_size=self.pool_kernel, stride=self.pool_kernel).squeeze(1)
        pos_emb = self.fourier(current_pos)
        goal_emb = self.fourier(goal_pos)

        x = torch.cat([ring_60, pos_emb, goal_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))

class TD3Critic(nn.Module):
    def __init__(self, pool_kernel=6):
        super().__init__()
        self.pool_kernel = pool_kernel
        self.fourier = FourierEncoder(in_dim=3, embed_dim=32)

        in_dim = 60 + 32 + 32 + 3 # 127 features (including 3D action)

        # Twin Critic 1
        self.q1_1 = nn.Linear(in_dim, 256)
        self.q1_2 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)

        # Twin Critic 2
        self.q2_1 = nn.Linear(in_dim, 256)
        self.q2_2 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

    def forward(self, cost_ring, current_pos, goal_pos, action):
        if cost_ring.dim() == 2:
            cost_ring = cost_ring.unsqueeze(1)
        elif cost_ring.dim() == 4:
            cost_ring = cost_ring.squeeze(1)

        ring_60 = F.max_pool1d(cost_ring, kernel_size=self.pool_kernel, stride=self.pool_kernel).squeeze(1)
        pos_emb = self.fourier(current_pos)
        goal_emb = self.fourier(goal_pos)

        feat = torch.cat([ring_60, pos_emb, goal_emb, action], dim=-1)

        # Q1
        q1 = F.relu(self.q1_1(feat))
        q1 = F.relu(self.q1_2(q1))
        q1 = self.q1_out(q1)

        # Q2
        q2 = F.relu(self.q2_1(feat))
        q2 = F.relu(self.q2_2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

    def q1_forward(self, cost_ring, current_pos, goal_pos, action):
        if cost_ring.dim() == 2:
            cost_ring = cost_ring.unsqueeze(1)
        elif cost_ring.dim() == 4:
            cost_ring = cost_ring.squeeze(1)

        ring_60 = F.max_pool1d(cost_ring, kernel_size=self.pool_kernel, stride=self.pool_kernel).squeeze(1)
        pos_emb = self.fourier(current_pos)
        goal_emb = self.fourier(goal_pos)
        feat = torch.cat([ring_60, pos_emb, goal_emb, action], dim=-1)

        q1 = F.relu(self.q1_1(feat))
        q1 = F.relu(self.q1_2(q1))
        return self.q1_out(q1)

class BrainAgent:
    def __init__(self, device="cuda", lr=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

        self.actor = TD3Actor().to(self.device)
        self.actor_target = TD3Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TD3Critic().to(self.device)
        self.critic_target = TD3Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def select_target_batch(self, cost_ring, current_pos, goal_pos):
        return self.actor(cost_ring, current_pos, goal_pos)

    def update(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample batched experiences directly on GPU
        b_c_ring, b_c_pos, b_g_pos, b_action, b_reward, b_nc_ring, b_nc_pos, b_ng_pos, b_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(b_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(b_nc_ring, b_nc_pos, b_ng_pos) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(b_nc_ring, b_nc_pos, b_ng_pos, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = b_reward + ((1 - b_done) * self.gamma * target_q)

        current_q1, current_q2 = self.critic(b_c_ring, b_c_pos, b_g_pos, b_action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed Actor Updates (reduces backward pass overhead)
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(b_c_ring, b_c_pos, b_g_pos, self.actor(b_c_ring, b_c_pos, b_g_pos)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
