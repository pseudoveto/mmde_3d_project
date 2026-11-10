import os
import time
import atexit
import signal
import argparse
import random
import math
import numpy as np
import torch
from tqdm import tqdm
from collections import deque

# Import your custom modules
from Agentic.EAgent.env import EnvAgent
from Agentic.IAgent.inputAgent import InAgent
from Agentic.BAgent.brainAgent import BrainAgent
from trainer_utils import VectorizedGPUReplayBuffer

# ==========================================
# 1. Global Configurations & Path setup
# ==========================================
WAYPOINT_DIR = "/content/drive/MyDrive/Capestone/agentic/waypoints"
os.makedirs(WAYPOINT_DIR, exist_ok=True)

CHAPTER_CONFIGS = {
    1: {
        "max_epochs": 100,
        "test_cadence": 10,
        "w1_progress": 1.0,
        "w2_stagnation": 1.0,
        "w3_smoothness": 1.0,
        "w6_jerk": 0.3,
        "dt": 0.1
    }
}

PASS_THRESHOLD = 90.0
INITIAL_SIGMA = 0.5
N_MINI_TEST_MAPS = 10
TEST_LIDAR_POINTS = 2000
SAVE_INTERVAL_SECONDS = 300  # Saves every 5 minutes in background

# ==========================================
# 2. Reproducibility & Utilities
# ==========================================
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Seed] Global RNG seeded with {seed}")

class CurriculumUtils:
    @staticmethod
    def generate_episode_params(is_exam=False):
        """Generates random start and goal positions."""
        speed = 2.0
        g_rad = 1.0
        start_pos = (0.0, 0.0, 0.0) # x, y, heading

        # Randomize goal between 10m and 40m away
        r = random.uniform(10, 40)
        theta = random.uniform(-math.pi, math.pi)
        goal_pos = (r * math.cos(theta), r * math.sin(theta))

        return speed, start_pos, goal_pos, g_rad

    @staticmethod
    def kinematic_step(curr_pos, cmd_xy, speed, dt):
        """Simple kinematic update."""
        next_x = curr_pos[0] + cmd_xy[0] * speed * dt
        next_y = curr_pos[1] + cmd_xy[1] * speed * dt
        next_heading = math.atan2(cmd_xy[1], cmd_xy[0])
        return (next_x, next_y, next_heading)

# ==========================================
# 3. Environment-based Test Harness
# ==========================================
def run_env_based_test(brain, env, in_agent, utils, config, n_maps):
    """
    Runs n_maps freshly randomized episodes in an obstacle-free EnvAgent.
    Scores path straightness: 100 * clip(straight_line_dist / actual_path_length).
    """
    brain.actor.eval()
    scores = []
    successes = 0

    device = brain.device

    for _ in range(n_maps):
        speed, start_pos, goal_pos, g_rad = utils.generate_episode_params(is_exam=True)
        sl_dist = math.hypot(start_pos[0] - goal_pos[0], start_pos[1] - goal_pos[1])
        max_steps = math.ceil(2.0 * sl_dist / (speed * config["dt"]))

        curr_pos = start_pos
        path_length = 0.0
        reached_goal = False

        with torch.no_grad():
            for _ in range(max_steps):
                heading = curr_pos[2]
                lidar_cloud = env.generate_lidar_cloud(
                    origin=(curr_pos[0], curr_pos[1], 0.0),
                    heading=heading,
                    num_points=TEST_LIDAR_POINTS
                )
                cost_ring = in_agent.compute_cost_ring(
                    lidar_cloud,
                    current_pos=torch.tensor([curr_pos[0], curr_pos[1]], device=device, dtype=torch.float32),
                    car_heading=heading
                )

                rel_goal = (goal_pos[0] - curr_pos[0], goal_pos[1] - curr_pos[1], 0.0)

                # Format for TD3 MLP
                c_pos_t = torch.tensor([curr_pos], dtype=torch.float32, device=device)
                g_pos_t = torch.tensor([rel_goal], dtype=torch.float32, device=device)
                c_ring_t = cost_ring.unsqueeze(0)

                cmd_action = brain.select_target_batch(c_ring_t, c_pos_t, g_pos_t)[0]
                cmd_xy = cmd_action.cpu().numpy()[:2]

                next_pos = utils.kinematic_step(curr_pos, cmd_xy, speed, config["dt"])
                step_dist = math.hypot(next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])
                path_length += step_dist
                curr_pos = next_pos

                dist_to_goal = math.hypot(curr_pos[0] - goal_pos[0], curr_pos[1] - goal_pos[1])
                if dist_to_goal < g_rad:
                    reached_goal = True
                    break

        if reached_goal and path_length > 0:
            successes += 1
            map_score = 100.0 * float(np.clip(sl_dist / path_length, 0.0, 1.0))
        else:
            map_score = 0.0
        scores.append(map_score)

    brain.actor.train()
    return float(np.mean(scores)), successes / n_maps

# ==========================================
# 4. Main Trainer Loop
# ==========================================
def train_chapter_1(force_restart=False, seed=42):
    set_global_seed(seed)
    config = CHAPTER_CONFIGS[1]

    # Initialize Agents
    env = EnvAgent()
    in_agent = InAgent()
    brain = BrainAgent(device="cuda" if torch.cuda.is_available() else "cpu")
    utils = CurriculumUtils()

    # Vectorized buffer for TD3 MLP
    replay_buffer = VectorizedGPUReplayBuffer(max_size=200000, device=brain.device)
    batch_size = 256

    state = {
        "phase": "A",
        "epoch": 0,
        "consecutive_fail_count": 0,
        "exploration_sigma": INITIAL_SIGMA,
        "best_score": 0.0,
    }

    ckpt_path = os.path.join(WAYPOINT_DIR, "latest_checkpoint.pt")

    if not force_restart and os.path.exists(ckpt_path):
        print("=> Resuming from latest checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=brain.device)
        brain.actor.load_state_dict(checkpoint['actor_state_dict'])
        brain.critic.load_state_dict(checkpoint['critic_state_dict'])
        state = checkpoint['state']
    else:
        print("=> Starting Chapter 1 from scratch.")

    # -----------------------------------------------------------
    # Google Colab Timeout Defenses (Checkpointing)
    # -----------------------------------------------------------
    def save_now(reason="periodic"):
        torch.save({
            'actor_state_dict': brain.actor.state_dict(),
            'critic_state_dict': brain.critic.state_dict(),
            'state': state
        }, ckpt_path)
        print(f"[Checkpoint:{reason}] phase={state['phase']} epoch={state['epoch']} sigma={state['exploration_sigma']:.3f}")

    atexit.register(lambda: save_now("atexit"))
    signal.signal(signal.SIGTERM, lambda signum, frame: (save_now("SIGTERM"), exit(0)))

    _last_save_time = time.time()
    def autosave_if_due():
        nonlocal _last_save_time
        if time.time() - _last_save_time >= SAVE_INTERVAL_SECONDS:
            save_now("timer")
            _last_save_time = time.time()

    # -----------------------------------------------------------
    # Phase A: Stationary Bootstrap (Epochs 0 - 9)
    # -----------------------------------------------------------
    if state["phase"] == "A":
        print("\n--- Entering Phase A: Stationary Alignment ---")
        while state["epoch"] < 10:
            state["epoch"] += 1
            epoch_reward = 0

            pbar = tqdm(range(200), desc=f"Phase A | Epoch {state['epoch']}/10")
            for _ in pbar:
                for step in range(20):
                    curr_pos = (0.0, 0.0, 0.0)
                    r = random.uniform(5, 50)
                    theta = random.uniform(0, 2 * math.pi)
                    goal_pos = (r * math.cos(theta), r * math.sin(theta), 0.0)

                    cost_ring = torch.zeros(360, device=brain.device)

                    c_pos_t = torch.tensor([curr_pos], dtype=torch.float32, device=brain.device)
                    g_pos_t = torch.tensor([goal_pos], dtype=torch.float32, device=brain.device)
                    c_ring_t = cost_ring.unsqueeze(0)

                    with torch.no_grad():
                        cmd_action = brain.select_target_batch(c_ring_t, c_pos_t, g_pos_t)[0]

                    cmd_xy = cmd_action[:2]
                    goal_tensor = torch.tensor(goal_pos[:2], device=cmd_xy.device, dtype=cmd_xy.dtype)

                    # Reward alignment
                    cos_sim = torch.dot(cmd_xy, goal_tensor) / (torch.linalg.norm(cmd_xy) * torch.linalg.norm(goal_tensor) + 1e-8)
                    r_align = (10.0 * cos_sim).item()
                    epoch_reward += r_align

                    # Convert back to batched tensors to push to buffer
                    done_t = torch.tensor([step == 19], dtype=torch.float32, device=brain.device)
                    rew_t = torch.tensor([r_align], dtype=torch.float32, device=brain.device)

                    replay_buffer.push_batch(
                        c_ring_t, c_pos_t, g_pos_t, cmd_action.unsqueeze(0), rew_t,
                        c_ring_t, c_pos_t, g_pos_t, done_t
                    )

                    if step % 4 == 0 and len(replay_buffer) > batch_size:
                        brain.update(replay_buffer, batch_size=batch_size)

                autosave_if_due()

            print(f"Phase A Epoch {state['epoch']} | Avg Alignment Reward: {epoch_reward/(200*20):.2f}")
            save_now("phaseA-epoch")

        state["phase"] = "B"
        print("\n=> Phase A complete. Transitioning to Phase B (Moving Car).\n")
        save_now("phase-transition")

    # -----------------------------------------------------------
    # Phase B: Movement Curriculum & Exams
    # -----------------------------------------------------------
    UPDATE_EVERY_STEPS = 4
    global_step_counter = 0
    device = brain.device
    noise_buffer = torch.empty(2, device=device, dtype=torch.float32)

    if state["phase"] == "B":
        while state["epoch"] < config["max_epochs"]:
            state["epoch"] += 1
            pbar = tqdm(range(100), desc=f"Phase B | Epoch {state['epoch']}")

            for episode in pbar:
                speed, curr_pos_raw, goal_pos_raw, g_rad = utils.generate_episode_params()
                sl_dist = math.hypot(curr_pos_raw[0]-goal_pos_raw[0], curr_pos_raw[1]-goal_pos_raw[1])
                max_steps = math.ceil(2.0 * sl_dist / (speed * config["dt"]))

                curr_pos_t = torch.tensor([[curr_pos_raw[0], curr_pos_raw[1], curr_pos_raw[2]]], device=device, dtype=torch.float32)
                goal_pos_t = torch.tensor([[goal_pos_raw[0], goal_pos_raw[1], 0.0]], device=device, dtype=torch.float32)

                cmd_history = deque(maxlen=10)
                ach_history = deque([curr_pos_raw], maxlen=10)
                cost_ring = torch.zeros((1, 360), device=device, dtype=torch.float32) # Obstacle free

                for step in range(max_steps):
                    global_step_counter += 1
                    rel_goal_t = goal_pos_t - curr_pos_t

                    with torch.no_grad():
                        cmd_action = brain.select_target_batch(cost_ring, curr_pos_t, rel_goal_t)[0]

                    noise_buffer.normal_(0, state["exploration_sigma"])
                    cmd_xy_gpu = torch.clamp(cmd_action[:2] + noise_buffer, -1.0, 1.0)

                    next_x = curr_pos_t[0, 0] + cmd_xy_gpu[0] * speed * config["dt"]
                    next_y = curr_pos_t[0, 1] + cmd_xy_gpu[1] * speed * config["dt"]
                    next_heading = torch.atan2(cmd_xy_gpu[1], cmd_xy_gpu[0])

                    next_pos_t = torch.stack([next_x, next_y, next_heading]).unsqueeze(0)
                    cmd_history.append(cmd_action.cpu().numpy())
                    ach_history.append((next_x.item(), next_y.item(), next_heading.item()))

                    dist_to_goal = math.hypot((next_x - goal_pos_t[0, 0]).item(), (next_y - goal_pos_t[0, 1]).item())
                    reached_goal = dist_to_goal < g_rad

                    if len(cmd_history) >= 3:
                        # Reward calculations based on waypoints
                        p_ach_prev, p_ach_curr = ach_history[-3], ach_history[-2]
                        p_cmd_prev, p_cmd_curr, p_cmd_next = cmd_history[-3], cmd_history[-2], cmd_history[-1]

                        dist_prev = math.hypot(p_ach_prev[0]-goal_pos_raw[0], p_ach_prev[1]-goal_pos_raw[1])
                        dist_curr = math.hypot(p_ach_curr[0]-goal_pos_raw[0], p_ach_curr[1]-goal_pos_raw[1])

                        r_prog = config["w1_progress"] * (dist_prev - dist_curr)
                        moved = math.hypot(p_ach_curr[0]-p_ach_prev[0], p_ach_curr[1]-p_ach_prev[1])
                        r_stag = config["w2_stagnation"] * (-0.5 if moved < (0.1 * speed * config["dt"]) else 1.0)

                        v1, v2 = p_cmd_curr[:2] - p_cmd_prev[:2], p_cmd_next[:2] - p_cmd_curr[:2]
                        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        r_jerk = config["w6_jerk"] * -abs(norm_v1 - norm_v2)

                        total_reward = torch.tensor([r_prog + r_stag + r_jerk], dtype=torch.float32, device=device)
                        done_t = torch.tensor([reached_goal], dtype=torch.float32, device=device)

                        # Push to GPU Buffer
                        replay_buffer.push_batch(
                            cost_ring, curr_pos_t, rel_goal_t, torch.tensor([cmd_history[-3]], device=device),
                            total_reward, cost_ring, next_pos_t, rel_goal_t, done_t
                        )

                        if global_step_counter % UPDATE_EVERY_STEPS == 0 and len(replay_buffer) > batch_size:
                            brain.update(replay_buffer, batch_size=batch_size)

                    curr_pos_t = next_pos_t
                    if reached_goal:
                        break

                autosave_if_due()

            save_now("phaseB-epoch")

            # -----------------------------------------------------------
            # Mini-Test Exam (Every 10 Epochs)
            # -----------------------------------------------------------
            if state["epoch"] % config["test_cadence"] == 0:
                print(f"\n--- Running Epoch {state['epoch']} EnvAgent Exam ---")
                score, sr = run_env_based_test(
                    brain, env, in_agent, utils, config, n_maps=N_MINI_TEST_MAPS
                )
                print(f"[Mini-Test] Score: {score:.1f}/100 | Success Rate: {sr*100:.1f}%")

                if score >= PASS_THRESHOLD:
                    state["consecutive_fail_count"] = 0
                    print(f"Passed threshold! Agent is successfully learning waypoints.")
                    if score > state["best_score"]:
                        state["best_score"] = score
                        torch.save(brain.actor.state_dict(), os.path.join(WAYPOINT_DIR, "best_chapter_1.pt"))
                else:
                    state["consecutive_fail_count"] += 1
                    print(f"Failed threshold. Consecutive fails: {state['consecutive_fail_count']}")

                    if state["consecutive_fail_count"] >= 3:
                        state["exploration_sigma"] = 0.5 * INITIAL_SIGMA + 0.5 * state["exploration_sigma"]
                        state["consecutive_fail_count"] = 0
                        print(f"[Adjusting Hyperparams] Bumped exploration sigma back up to {state['exploration_sigma']:.3f} to unstick model.")

                save_now("post-test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_restart", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_chapter_1(force_restart=args.force_restart, seed=args.seed)
