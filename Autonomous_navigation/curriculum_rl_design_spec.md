# Curriculum Reinforcement Learning Design Spec (v3)
## Multi-UGV Path Planning — shared curriculum, custom architecture per teammate

This version is algorithm-agnostic. The environment contract (observation
format, action format, reward function, chapter structure) is fixed and
shared across the team. Everything below it — network architecture, on/off
-policy handling, exploration mechanism, memory mechanism — is each
teammate's own design choice, and this doc calls out explicitly where the
guidance branches by that choice.

---

## 0. Purpose — the only reason this curriculum exists

Cut wall-clock training time relative to throwing the full task at an
agent from scratch. Three mechanisms, and every design choice below is
checked against them:

1. **Smaller credit-assignment problems per chapter**, so whatever memory
   mechanism a teammate's network uses (recurrence, attention, frame
   stacking, or none) doesn't have to disentangle five compounded failure
   modes at once.
2. **Warm-started weights, not restarts** — each chapter continues training
   the same network rather than reinitializing. This is the biggest time
   saver, and it only pays off if forgetting is actively guarded against
   (§5) — which looks different depending on whether the algorithm is
   on-policy, off-policy, or model-based.
3. **Compute goes where it's needed** — epoch budget and test cadence are
   per-chapter, not fixed, so easy chapters don't waste budget and hard
   ones aren't starved.

---

## 1. Shared environment contract (fixed for everyone)

This is the interface every teammate's custom network must consume and
produce, regardless of internal architecture:

- **Observation:** a 360-element directional cost vector (from the
  multimodal image encoder — shared perception pipeline, not each
  teammate's concern) + relative goal coordinates.
- **Action:** a single commanded waypoint — the "next point" the car
  should move toward. Not raw motor commands.
- **Downstream:** a separate controller agent executes physical motion
  toward that waypoint, subject to kinematics/speed limits.

Because of that separation, there are two point streams in every episode,
and this matters for anyone's reward computation regardless of network:

| Stream | Symbol | Source |
|---|---|---|
| Commanded | `p^cmd_t` | the network's output at time t |
| Achieved | `p^ach_t` | ground-truth car position at time t |

**Reward the outcome on achieved, reward the output behavior on
commanded** — see §6.

---

## 2. Environment setup

```python
import os

REPO_URL = f"https://{os.environ.get('GH_PAT', '')}@github.com/Ams2123/Multi-UGV-path-planning-and-task-assignment.git"

!git clone --filter=blob:none --no-checkout {REPO_URL} repo_temp
!cd repo_temp && git sparse-checkout init --cone && git sparse-checkout set Environment
!cd repo_temp && git checkout

!pip install -e ./repo_temp/Environment

import os
os.chdir('/content/repo_temp')

from Environment import Environment
print("Environment installed successfully")
```

---

## 3. Chapter table (reordered — see reasoning below the table)

| # | Chapter | test cadence | max epochs | reward weight notes |
|---|---|---|---|---|
| 1 | Learn direction | 10 | 60 | `w1` (progress) dominant; **light input noise on from here (§7)** |
| 2 | Static obstacle avoidance | 10 | 80 | `w4` (collision) raised |
| 3 | **Mistake recovery** (moved up) | 15 | 100 | recovery bonus active |
| 4 | **Step back / concave detours** (moved down) | 15 | 100 | `w1` down-weighted vs `w5` |
| 5 | Dynamic obstacle avoidance | 20 | 120 | `w4` weighted higher than ch.2 |
| 6 | Chaser evasion | 25–50 | 150 | `w5` weighted so "never caught" beats "caught but survived" |
| 7 | Shrinking safe zone | 20 | 120 | `w7` active; combine with ch.2/5 elements |
| 8 | **Perception stress exam** (final, not a new-skill chapter) | 20 | 100 | no new reward term; aggressive occlusion, combined skills |

**Why chapters 3 and 4 swapped:** both teach the same underlying thing —
that temporarily increasing distance-to-goal can be correct — but chapter
4 (recovery) gives the agent an explicit trigger ("I'm in a dead end")
while chapter 3 (concave detour) requires anticipating the trap with no
trigger at all. That's strictly harder. Building the "temporary loss is
OK" credit assignment on the easier, clearly-triggered case first, then
generalizing it to the anticipatory case, should transfer faster than the
reverse order.

**Why perception robustness isn't its own early-in-sequence chapter
anymore:** if chapters 1–7 train on clean input and only chapter 8
introduces noise, everything learned earlier has to partially re-adjust to
a shifted input distribution at the very end — the opposite of what a
curriculum is for. Instead:
- **From chapter 1 onward**, apply *light* random dropout/perturbation to
  a small subset of the 360-vector's entries during training. Cheap,
  doesn't need its own gate, and every chapter's skill is learned under
  realistic (slightly noisy) input from the start.
- **Chapter 8 stays as a final gate**, but its job changes from "learn to
  handle noise" to "prove the noise-tolerance already built in survives
  aggressive, structured occlusion (e.g. a contiguous 30–60° blind arc)
  stacked on top of every earlier skill." It's a stress exam, not a new
  lesson.

---

## 4. Outer-loop hyperparameter controller

After each mini-test, adjust `difficulty_level` by at most ±1:
- Pass in fewer than ~40% of `max_epochs` → +1 difficulty.
- 3+ consecutive mini-tests below 60 → −1 difficulty, raise exploration.
- Otherwise hold steady.

Test score never touches the gradient; it only edits
`curriculum_config.json` (difficulty, reward weights, exploration-reset
scale) for the next chapter or next attempt.

---

## 5. Continual-learning strategy — branches by algorithm family

This is the part that has to change per teammate. "Exploration reset" and
"don't forget the previous chapter" mean different things depending on
what you're building. Pick the row that matches your architecture:

**Off-policy, replay-buffer-based (DDPG, SAC, TD3, DQN-family):**
- Exploration reset = reset the noise/entropy parameter (OU sigma,
  entropy coefficient, ε) to its chapter-start value.
- Forgetting guard: cap a **legacy pool** per completed chapter (e.g. 20k
  transitions, reservoir-sampled), saved when the chapter is passed.
  Sample new batches as ~80% fresh current-chapter / ~20% uniformly from
  all legacy pools.

**On-policy (PPO, A2C, vanilla policy gradient):**
- No persistent buffer, so the forgetting risk is higher, not lower — the
  network only ever sees the current chapter's distribution once old
  rollouts are discarded.
- Forgetting guard: either (a) cache a small fixed rollout set from each
  completed chapter and run periodic auxiliary fine-tuning passes over it
  between chapters, or (b) add an L2 (or EWC-style) regularization term
  anchoring weights toward their value at the end of the previous chapter,
  loosened gradually as the new chapter's training progresses.

**Model-based (Dreamer-style, MuZero-style):**
- The world model naturally accumulates a dataset across chapters, so
  retention is less fragile by default — but keep chapter tags on stored
  transitions so you can still analyze per-chapter data quality and debug
  which chapter's dynamics the model is weakest on.

**Recurrent architectures (GRU, LSTM, transformer-with-memory):**
- Reset hidden state every **episode**, independent of the chapter-level
  exploration reset — different clocks, don't conflate them.

**Non-recurrent (pure MLP/CNN on single-frame input):**
- No hidden state to reset, but chapters 5–7 need some notion of history
  (dynamic obstacles, a chaser, a shrinking radius) — frame-stack the last
  k observations or explicitly feed a velocity/delta feature, or the
  architecture is structurally missing information those chapters require.

---

## 6. Reward function (commanded vs. achieved — applies to every architecture)

```
R_total = w1*R_progress + w2*R_stagnation + w3*R_smoothness
        + w4*R_collision + w5*R_goal + w6*R_jerk + w7*R_zone
```

| Term | Formula | Basis | Purpose |
|---|---|---|---|
| `R_progress` | `dist(p^ach_{t-1}, goal) - dist(p^ach_t, goal)` (signed) | achieved | real progress, not just movement |
| `R_stagnation` | `-0.5` if `dist(p^ach_t, p^ach_{t-1}) < threshold` else `+1` | achieved | discourage physically sitting still |
| `R_smoothness` | `-0.5` if `angle(p^cmd_{t-1}, p^cmd_t, p^cmd_{t+1})` exceeds threshold else `+1` | commanded | discourage skewed/zigzag GPS output |
| `R_collision` | `-100` on collision | achieved | safety, ground truth |
| `R_goal` | `+200` on reaching goal (terminal) | achieved | task completion |
| `R_jerk` | `-abs(dist(p^cmd_{t-1},p^cmd_t) - dist(p^cmd_t,p^cmd_{t+1}))` | commanded | penalize uneven commanded step sizes |
| `R_zone` | `-50` scaled by distance outside shrink radius, else `0` | achieved | ch.7 only |

Starting weights `w1..w5,w7 = 1.0`, `w6 = 0.3`, tuned per-chapter by §4's
controller. Per-chapter overrides noted in the §3 table.

---

## 7. Perception noise schedule (see §3 reasoning)

| Chapter | dropout style | severity |
|---|---|---|
| 1–7 | random scatter, small subset of 360 entries | light, constant |
| 8 (exam only) | structured contiguous blind arc | aggressive |

Implementation-wise this is a data-augmentation step on the shared
observation, so it's identical regardless of which teammate's network
consumes it.

---

## 8. Google Colab safety — checkpointing & resume

Applies identically regardless of architecture — only the *contents* of
what gets saved differs slightly (see note).

- **Mount Drive first**, write all checkpoints/logs there, never rely on
  `/content` surviving a disconnect.
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  CKPT_DIR = '/content/drive/MyDrive/ugv_curriculum_ckpts'
  ```
- **Checkpoint contents:** all network weights + optimizer state(s), any
  target networks if applicable, exploration/noise parameter, current
  chapter, `difficulty_level`, epoch count within chapter, mini-test score
  history, `curriculum_config.json` snapshot, RNG states (`torch`,
  `numpy`, `random`). Off-policy: also the replay buffer (or legacy
  pools). On-policy: also the cached rollout sets from §5, if using that
  strategy.
- **Two checkpoint types:** `latest_checkpoint.pt` (overwritten every
  save, pure crash-resume) and `best_chapter_{n}.pt` (written only on a
  new best exam score — never overwritten by a worse run; this is the
  deployable artifact).
- **Save on two triggers:** every epoch, and on a background timer (e.g.
  every 5 minutes), since a disconnect can happen mid-epoch.
- **Best-effort exit hook:**
  ```python
  import atexit, signal
  atexit.register(save_checkpoint)
  signal.signal(signal.SIGTERM, lambda *a: save_checkpoint())
  ```
- **Resume on startup:** always check for `latest_checkpoint.pt` before
  initializing fresh weights.
- **Buffer/rollout-cache persistence:** save less often than model weights
  (e.g. every 3rd checkpoint) — losing some history is recoverable, losing
  weights isn't. Store compressed, not raw pickled objects.

---

## 9. GPU resource optimization (Colab, applies broadly)

- **Mixed precision** (`torch.cuda.amp`) on forward/backward passes.
- **Don't store raw images anywhere in training data** — only the
  precomputed 360-cost-vector + goal coords (the shared observation from
  §1). This is the single biggest memory saver, regardless of algorithm.
- **Keep any buffer/rollout-cache on CPU RAM**, move only sampled batches
  to GPU.
- **Batch size tuned once per session tier** (e.g. ~256 on a T4), not
  re-tuned every chapter.
- **`torch.cuda.empty_cache()`** at chapter transitions when
  `difficulty_level` changes tensor shapes.
- **Parallelize environment rollouts** if `Environment` supports multiple
  simultaneous instances — for small networks like these, CPU-bound
  simulation is usually the real bottleneck, not the GPU forward pass.

---

## 10. Team architecture menu

No one is locked into a specific algorithm or memory mechanism — this is
just a menu to make sure the team's six builds are a genuinely useful
comparison rather than six copies of the same idea:

| Slot | Algorithm | Memory mechanism | What it tests |
|---|---|---|---|
| A | DDPG | GRU | deterministic policy + recurrence baseline |
| B | DDPG | Transformer (in place of GRU) | attention vs. recurrence, algorithm held fixed |
| C | SAC | GRU | stochastic/entropy-driven vs. deterministic, on ch.3/4's "step back" skill specifically |
| D | TD3 | GRU | twin-critic ablation on DDPG's overestimation bias |
| E | PPO | GRU or frame-stack | on-policy stability baseline; easiest to debug if the curriculum logic itself looks broken |
| F | Rainbow DQN (discretized waypoint grid) | frame-stack | whether discretizing the continuous action loses much |
| G (stretch) | Model-based (Dreamer-style) | learned world model | sample efficiency ceiling, if the 360-vector dynamics are learnable |

Pick 5–6 slots across the team; whichever's left over is a good fallback
if someone's build stalls.
