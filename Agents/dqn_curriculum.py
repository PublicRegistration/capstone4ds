"""
==============================================================================
DQN CUSTOM MODEL FOR TRACKMANIA 2020 — LIDAR PROGRESS ENVIRONMENT
==============================================================================

Adapted from the TMRL competition tutorial script (custom_actor_module.py).
Replaces the SAC algorithm and continuous actor with a Deep Q-Network (DQN)
using a fixed discrete action space.

Usage (run each in a separate terminal):
    python dqn.py --server
    python dqn.py --trainer
    python dqn.py --worker

    For testing a trained model without connecting to a server:
    python dqn.py --test
      Auto-logs per-episode metrics to TmrlData/test_logs/ — see TEST-MODE
      LOGGING section below for details.

IMPORTANT:
    - Set a unique 'RUN_NAME' in config.json before running
    - Set "RTGYM_INTERFACE": "TM20LIDARPROGRESS" in config.json
    - Set "WINDOW_WIDTH": 958 and "WINDOW_HEIGHT": 488 in config.json
    - Use the front camera with the car hidden (press 3 until car disappears)
    - Track must have plain road with black borders for LIDAR to work
"""

# =====================================================================
# IMPORTS
# =====================================================================

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.util import partial
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.training_offline import TrainingOffline
from tmrl.actor import TorchActorModule
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import no_grad

import numpy as np
import gymnasium as gym
import os
import csv
import json
from copy import deepcopy
from datetime import datetime
from statistics import stdev, median
import vgamepad as vg
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# =====================================================================
# USEFUL PARAMETERS
# =====================================================================

epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

wandb_run_id = cfg.WANDB_RUN_ID
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]
os.environ['WANDB_API_KEY'] = wandb_key

max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
server_port = cfg.PORT
password = cfg.PASSWORD
security = cfg.SECURITY

# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================

memory_base_cls = cfg_obj.MEM
sample_preprocessor = None
dataset_path = cfg.DATASET_PATH
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

# =====================================================================
# REAL-TIME REWARD DISPLAY (training-time per-step prints)
# =====================================================================

DEBUG_REWARDS = True

# =====================================================================
# ROLLING CHECKPOINT SNAPSHOTS
# =====================================================================

CHECKPOINT_SNAPSHOT_EVERY = 5000
CHECKPOINT_SNAPSHOT_DIR = str(cfg.WEIGHTS_FOLDER / "rolling_snapshots")
CHECKPOINT_SNAPSHOT_KEEP = 400

# =====================================================================
# DIVERGENCE EARLY-WARNING THRESHOLD
# =====================================================================

DIVERGENCE_Q_MAX_THRESHOLD = 2.0

# =====================================================================
# WALL PROXIMITY PENALTY PARAMETERS
# =====================================================================

WALL_MAX_PENALTY = 0.25
WALL_K = 8
WALL_THRESHOLD = 300
WALL_BEAMS = 19
WALL_BEAM_SIDES = 5

# =====================================================================
# FORWARD MOTION REWARD PARAMETERS
# =====================================================================

SPEED_REWARD_COEFF = 0.08

# =====================================================================
# GLOBAL REWARD SCALING
# =====================================================================

REWARD_SCALE = 0.1

# =====================================================================
# OBSERVATION INDICES (TM20LIDARPROGRESS)
# =====================================================================

SPEED_OBS_INDEX = 0
PROGRESS_OBS_INDEX = 1
LIDAR_OBS_INDEX = 2

# =====================================================================
# TEST-MODE LOGGING
#
# When --test is invoked, every episode's summary is appended to a
# per-session CSV file under TmrlData/test_logs/. A running summary is
# also printed to stdout after each episode, and a final summary is
# printed when you CTRL+C out.
#
# CSV columns per row:
#   episode, timestamp_iso, steps, time_seconds, final_progress,
#   completed (0/1), end_reason, total_scaled_reward,
#   speed_reward_total, wall_penalty_total
#
# end_reason values: "completed" | "crashed" | "truncated"
#
# The CSV is flushed after every row so CTRL+C cannot lose data.
#
# "completion" threshold is final_progress > TEST_COMPLETION_THRESHOLD.
# =====================================================================

TEST_MODE_ACTIVE = False
TEST_COMPLETION_THRESHOLD = 0.95
_test_log_handle = None
_test_log_writer = None
_test_log_path = None
_test_stats = {
    "total_episodes": 0,
    "completed_episodes": 0,
    "crash_episodes": 0,
    "timeout_episodes": 0,
    "completed_times": [],
}


def _init_test_log():
    """Open a timestamped CSV for this test session and write the header."""
    global TEST_MODE_ACTIVE, _test_log_handle, _test_log_writer, _test_log_path

    run_name = cfg.TMRL_CONFIG.get("RUN_NAME", "run")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(cfg.WEIGHTS_FOLDER.parent / "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    _test_log_path = os.path.join(log_dir, f"test_{run_name}_{ts}.csv")

    _test_log_handle = open(_test_log_path, "w", newline="", encoding="utf-8")
    _test_log_writer = csv.writer(_test_log_handle)
    _test_log_writer.writerow([
        "episode", "timestamp_iso", "steps", "time_seconds",
        "final_progress", "completed", "end_reason",
        "total_scaled_reward", "speed_reward_total", "wall_penalty_total"
    ])
    _test_log_handle.flush()
    TEST_MODE_ACTIVE = True

    print("\n" + "=" * 70)
    print(f"Test log: {_test_log_path}")
    print("=" * 70 + "\n")


def _log_test_episode(steps, final_progress, terminated, truncated,
                      total_reward, speed_reward_total, wall_penalty_total):
    """Write one CSV row, update in-memory stats, then print a one-liner."""
    global _test_stats

    time_seconds = steps * 0.05
    completed = final_progress > TEST_COMPLETION_THRESHOLD
    if completed:
        end_reason = "completed"
    elif terminated:
        end_reason = "crashed"
    else:
        end_reason = "truncated"

    _test_stats["total_episodes"] += 1
    if completed:
        _test_stats["completed_episodes"] += 1
        _test_stats["completed_times"].append(time_seconds)
    elif terminated:
        _test_stats["crash_episodes"] += 1
    else:
        _test_stats["timeout_episodes"] += 1

    _test_log_writer.writerow([
        _test_stats["total_episodes"],
        datetime.now().isoformat(timespec="seconds"),
        steps,
        round(time_seconds, 3),
        round(final_progress, 4),
        int(completed),
        end_reason,
        round(total_reward, 4),
        round(speed_reward_total, 4),
        round(wall_penalty_total, 4),
    ])
    _test_log_handle.flush()

    _print_test_episode_line(completed, end_reason, time_seconds, final_progress)


def _print_test_episode_line(completed, end_reason, time_seconds, final_progress):
    """Per-episode one-line summary plus running stats."""
    s = _test_stats
    total = s["total_episodes"]
    times = s["completed_times"]

    if completed:
        result_str = f"[ep {total:4d}]  OK {time_seconds:6.3f}s"
    else:
        result_str = f"[ep {total:4d}]  {end_reason.upper()} @ {final_progress*100:5.1f}%"

    if times:
        best = min(times)
        mean_t = sum(times) / len(times)
        med_t = median(times)
        std_t = stdev(times) if len(times) > 1 else 0.0
        completion_rate = s["completed_episodes"] / total * 100
        stats_str = (
            f"PR={best:6.3f}s  "
            f"mean={mean_t:6.3f}s  "
            f"med={med_t:6.3f}s  "
            f"std={std_t:.3f}s  "
            f"completions={s['completed_episodes']}/{total} ({completion_rate:.1f}%)"
        )
    else:
        stats_str = (
            f"no completions yet  "
            f"({s['crash_episodes']} crash / {s['timeout_episodes']} timeout)"
        )

    print(f"\n{result_str}  |  {stats_str}")


def _print_final_test_summary():
    """Final end-of-session summary on CTRL+C."""
    s = _test_stats
    total = s["total_episodes"]
    if total == 0:
        print("\n\nNo episodes completed in this session.\n")
        return

    completed = s["completed_episodes"]
    crashed = s["crash_episodes"]
    timed_out = s["timeout_episodes"]
    completion_rate = completed / total * 100
    times = s["completed_times"]

    print("\n\n" + "=" * 70)
    print("FINAL TEST SESSION SUMMARY")
    print("=" * 70)
    print(f"  Total episodes:     {total}")
    print(f"  Completed:          {completed} ({completion_rate:.1f}%)")
    print(f"  Crashed:            {crashed}")
    print(f"  Timed out:          {timed_out}")

    if times:
        best = min(times)
        mean_t = sum(times) / len(times)
        med_t = median(times)
        std_t = stdev(times) if len(times) > 1 else 0.0
        print(f"\n  Completion time stats (seconds):")
        print(f"    PR (best):        {best:.3f}")
        print(f"    Mean:             {mean_t:.3f}")
        print(f"    Median:           {med_t:.3f}")
        print(f"    Std deviation:    {std_t:.3f}")
        print(f"    Range:            [{best:.3f}, {max(times):.3f}]")

    print(f"\n  Full CSV log: {_test_log_path}")
    print("=" * 70 + "\n")


# Episode state accumulator (used by both debug prints and test logging)
_episode_state = {
    "cumulative_reward": 0.0,
    "cumulative_wall_penalty": 0.0,
    "cumulative_speed_reward": 0.0,
    "step": 0,
}

_default_compressor = cfg_obj.SAMPLE_COMPRESSOR


def _wall_proximity_penalty(obs):
    lidar = obs[LIDAR_OBS_INDEX]
    last_frame = lidar[-WALL_BEAMS:]
    left_beams = float(last_frame[:WALL_BEAM_SIDES].min())
    right_beams = float(last_frame[-WALL_BEAM_SIDES:].min())
    proximity = min(left_beams, right_beams)
    if proximity >= WALL_THRESHOLD:
        return 0.0
    t = 1.0 - (proximity / WALL_THRESHOLD)
    return WALL_MAX_PENALTY * (t ** WALL_K)


def _speed_reward(obs):
    speed = float(obs[SPEED_OBS_INDEX].flat[0])
    speed = max(0.0, min(speed, SPEED_MAX))
    return SPEED_REWARD_COEFF * (speed / SPEED_MAX)


def sample_compressor(act, obs, rew, terminated, truncated, info):
    """
    Custom sample compressor with reward scaling, wall penalty, speed
    reward, real-time display (training), and CSV logging (test).
    """
    wall_penalty = _wall_proximity_penalty(obs)
    speed_rew = _speed_reward(obs)
    rew_modified = (rew + speed_rew - wall_penalty) * REWARD_SCALE
    progress_val = float(obs[PROGRESS_OBS_INDEX].flat[0])

    # Always accumulate per-episode state (cheap). Used by both DEBUG_REWARDS
    # prints and by the --test CSV logger.
    _episode_state["cumulative_reward"] += float(rew_modified)
    _episode_state["cumulative_wall_penalty"] += wall_penalty * REWARD_SCALE
    _episode_state["cumulative_speed_reward"] += speed_rew * REWARD_SCALE
    _episode_state["step"] += 1

    # Training-time verbose per-step print
    if DEBUG_REWARDS:
        speed_val = float(obs[SPEED_OBS_INDEX].flat[0])
        print(
            f"\rStep {_episode_state['step']:4d} | "
            f"spd: {speed_val:5.1f} | "
            f"prog: {progress_val:5.1%} | "
            f"raw: {float(rew):+6.3f} | "
            f"spd_r: +{speed_rew:.3f} | "
            f"wall: -{wall_penalty:.3f} | "
            f"scaled: {rew_modified:+6.4f} | "
            f"total: {_episode_state['cumulative_reward']:+7.3f}",
            end="",
            flush=True,
        )

    # End-of-episode handling
    if terminated or truncated:
        if DEBUG_REWARDS:
            print(
                f"\n--- Episode ended ({'terminated' if terminated else 'truncated'}) | "
                f"steps: {_episode_state['step']} | "
                f"final progress: {progress_val:5.1%} | "
                f"total scaled reward: {_episode_state['cumulative_reward']:+.3f} | "
                f"speed reward total: +{_episode_state['cumulative_speed_reward']:.3f} | "
                f"wall penalty total: -{_episode_state['cumulative_wall_penalty']:.3f} ---"
            )

        if TEST_MODE_ACTIVE:
            _log_test_episode(
                steps=_episode_state["step"],
                final_progress=progress_val,
                terminated=terminated,
                truncated=truncated,
                total_reward=_episode_state["cumulative_reward"],
                speed_reward_total=_episode_state["cumulative_speed_reward"],
                wall_penalty_total=_episode_state["cumulative_wall_penalty"],
            )

        # reset for next episode
        _episode_state["cumulative_reward"] = 0.0
        _episode_state["cumulative_wall_penalty"] = 0.0
        _episode_state["cumulative_speed_reward"] = 0.0
        _episode_state["step"] = 0

    return _default_compressor(act, obs, rew_modified, terminated, truncated, info)

# =====================================================================
# TEST-MODE ENV WRAPPER
#
# tmrl's RolloutWorker.run_episode(..., train=False) path sets
# collect_samples=False, which means sample_compressor is never
# invoked during test episodes. To log anything in test mode we must
# hook somewhere that IS called every step — env.step() itself.
#
# This wrapper sits between the tmrl env and tmrl's worker loop. It
# runs the same reward-modification math as sample_compressor (so test
# metrics are directly comparable to training ones) and writes a CSV
# row at every episode end. It is activated only for --test.
# =====================================================================

class TestLoggingEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._ep_steps = 0
        self._ep_reward_scaled = 0.0
        self._ep_wall_penalty_scaled = 0.0
        self._ep_speed_reward_scaled = 0.0

    def reset(self, **kwargs):
        self._reset_accumulators()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Mirror the sample_compressor math so training and test metrics
        # are comparable.
        wall_penalty = _wall_proximity_penalty(obs)
        speed_rew = _speed_reward(obs)
        rew_scaled = (reward + speed_rew - wall_penalty) * REWARD_SCALE

        self._ep_steps += 1
        self._ep_reward_scaled += float(rew_scaled)
        self._ep_wall_penalty_scaled += wall_penalty * REWARD_SCALE
        self._ep_speed_reward_scaled += speed_rew * REWARD_SCALE

        if terminated or truncated:
            final_progress = float(obs[PROGRESS_OBS_INDEX].flat[0])
            if TEST_MODE_ACTIVE:
                _log_test_episode(
                    steps=self._ep_steps,
                    final_progress=final_progress,
                    terminated=terminated,
                    truncated=truncated,
                    total_reward=self._ep_reward_scaled,
                    speed_reward_total=self._ep_speed_reward_scaled,
                    wall_penalty_total=self._ep_wall_penalty_scaled,
                )

        return obs, reward, terminated, truncated, info

# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================

env_cls = cfg_obj.ENV_CLS
device_worker = 'cpu'

# =====================================================================
# ENVIRONMENT PARAMETERS
# =====================================================================

window_width = cfg.WINDOW_WIDTH
window_height = cfg.WINDOW_HEIGHT
imgs_buf_len = cfg.IMG_HIST_LEN
act_buf_len = cfg.ACT_BUF_LEN

LIDAR_BEAMS = 19
LIDAR_HISTORY = imgs_buf_len

SPEED_MAX = 300.0
LIDAR_MAX = 458.0

MLP_INPUT_SIZE = 1 + 1 + (LIDAR_HISTORY * LIDAR_BEAMS) + (act_buf_len * 3)

# =====================================================================
# MEMORY CLASS
# =====================================================================

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)

# =====================================================================
# DISCRETE ACTION SPACE — 10 ACTIONS
# =====================================================================

DISCRETE_ACTIONS = [
    np.array([1.0, 0.0,  0.0], dtype=np.float32),  # 0: gas, straight
    np.array([1.0, 0.0, -0.3], dtype=np.float32),  # 1: gas, slight left
    np.array([1.0, 0.0,  0.3], dtype=np.float32),  # 2: gas, slight right
    np.array([1.0, 0.0, -0.6], dtype=np.float32),  # 3: gas, soft left
    np.array([1.0, 0.0,  0.6], dtype=np.float32),  # 4: gas, soft right
    np.array([1.0, 0.0, -1.0], dtype=np.float32),  # 5: gas, hard left
    np.array([1.0, 0.0,  1.0], dtype=np.float32),  # 6: gas, hard right
    np.array([0.0, 0.0,  0.0], dtype=np.float32),  # 7: coast, straight
    np.array([0.0, 0.0, -0.5], dtype=np.float32),  # 8: coast, 50% left
    np.array([0.0, 0.0,  0.5], dtype=np.float32),  # 9: coast, 50% right
]

NUM_ACTIONS = len(DISCRETE_ACTIONS)

ACTION_LABELS = [
    "GAS+STR ", "GAS+sL  ", "GAS+sR  ", "GAS+SL  ", "GAS+SR  ",
    "GAS+HL  ", "GAS+HR  ", "COA+STR ", "COA+L50 ", "COA+R50 ",
]


def _action_to_label(act):
    act = np.asarray(act, dtype=np.float32)
    best_idx = -1
    best_dist = float('inf')
    for i, ref in enumerate(DISCRETE_ACTIONS):
        dist = float(np.linalg.norm(act - ref))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_dist > 0.1:
        return f"???[{act[0]:+.1f},{act[1]:+.1f},{act[2]:+.1f}]"
    return ACTION_LABELS[best_idx]


ACTIONS_TENSOR = torch.tensor(
    np.stack(DISCRETE_ACTIONS), dtype=torch.float32
)

# =====================================================================
# JSON SERIALIZERS
# =====================================================================

class TorchJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


# =====================================================================
# DQN ACTOR MODULE
# =====================================================================

class DQNActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space, epsilon=1.0):
        super().__init__(observation_space, action_space)
        self.epsilon = epsilon
        self.mlp = nn.Sequential(
            nn.Linear(MLP_INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS),
        )

    def forward(self, obs):
        speed, progress, lidar, act1, act2 = obs
        speed = speed.view(speed.size(0), -1) / SPEED_MAX
        progress = progress.view(progress.size(0), -1)
        lidar = lidar.view(lidar.size(0), -1) / LIDAR_MAX
        act1 = act1.view(act1.size(0), -1)
        act2 = act2.view(act2.size(0), -1)
        x = torch.cat((speed, progress, lidar, act1, act2), dim=-1)
        return self.mlp(x)

    def act(self, obs, test=False):
        if not test and np.random.random() < self.epsilon:
            return DISCRETE_ACTIONS[np.random.randint(NUM_ACTIONS)]
        with torch.no_grad():
            obs_batched = tuple(o.unsqueeze(0) for o in obs)
            q_values = self.forward(obs_batched)
            action_idx = q_values.argmax(dim=-1).item()
        return DISCRETE_ACTIONS[action_idx]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        if not os.path.exists(path):
            print(f"INFO: No weights file at {path}, using random initialization")
            self.to_device(device)
            return self
        try:
            state_dict = torch.load(path, map_location=device)
            self.load_state_dict(state_dict, strict=True)
            print(f"INFO: Loaded DQN weights from {path}")
        except RuntimeError as e:
            print(f"WARNING: Skipping incompatible weights at {path}: {e}")
        except Exception as e:
            print(f"WARNING: Could not load weights from {path}: {e}")
        self.to_device(device)
        return self


# =====================================================================
# DQN TRAINING AGENT
# =====================================================================

class DQNTrainingAgent(TrainingAgent):
    def get_actor(self):
        return self.model

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=DQNActorModule,
                 gamma=0.93,
                 lr=0.00003,
                 target_update_interval=250,
                 epsilon_start=0.7,
                 epsilon_end=0.20,
                 epsilon_decay=0.9897,
                 max_grad_norm=10.0,
                 snapshot_every=CHECKPOINT_SNAPSHOT_EVERY,
                 snapshot_dir=CHECKPOINT_SNAPSHOT_DIR,
                 snapshot_keep=CHECKPOINT_SNAPSHOT_KEEP):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        self.model = model_cls(observation_space, action_space,
                               epsilon=epsilon_start).to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.gamma = gamma
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.target_update_interval = target_update_interval
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.max_grad_norm = max_grad_norm
        self.steps = 0
        self.steps_per_round = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
        self.actions_tensor = ACTIONS_TENSOR.to(device)

        self.snapshot_every = snapshot_every
        self.snapshot_dir = snapshot_dir
        self.snapshot_keep = snapshot_keep
        if self.snapshot_every and self.snapshot_dir:
            os.makedirs(self.snapshot_dir, exist_ok=True)

    def _take_snapshot(self):
        if not self.snapshot_every or not self.snapshot_dir:
            return
        try:
            run_name = cfg.TMRL_CONFIG.get("RUN_NAME", "run")
            filename = f"{run_name}_step{self.steps:07d}.tmod"
            full_path = os.path.join(self.snapshot_dir, filename)
            torch.save(self.model.state_dict(), full_path)
            existing = sorted(
                f for f in os.listdir(self.snapshot_dir)
                if f.startswith(f"{run_name}_step") and f.endswith(".tmod")
            )
            while len(existing) > self.snapshot_keep:
                old = existing.pop(0)
                try:
                    os.remove(os.path.join(self.snapshot_dir, old))
                except OSError:
                    pass
        except Exception as e:
            print(f"WARNING: snapshot at step {self.steps} failed: {e}")

    def train(self, batch):
        o, a, r, o2, d, _ = batch

        diffs = a.unsqueeze(1) - self.actions_tensor.unsqueeze(0)
        action_indices = diffs.norm(dim=2).argmin(dim=1)

        q_values = self.model.forward(o)
        q_taken = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model.forward(o2).argmax(dim=1)
            q_next = self.model_target.forward(o2)
            q_next_selected = q_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target = r + self.gamma * (1.0 - d) * q_next_selected

        loss = F.smooth_l1_loss(q_taken, q_target)
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.model_target = no_grad(deepcopy(self.model))

        if self.steps % self.steps_per_round == 0:
            self.model.epsilon = max(self.epsilon_end,
                                     self.model.epsilon * self.epsilon_decay)

        if self.snapshot_every and self.steps % self.snapshot_every == 0:
            self._take_snapshot()

        with torch.no_grad():
            q_max_all = q_values.detach().max(dim=1).values
            td_error = (q_taken.detach() - q_target).abs()

            done_mask = d.bool()
            done_count = done_mask.sum().item()

            if done_count > 0:
                progress_at_done = o2[PROGRESS_OBS_INDEX][done_mask].view(-1)
                progress_at_done_mean = progress_at_done.mean().item()
                track_completions = (progress_at_done > 0.95).sum().item()
            else:
                progress_at_done_mean = float('nan')
                track_completions = 0

            action_counts = torch.zeros(NUM_ACTIONS, device=self.device)
            for i in range(NUM_ACTIONS):
                action_counts[i] = (action_indices == i).sum().item()
            most_common_action = action_counts.argmax().item()
            action_probs = action_counts / action_counts.sum()
            action_entropy = -(action_probs * (action_probs + 1e-8).log()).sum().item()

            action_share = {
                f"action_share_{i}_{ACTION_LABELS[i].strip()}":
                    (action_counts[i] / action_counts.sum()).item()
                for i in range(NUM_ACTIONS)
            }

            coast_indices = {7, 8, 9}
            coast_mask = torch.tensor(
                [i.item() in coast_indices for i in action_indices],
                device=self.device, dtype=torch.float32
            )
            coast_share = coast_mask.mean().item()

            speed_batch = o[SPEED_OBS_INDEX].view(-1)

            q_max_current = q_max_all.max().item()
            divergence_flag = 1.0 if q_max_current > DIVERGENCE_Q_MAX_THRESHOLD else 0.0

        ret = dict(
            loss_dqn=loss.detach().item(),
            epsilon=self.model.epsilon,
            q_mean=q_max_all.mean().item(),
            q_max=q_max_current,
            q_min=q_max_all.min().item(),
            q_std=q_max_all.std().item(),
            q_target_mean=q_target.mean().item(),
            td_error_mean=td_error.mean().item(),
            td_error_max=td_error.max().item(),
            reward_batch_mean=r.mean().item(),
            reward_batch_max=r.max().item(),
            reward_batch_min=r.min().item(),
            progress_at_done_mean=progress_at_done_mean,
            track_completions=track_completions,
            done_count=done_count,
            grad_norm_pre_clip=grad_norm,
            action_most_common=most_common_action,
            action_entropy=action_entropy,
            coast_share=coast_share,
            divergence_flag=divergence_flag,
            speed_batch_mean=speed_batch.mean().item(),
        )
        ret.update(action_share)
        return ret


# =====================================================================
# TRAINING AGENT INSTANTIATION
# =====================================================================

training_agent_cls = partial(DQNTrainingAgent,
                             model_cls=DQNActorModule,
                             gamma=0.93,
                             lr=0.00003,
                             target_update_interval=250,
                             epsilon_start=0.7,
                             epsilon_end=0.2,
                             epsilon_decay=0.9897,
                             max_grad_norm=10.0)

# =====================================================================
# TMRL TRAINER
# =====================================================================

training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)

# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode with CSV logging')
    parser.add_argument('--wandb', action='store_true', help='logs training metrics to wandb')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security,
                             checkpoint_path=cfg.CHECKPOINT_PATH)
        if args.wandb:
            print(f"Running trainer with:\nWandb Entity: {wandb_entity}\nWandb Project: {wandb_project}\nWandb Run ID: {wandb_run_id}")
            my_trainer.run_with_wandb(entity=wandb_entity,
                                      project=wandb_project,
                                      run_id=wandb_run_id)
        else:
            print("Running without Wandb")
            my_trainer.run()

    elif args.worker or args.test:
        if args.test:
            # Wrap env_cls so env.step()/reset() go through the logger.
            # sample_compressor is NOT called in test mode (tmrl uses
            # collect_samples=False), so logging must happen here instead.
            original_env_cls = env_cls
            def worker_env_cls(*a, **kw):
                return TestLoggingEnvWrapper(original_env_cls(*a, **kw))
        else:
            worker_env_cls = env_cls

        rw = RolloutWorker(env_cls=worker_env_cls,
                           actor_module_cls=DQNActorModule,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=args.test)
        if args.test:
            _init_test_log()
            print("Running in test-only mode (greedy, no exploration).")
            print("Press CTRL+C to stop and see the final summary.\n")
            try:
                while True:
                    rw.run_episode(max_samples_per_episode, train=False)
            except KeyboardInterrupt:
                _print_final_test_summary()
                if _test_log_handle is not None:
                    _test_log_handle.close()
        else:
            rw.run(test_episode_interval=5)

    elif args.server:
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)