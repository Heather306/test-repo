# Agent Continuation - Clinic Scheduling RL

Continuation context for the `continue-project-context` track, focused on extending the clinic scheduling reinforcement learning (RL) work. Use this brief to spin up a new agent quickly.

## Current State
- **Branch**: `cursor/continue-rl-training-for-project-context-4a2f` (up to date, clean tree).
- **Primary assets**:
  - `notebooks/clinic_scheduling_rl.ipynb`: end-to-end Colab-ready workflow defining `ClinicSchedulingEnv` (single provider) and `ClinicSchedulingEnvV2` (multi-provider + action masking), training PPO/MaskablePPO agents, evaluating, and plotting.
  - `docs/clinic_scheduling_rl_docs.ipynb`: prose documentation of the environments, booking planner, and training flow.
  - `models/clinic_ppo.zip`: latest exported PPO weights; stored under `/workspace/models`.
- Android app sources exist but are unrelated to the RL workflow.

## Environment & Dependencies
Install requirements when working locally/Colab:

```bash
pip install gymnasium==0.29.1 stable-baselines3==2.3.2 sb3-contrib==2.3.2 shimmy==1.3.0 plotly==5.24.1 numpy pandas ipywidgets
```

If running in Colab, execute the “Keep Notebook Synced With GitHub” cell near the top of the notebook to pull this branch into `/content/test-repo`.

## Continuing Training (Env V1)
Use the PPO block already defined in the notebook. To resume from the saved checkpoint:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

log_dir = "/content/logs" if os.path.exists("/content") else "/workspace/logs"
env_fn = lambda: ClinicSchedulingEnv(slot_minutes=7)
vec_env = make_vec_env(env_fn, n_envs=1, monitor_dir=log_dir)

model_path = os.path.join(log_dir, "ppo_clinic_scheduling")
model = PPO.load(model_path, env=vec_env)
model.learn(total_timesteps=100_000)  # adjust as needed
model.save(model_path)
```

This reuses the PPO hyperparameters already in the notebook (learning_rate `3e-4`, `n_steps` 2048, `batch_size` 256, `gamma` 0.995, `gae_lambda` 0.95).

## Continuing Training (Env V2)
`ClinicSchedulingEnvV2` supports multi-provider simulation and action masking:

```python
from sb3_contrib import MaskablePPO
from gymnasium.wrappers import TimeLimit

def make_env_v2():
    base = ClinicSchedulingEnvV2(
        num_providers=2,
        walkin_rate_per_hour=6.0,
        service_minutes_mean=12,
        late_prob=0.15,
        seeded_schedule=None,  # optionally feed planner output
    )
    return TimeLimit(base, max_episode_steps=base.day_minutes)

vec_env_v2 = make_vec_env(make_env_v2, n_envs=1)
model_v2_path = os.path.join(log_dir, "ppo_clinic_scheduling_v2")

try:
    model_v2 = MaskablePPO.load(model_v2_path, env=vec_env_v2)
except FileNotFoundError:
    model_v2 = MaskablePPO("MlpPolicy", vec_env_v2, verbose=1, learning_rate=2.5e-4)

model_v2.learn(total_timesteps=200_000)
model_v2.save(model_v2_path)
```

Action masks are produced by the env (invalid actions zeroed). Swap to regular `PPO` if `sb3-contrib` is unavailable.

## Evaluation Hooks
- Notebook cells already compute per-episode summaries (served counts, wait times, leftover queues) and draw Plotly charts; rerun them after training.
- `served_log` (list of dicts) plus pandas helpers convert episodes to DataFrames for offline analysis.
- Export refreshed weights to `/workspace/models/clinic_ppo.zip` if you want the repo artifact updated (`model.save("/workspace/models/clinic_ppo")`).

## Open Items / Next Steps
1. **Hyperparameter sweep** - run short grid/bayesian sweeps (learning rate, walk-in rate priors, provider count) with logging to compare throughput and wait metrics.
2. **Planner integration** - wire the 7-minute booking planner outputs into `ClinicSchedulingEnvV2.seeded_schedule` and log booked vs served patients.
3. **Model registry** - script repeatable export (`python scripts/export_model.py --log-dir ...`) so future agents can update `models/clinic_ppo.zip` without manual notebook edits.
4. **Evaluation automation** - persist aggregated metrics/plots (e.g., `logs/metrics.json`, `logs/queues.png`) for regression tracking between training runs.

Use this context as the starting point for the next agent tasked with advancing the RL training pipeline.
