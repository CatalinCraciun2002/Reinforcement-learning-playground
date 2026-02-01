# TensorBoard Guide for PacMan RL Project

## Overview

TensorBoard is now integrated into both training pipelines:
- **Human Feedback Training** (`human_feedback/train_human_feedback.py`)
- **RL Policy Training** (`reinforcement_learning/train_policy_rl.py`)

Both scripts automatically create timestamped log directories and record metrics during training.

## Quick Start

### 1. Start Training

When you run either training script, logs are automatically created:

```bash
# Human feedback training
python human_feedback/train_human_feedback.py

# RL policy training  
python reinforcement_learning/train_policy_rl.py
```

You'll see a message like:
```
TensorBoard logging to: runs/human_feedback/20260201_211500
```

### 2. Launch TensorBoard

Open a **new terminal** (keep training running) and run:

```bash
tensorboard --logdir=runs
```

This will start TensorBoard server. You'll see output like:
```
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
```

### 3. View in Browser

Open your browser and go to:
```
http://localhost:6006
```

## What Metrics Are Logged?

### Human Feedback Training

**Losses:**
- `Loss/train_actor` - Actor network training loss (how well it predicts actions)
- `Loss/val_actor` - Actor network validation loss
- `Loss/train_critic` - Critic network training loss (how well it estimates returns)
- `Loss/val_critic` - Critic network validation loss

**Accuracy:**
- `Accuracy/train` - Training accuracy (% of correctly predicted actions)
- `Accuracy/val` - Validation accuracy

### RL Policy Training

**Losses:**
- `Loss/total` - Combined loss (actor + critic + entropy)
- `Loss/actor` - Policy gradient loss
- `Loss/critic` - Value function loss (TD error)
- `Loss/entropy_bonus` - Entropy regularization (encourages exploration)

**Performance:**
- `Performance/win_rate` - Percentage of games won
- `Performance/total_wins` - Cumulative number of wins
- `Performance/avg_steps` - Average steps per game

**Training:**
- `Training/advantage_mean` - Mean advantage value
- `Training/advantage_std` - Standard deviation of advantages

**Validation (every N epochs):**
- `Validation/score` - Game score
- `Validation/won` - Whether the validation game was won
- `Validation/steps` - Steps taken in validation game

## TensorBoard Features

### 1. Scalars Tab
View all metrics over time. You can:
- Toggle between different runs to compare experiments
- Smooth curves using the smoothing slider
- Download data as CSV or JSON

### 2. Compare Multiple Runs
TensorBoard automatically groups runs by timestamp. To compare:
- All runs in `runs/human_feedback/` will appear together
- All runs in `runs/rl_training/` will appear together
- Use the left sidebar to select which runs to display

### 3. Useful Tips

**Filter metrics:**
- Use the search box to filter metrics (e.g., type "Loss" to see only loss curves)

**Smooth noisy data:**
- Adjust the "Smoothing" slider at the top (useful for RL training which can be noisy)

**Compare experiments:**
- Keep old run directories to compare different hyperparameters
- Rename run directories for clarity (e.g., `runs/rl_training/lr_1e-4_batch_32`)

## Directory Structure

```
PacMan/
├── runs/
│   ├── human_feedback/
│   │   ├── 20260201_211500/  # Timestamped run
│   │   └── 20260201_213000/  # Another run
│   └── rl_training/
│       ├── 20260201_214500/
│       └── 20260201_220000/
```

## Advanced Usage

### Running TensorBoard on a Specific Directory

To view only one type of training:

```bash
# Only human feedback runs
tensorboard --logdir=runs/human_feedback

# Only RL training runs
tensorboard --logdir=runs/rl_training
```

### Comparing Different Experiments

Rename your run directories to make them easier to identify:

```bash
# After training completes
mv runs/rl_training/20260201_211500 runs/rl_training/baseline_lr1e-5
mv runs/rl_training/20260201_213000 runs/rl_training/pretrained_lr1e-5
```

### Change TensorBoard Port

If port 6006 is already in use:

```bash
tensorboard --logdir=runs --port=6007
```

### Access from Another Machine

To access TensorBoard from another computer on your network:

```bash
tensorboard --logdir=runs --bind_all
```

Then visit `http://<your-ip>:6006` from any device on the network.

## Troubleshooting

**Problem: "No dashboards are active for the current data set"**
- Make sure your training script is running and generating logs
- Check that the `runs/` directory exists and contains subdirectories

**Problem: TensorBoard not updating**
- TensorBoard auto-refreshes every 30 seconds
- Click the refresh button in the top-right corner to force update

**Problem: Port already in use**
- Another TensorBoard instance may be running
- Kill it: `taskkill /F /IM tensorboard.exe` (Windows)
- Or use a different port: `tensorboard --logdir=runs --port=6007`

## Best Practices

1. **Keep training running** - TensorBoard updates in real-time as new data is written

2. **Compare pretrained vs from-scratch** - Run both configurations and compare in TensorBoard

3. **Monitor overfitting** - Watch the gap between train and validation metrics

4. **Track hyperparameter experiments** - Rename run directories with meaningful names

5. **Clean old runs** - Delete old experiment directories you no longer need

## Summary

**To use TensorBoard:**
1. Start your training script (logs are created automatically)
2. Open a new terminal and run: `tensorboard --logdir=runs`
3. Open browser to: `http://localhost:6006`
4. Explore your metrics!

Happy experimenting! 🚀
