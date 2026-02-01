# Human Feedback Recording System

This directory contains tools for recording human gameplay to collect expert demonstrations for training.

## Files

- **`record_gameplay.py`**: Play Pacman with keyboard controls and record all gameplay data
- **`data_loader.py`**: Load and process recorded gameplay for training
- **`game_runs_data/`**: Directory where recorded games are saved

## Usage

### Recording Gameplay

```bash
# Record gameplay on default layout (mediumClassic)
python human_feedback/record_gameplay.py

# Record on a different layout
python human_feedback/record_gameplay.py --layout smallClassic

# Adjust graphics zoom
python human_feedback/record_gameplay.py --zoom 1.5
```

**Controls:**
- Arrow keys or WASD to move Pacman
- ESC to exit

After each game, data is automatically saved to `game_runs_data/` with:
- **Pickle file (`.pkl`)**: Full game data including all states, actions, rewards
- **CSV file (`.csv`)**: Human-readable summary

### Loading Recorded Data

```python
from human_feedback.data_loader import GameplayDataset

# Load all recorded games
dataset = GameplayDataset()
dataset.print_statistics()

# Get a batch for training
states, actions, rewards = dataset.get_training_batch(batch_size=32)

# Get specific episode
episode = dataset.get_episode_data(0)
```

## Data Format

### Pickle File Structure
```python
{
    'layout_name': str,
    'transitions': [
        {
            'step': int,
            'state': {
                'pacman_pos': (x, y),
                'ghost_positions': [(x1, y1), ...],
                'ghost_scared_timers': [t1, t2, ...],
                'food_grid': np.array,
                'capsules': [(x, y), ...]
            },
            'action': str,  # 'North', 'South', 'East', 'West'
            'reward': float,
            'done': bool
        },
        ...
    ],
    'walls': np.array,  # Static wall grid
    'final_score': float,
    'outcome': str,  # 'WIN', 'LOSS', or 'INCOMPLETE'
    'num_steps': int,
    'reward_constants': dict
}
```

## Imitation Learning

Use recorded human gameplay to:
1. **Bootstrap RL training**: Initialize policy with expert demonstrations
2. **Behavioral cloning**: Train network to mimic human actions
3. **Reward shaping**: Analyze which actions lead to higher rewards
4. **Curriculum learning**: Start with good examples before self-play
