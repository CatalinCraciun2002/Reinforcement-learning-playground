"""
Autoencoder Trainer — Next-Frame Prediction from Human Gameplay

Trains AutoencoderNetwork to predict the next game state given current state.
On each checkpoint, two weight files are saved:
  - model_last.pth / model_best.pth  — full autoencoder (resume autoencoder training)
  - backbone_for_actor_critic.pth     — backbone weights only, compatible with
                                        ActorCriticNetwork (strict=False load)
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from reinforcement_learning.base_trainer import BaseTrainer
from models.autoencoder_models.autoencoder_network import AutoencoderNetwork
from human_feedback.data_loader import GameplayDataset


ACTION_TO_IDX = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Stop': 4}


def _build_transition_pairs(dataset, memory_length):
    """Build (state_tensor, next_state_tensor) pairs from all episodes.

    next_state_tensor uses only the 6 base channels (no memory history)
    since that is what the decoder must learn to predict.

    Returns:
        List of dicts with keys 'state' (6+mem, H, W) and 'target' (6, H, W).
        Also returns average episode score for display.
    """
    pairs = []
    episode_scores = []

    for episode in dataset.episodes:
        walls = episode['walls']
        transitions = episode['transitions']
        if len(transitions) < 2:
            continue

        if transitions:
            episode_scores.append(transitions[-1]['state'].get('score', 0))

        for i in range(len(transitions) - 1):
            # Current state with memory
            past_positions = [
                transitions[j]['state']['pacman_pos']
                for j in range(max(0, i - memory_length), i)
            ]
            state_t = dataset.state_to_tensor(
                transitions[i]['state'], walls,
                past_positions=past_positions,
                memory_length=memory_length
            )

            # Next state — base 6 channels only (no memory)
            target_t = dataset.state_to_tensor(
                transitions[i + 1]['state'], walls,
                past_positions=[],
                memory_length=0
            )

            pairs.append({'state': state_t, 'target': target_t})

    return pairs, episode_scores


class AutoencoderTrainer(BaseTrainer):
    """Next-frame prediction trainer using human gameplay recordings."""

    def __init__(
        self,
        data_dir='game_runs_data',
        num_epochs=100,
        batch_size=64,
        lr=1e-4,
        memory_length=5,
        val_split=0.1,
        resume_from=None
    ):
        self.data_dir = data_dir
        self.batch_size_train = batch_size
        self.memory_length = memory_length
        self.val_split = val_split
        self.lr = lr

        self._train_pairs = None
        self._val_pairs = None

        hyperparams = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'memory_length': memory_length,
            'val_split': val_split,
        }

        super().__init__(
            training_type='autoencoding',
            num_epochs=num_epochs,
            hyperparams=hyperparams,
            resume_from=resume_from,
            use_best_checkpoint=False
        )

    def create_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}\n")

        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model = AutoencoderNetwork(memory_context=self.memory_length).to(device)
        
        if device.type == 'cuda':
            try:
                print("Attempting to compile the model with torch.compile...")
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compilation successful.\n")
            except Exception as e:
                print(f"torch.compile fallback: {e}\nContinuing with standard execution.\n")
                
        return model

    def create_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=self.lr)

    def post_setup(self):
        """Load human gameplay data and split into train / val."""
        dataset = GameplayDataset(self.data_dir)
        if not dataset.episodes:
            raise ValueError("No gameplay recordings found in: " + self.data_dir)
        dataset.print_statistics()

        all_pairs, episode_scores = _build_transition_pairs(dataset, self.memory_length)
        print(f"Transition pairs: {len(all_pairs)} from {len(episode_scores)} episodes")
        print(f"Average episode score: {np.mean(episode_scores):.1f}\n")

        split = max(1, int(len(all_pairs) * (1 - self.val_split)))
        self._train_pairs = all_pairs[:split]
        self._val_pairs = all_pairs[split:]
        print(f"Train: {len(self._train_pairs)}  Val: {len(self._val_pairs)}")

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train_epoch(self, epoch):
        self.model.train()
        device = next(self.model.parameters()).device
        np.random.shuffle(self._train_pairs)

        total_loss = 0.0
        channel_losses = np.zeros(AutoencoderNetwork.NUM_BASE_CHANNELS)
        num_batches = 0

        for i in range(0, len(self._train_pairs), self.batch_size_train):
            batch = self._train_pairs[i:i + self.batch_size_train]
            states  = torch.stack([b['state']  for b in batch]).to(device)
            targets = torch.stack([b['target'] for b in batch]).to(device)

            preds = self.model(states)          # (B, 6, H, W)
            loss, ch_losses = self._bce_loss(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss  += loss.item()
            channel_losses += ch_losses
            num_batches += 1

        avg = total_loss / num_batches
        ch_avg = channel_losses / num_batches

        metrics = {'Loss/train': avg}
        for c, name in enumerate(['pacman', 'ghosts', 'walls', 'scared', 'food', 'capsules']):
            metrics[f'Loss/ch_{name}'] = float(ch_avg[c])
        return metrics

    def validate(self, epoch):
        self.model.eval()
        device = next(self.model.parameters()).device

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(self._val_pairs), self.batch_size_train):
                batch = self._val_pairs[i:i + self.batch_size_train]
                states  = torch.stack([b['state']  for b in batch]).to(device)
                targets = torch.stack([b['target'] for b in batch]).to(device)
                preds = self.model(states)
                loss, _ = self._bce_loss(preds, targets)
                total_loss += loss.item()
                num_batches += 1

        return {'Loss/val': total_loss / max(num_batches, 1)}

    def get_metric_for_checkpoint(self, val_metrics):
        # Lower BCE is better — negate so BaseTrainer's "is_best if > best" logic works
        return -val_metrics['Loss/val'], 'neg_val_bce'

    def get_progress_bar_dict(self, train_metrics, val_metrics):
        return {
            'TrainLoss': f"{train_metrics.get('Loss/train', 0):.4f}",
            'ValLoss':   f"{val_metrics.get('Loss/val', 0):.4f}",
        }

    def get_final_summary(self):
        return {'Best Val BCE': f"{-self.best_metric:.5f}"}

    # ------------------------------------------------------------------ #
    #  Dual checkpoint export                                              #
    # ------------------------------------------------------------------ #

    def on_epoch_end(self, epoch, pbar):
        """Save a backbone-only weights file alongside the main checkpoint."""
        if self.log_dir is None:
            return
        backbone_path = os.path.join(self.log_dir, 'backbone_for_actor_critic.pth')
        torch.save(self.model.backbone_state_dict(), backbone_path)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bce_loss(preds, targets):
        """Per-channel BCE, returns (total_scalar, per_channel_array)."""
        # preds/targets: (B, 6, H, W)
        channel_losses = torch.stack([
            nn.functional.binary_cross_entropy(preds[:, c], targets[:, c])
            for c in range(AutoencoderNetwork.NUM_BASE_CHANNELS)
        ])  # (6,)
        return channel_losses.sum(), channel_losses.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Autoencoder next-frame prediction trainer')
    parser.add_argument('--data-dir', type=str, default='game_runs_data',
                        help='Directory containing human gameplay recordings')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--memory-length', type=int, default=5,
                        help='Past Pacman positions to include in the input')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Fraction of transitions held out for validation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint or run directory to resume from')

    args = parser.parse_args()

    trainer = AutoencoderTrainer(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        memory_length=args.memory_length,
        val_split=args.val_split,
        resume_from=args.resume
    )
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
