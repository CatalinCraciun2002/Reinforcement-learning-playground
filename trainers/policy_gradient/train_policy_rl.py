"""
PPO (Proximal Policy Optimization) Training Script
Uses GAE for advantage calculation and clipped surrogate objective.
"""
import torch
import torch.optim as optim
import numpy as np
import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from trainers.base_trainer import BaseTrainer
from models.policy_gradient_models.simple_residual_conv import ActorCriticNetwork
from agents.policy_gradient_agents.deepRlAgent import RLAgent
from core.game_orchestrator import GameOrchestrator
from core.game import Directions
from display import graphicsDisplay


class PolicyGradientTrainer(BaseTrainer):
    """PPO Trainer using GAE and clipped surrogate objective."""
    
    def __init__(self, args):
        super().__init__(
            training_type='policy_gradient',
            args=args
        )
        self.memory_context = getattr(args, 'memory_context', 5)
        
        self.agent = None
        self.orchestrator = None
        self.wins = 0
        self.total_steps = 0
    
    def create_model(self):
        """Create Actor-Critic network."""
        return ActorCriticNetwork(memory_context=self.memory_context)
    
    def create_optimizer(self, model):
        """Create Adam optimizer."""
        return optim.Adam(model.parameters(), lr=self.lr)
    
    def post_setup(self):
        """Setup agent and orchestrator after model/optimizer creation."""
        self.agent = RLAgent(self.model, memory_context=self.memory_context)
        
        self.orchestrator = GameOrchestrator(
            agent=self.agent,
            batch_size=self.batch_size,
            train_suite_name=self.train_suite,
            test_suite_name=self.test_suite,
        )
        
        # Create visualizer if enabled
        if self.save_visualization_data:
            self.visualizer = self.create_visualizer()
    
    
    def create_visualizer(self):
        """Create EpochVisualizer instance."""
        from reinforcement_learning.training_visualization.epoch_visualizer import EpochVisualizer
        
        # Store visualization data in the same directory as TensorBoard logs
        vis_dir = os.path.join(self.log_dir, 'visualization_data')
        os.makedirs(vis_dir, exist_ok=True)
        return EpochVisualizer(vis_dir, self.hyperparams)
    
    def train_epoch(self, epoch):
        """Train for one epoch: collect rollouts, compute GAE, run PPO updates."""
        self.model.train()

        # Sync epoch to all environments so scenarios can react to training progress
        self.orchestrator.set_epoch(epoch)
        
        if self.save_visualization_data:
            self.on_visualization_epoch_start(epoch, self.batch_size)
        
        # Storage per environment
        env_data = [
            {
                'state_tensors': [],
                'action_indices': [],
                'old_log_probs': [],
                'old_values': [],
                'rewards': [],
                'td_errors': [],
                'game_overs': [],
                'episode_steps': 0
            }
            for _ in range(self.batch_size)
        ]
        
        # Get initial states from all environments
        states = self.orchestrator.get_all_states()
        legal_actions = self.orchestrator.get_all_legal(states)
        env_ids = list(range(self.batch_size))
        
        # Batched forward pass for initial states
        probs_batch, values_batch = self.agent.forward_batch(states, env_ids)
        
        for step_idx in range(self.steps_per_epoch):
            next_states = []
            rewards = []
            dones = []
            actions_taken = []
            action_indices = []
            
            # Store state tensors for PPO re-evaluation (before actions)
            step_tensors = [
                self.agent.state_to_tensor(state, env_id).detach()
                for state, env_id in zip(states, env_ids)
            ]
            
            for env_idx, (probs, value, legal) in enumerate(
                zip(probs_batch, values_batch, legal_actions)
            ):
                action, action_idx = self.agent.getAction(legal, probs)
                actions_taken.append((action, action_idx))
                action_indices.append(action_idx)
                
                next_state, reward, done = self.orchestrator.step(env_idx, action)
                env_data[env_idx]['episode_steps'] += 1
                
                if done:
                    self.total_steps += env_data[env_idx]['episode_steps']
                    env_data[env_idx]['episode_steps'] = 0
                    self.orchestrator.reset(env_idx)
                    next_state = self.orchestrator.get_state(env_idx)
                
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
            
            # Compute and store old log probs (detached)
            action_indices_tensor = torch.tensor(action_indices, dtype=torch.long)
            old_log_probs = self.model.last_log_probs[
                torch.arange(self.batch_size), action_indices_tensor
            ].detach()
            
            for env_idx in range(self.batch_size):
                env_data[env_idx]['state_tensors'].append(step_tensors[env_idx])
                env_data[env_idx]['action_indices'].append(action_indices[env_idx])
                env_data[env_idx]['old_log_probs'].append(old_log_probs[env_idx])
                env_data[env_idx]['old_values'].append(values_batch[env_idx].detach())
                env_data[env_idx]['rewards'].append(rewards[env_idx])
            
            # Batched forward pass for next states
            next_probs_batch, next_values_batch = self.agent.forward_batch(next_states, env_ids)
            
            # TD errors (for GAE)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)
            td_errors = rewards_tensor + self.gamma * next_values_batch.detach() * (1 - dones_tensor) - values_batch.detach()
            
            for env_idx in range(self.batch_size):
                env_data[env_idx]['td_errors'].append(td_errors[env_idx])
                env_data[env_idx]['game_overs'].append(dones[env_idx])
                
                if self.save_visualization_data:
                    action, action_idx = actions_taken[env_idx]
                    step_data = {
                        'state': states[env_idx],
                        'legal_actions': legal_actions[env_idx],
                        'action_probs': probs_batch[env_idx],
                        'selected_action': action,
                        'selected_action_idx': action_idx,
                        'value': values_batch[env_idx],
                        'reward': rewards[env_idx],
                        'next_value': next_values_batch[env_idx],
                        'td_error': td_errors[env_idx],
                        'td_target': rewards_tensor[env_idx] + self.gamma * next_values_batch[env_idx].detach() * (1 - dones_tensor[env_idx]),
                        'done': dones[env_idx]
                    }
                    self.on_visualization_step(env_idx, step_data)
            
            states = next_states
            legal_actions = self.orchestrator.get_all_legal(states)
            probs_batch = next_probs_batch
            values_batch = next_values_batch
        
        # --- Compute GAE advantages and returns ---
        all_state_tensors = []
        all_action_indices = []
        all_old_log_probs = []
        all_old_values = []
        all_advantages = []
        
        for env_idx in range(self.batch_size):
            advantages = []
            gae = 0
            
            for done, td_error in zip(
                reversed(env_data[env_idx]['game_overs']),
                reversed(env_data[env_idx]['td_errors'])
            ):
                if done:
                    advantage = td_error
                else:
                    advantage = td_error + self.gamma * self.lam * gae
                advantages.append(advantage)
                gae = advantage.item()
            
            advantages.reverse()
            
            if self.save_visualization_data:
                self.on_visualization_advantages(env_idx, advantages)
            
            all_state_tensors.extend(env_data[env_idx]['state_tensors'])
            all_action_indices.extend(env_data[env_idx]['action_indices'])
            all_old_log_probs.extend(env_data[env_idx]['old_log_probs'])
            all_old_values.extend(env_data[env_idx]['old_values'])
            all_advantages.extend(advantages)
        
        # Stack into tensors
        all_state_tensors = torch.cat(all_state_tensors, dim=0)          # (N, C, H, W)
        all_action_indices = torch.tensor(all_action_indices, dtype=torch.long)  # (N,)
        all_old_log_probs = torch.stack(all_old_log_probs)               # (N,)
        all_old_values = torch.stack(all_old_values)                     # (N,)
        all_advantages = torch.stack(all_advantages)                     # (N,)
        
        # Returns for value loss
        all_returns = all_advantages + all_old_values
        
        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # --- PPO multi-epoch mini-batch updates ---
        n_samples = all_state_tensors.size(0)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_loss_val = 0.0
        num_updates = 0

        # Switch to eval so BatchNorm uses stable running statistics.
        # Re-compute old_log_probs in eval mode so that both old and new
        # log probs use the same BN behavior — the rollout-time log probs
        # used per-batch BN stats (train mode) which differ from mini-batch
        # stats, making the PPO ratio unreliable.
        self.model.eval()
        with torch.no_grad():
            old_lp_chunks = []
            for i in range(0, n_samples, self.mini_batch_size):
                chunk = all_state_tensors[i:i + self.mini_batch_size]
                self.model(chunk, return_both=True)
                chunk_actions = all_action_indices[i:i + self.mini_batch_size]
                old_lp_chunks.append(
                    self.model.last_log_probs[torch.arange(len(chunk_actions)), chunk_actions]
                )
            all_old_log_probs = torch.cat(old_lp_chunks).detach()

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n_samples)
            
            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                mb_idx = indices[start:end]
                
                mb_states = all_state_tensors[mb_idx]
                mb_actions = all_action_indices[mb_idx]
                mb_old_log_probs = all_old_log_probs[mb_idx]
                mb_advantages = all_advantages[mb_idx]
                mb_returns = all_returns[mb_idx]
                
                # Re-evaluate through model directly (bypass agent to avoid buffer side effects)
                new_probs, new_values = self.model(mb_states, return_both=True)
                new_values = new_values.squeeze(-1)
                
                new_log_probs = self.model.last_log_probs[
                    torch.arange(len(mb_idx)), mb_actions
                ]

                # PPO clipped surrogate
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                ratio = torch.clamp(ratio, 1e-3, 10.0)  # Safety clamp
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = ((new_values - mb_returns) ** 2).mean()
                
                # Entropy bonus
                entropy = -(new_probs * self.model.last_log_probs).sum(dim=1).mean()
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                if torch.isnan(loss) or torch.isinf(loss):
                    continue  # skip corrupted mini-batch

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_loss_val += loss.item()
                num_updates += 1

        self.model.train()

        self.wins += self.orchestrator.total_wins()
        
        if self.save_visualization_data:
            self.on_visualization_epoch_end()
        
        avg_steps = self.total_steps / self.orchestrator.total_wins_plus_one() if self.total_steps > 0 else 0
        n = max(num_updates, 1)

        return {
            'Loss/total': total_loss_val / n,
            'Loss/actor': total_actor_loss / n,
            'Loss/critic': total_critic_loss / n,
            'Loss/entropy_bonus': total_entropy / n,
            'Performance/total_wins': self.wins,
            'Performance/avg_steps': avg_steps,
        }
    
    def validate(self, epoch):
        """Run validation games via the orchestrator."""
        self.model.eval()
        
        results = self.orchestrator.run_validation(
            n_games=self.validation_games, with_graphics=False
        )
        
        val_scores = [r[0] for r in results]
        val_wins = [1 if r[1] else 0 for r in results]
        
        val_score = sum(val_scores) / len(val_scores)
        val_won = sum(val_wins) / len(val_wins)
        
        self.model.train()
        
        return {
            'Score/score': val_score,
            'Score/won': val_won
        }
    
    def get_metric_for_checkpoint(self, val_metrics):
        """Use validation average score as the metric for checkpointing."""
        return val_metrics['Score/score'], 'avg_score'
    
    def get_progress_bar_dict(self, train_metrics, val_metrics):
        """Customize progress bar display."""
        return {
            'Actor': f"{train_metrics.get('Loss/actor', 0):.4f}",
            'Critic': f"{train_metrics.get('Loss/critic', 0):.2f}",
            'ValScore': f"{val_metrics.get('Score/score', 0):.1f}"
        }
    
    def on_epoch_end(self, epoch, pbar):
        """Display validation with graphics at specified intervals."""
        if self.show_epochs > 0 and (epoch + 1) % self.show_epochs == 0:
            pbar.write(f"\n{'='*60}")
            pbar.write(f"Running validation game with graphics at epoch {epoch+1}...")
            pbar.write('='*60)
            
            self.model.eval()
            score, won, steps = self.orchestrator.run_single_validation(with_graphics=True)
            
            pbar.write(f"\nValidation Result - Score: {score}, {'WON!' if won else 'Lost'}, Steps: {steps}")
            pbar.write('='*60 + '\n')
            
            self.model.train()
    
    def get_final_summary(self):
        """Get final training summary."""
        end_epoch = self.start_epoch + self.num_epochs
        final_win_rate = f"{self.wins}/{end_epoch * self.batch_size} = {self.wins/(end_epoch * self.batch_size) if (end_epoch * self.batch_size) > 0 else 0:.2%}"
        
        return {
            'Final Win Rate': final_win_rate,
            'Total Steps': str(self.total_steps)
        }
    
    def get_additional_checkpoint_data(self):
        """Save additional policy gradient data."""
        return {
            'wins': self.wins,
            'total_steps': self.total_steps,
            'val_avg_score': self.best_metric
        }


    
def main():
    parser = PolicyGradientTrainer.build_parser()
    
    # Add script-specific arguments
    parser.add_argument('--memory-context', type=int, default=5, help='Number of past positions to remember')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run a validation game without training', default=False)
    
    # For parameters that differ in defaults from the base parser, we could update the defaults:
    parser.set_defaults(num_epochs=400, batch_size=32, steps_per_epoch=64, mini_batch_size=128, validation_games=8)
    
    args = parser.parse_args()
    
    # Validation-only mode: load checkpoint and run a single validation game
    if args.validate_only:
        if not args.resume:
            print("Error: --validate-only requires --resume to specify a checkpoint")
            return
        
        print(f"Running validation game from checkpoint: {args.resume}")
        
        # Determine checkpoint path
        checkpoint_name = 'model_best.pth' if getattr(args, 'use_best', False) else 'model_last.pth'
        checkpoint_path = os.path.join(args.resume, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return
        
        # Load checkpoint directly without creating trainer
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Create model and load weights
        model = ActorCriticNetwork(memory_context=args.memory_context)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        # Create agent and orchestrator
        agent = RLAgent(model, memory_context=args.memory_context)
        orchestrator = GameOrchestrator(
            agent=agent,
            batch_size=1,
            train_suite_name=args.train_suite,
            test_suite_name=args.test_suite,
        )
        
        # Run a single validation game with graphics
        print("Starting validation game with graphics...")
        score, won, steps = orchestrator.run_single_validation(with_graphics=True)
        
        print(f"\nValidation Result:")
        print(f"  Score: {score}")
        print(f"  Result: {'WON!' if won else 'Lost'}")
        print(f"  Steps: {steps}")
        
        return
    
    # Normal training mode
    trainer = PolicyGradientTrainer(args)
    
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
