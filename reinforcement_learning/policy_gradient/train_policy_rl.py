"""
Policy Gradient (Actor-Critic) Training Script - Refactored with BaseTrainer
Uses GAE (Generalized Advantage Estimation) for advantage calculation.
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

from reinforcement_learning.base_trainer import BaseTrainer
from models.policy_gradient_models.simple_residual_conv import ActorCriticNetwork
from agents.policy_gradient_agents.deepRlAgent import RLAgent
from core.environment import PacmanEnv
from core.game import Directions
from display import graphicsDisplay


def run_validation_game(agent, layout_name='mediumClassic', with_graphics=True, max_steps=1000):
    """Run a validation game and return score, win status, and steps taken."""
    display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.05) if with_graphics else None
    val_env = PacmanEnv(agent, layout_name, add_extra_ghost=True, display=display)
    val_env.reset()
    
    steps = 0
    game_done = False
    
    while not game_done and steps < max_steps:
        state = val_env.game.state
        legal = val_env.get_legal(state)
        
        if not legal:
            break
        
        with torch.no_grad():
            probs, _ = agent.forward(state)
        
        action, action_idx = agent.getAction(legal, probs)
        _, reward, game_done = val_env.step(action)
        steps += 1
    
    score = val_env.game.state.getScore()
    won = val_env.game.state.isWin()
    
    return score, won, steps


class PolicyGradientTrainer(BaseTrainer):
    """Policy Gradient Trainer using GAE and BaseTrainer framework."""
    
    def __init__(
        self,
        num_epochs=100,
        batch_size=32,
        steps_per_epoch=20,
        layout_name='mediumClassic',
        gamma=0.95,
        lam=0.95,
        lr=1e-4,
        memory_context=5,
        show_epochs=50,
        validation_games=8,
        resume_from=None,
        use_best_checkpoint=False,
        save_visualization_data=False
    ):
        """
        Initialize Policy Gradient Trainer.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Number of parallel environments
            steps_per_epoch: Number of steps per environment per epoch
            layout_name: Layout to train on
            gamma: Discount factor
            lam: GAE lambda parameter
            lr: Learning rate
            memory_context: Number of past positions to remember
            show_epochs: Render validation game every N epochs (0 to disable)
            validation_games: Number of validation games per epoch
            resume_from: Path to checkpoint to resume from
            use_best_checkpoint: If True, load best checkpoint instead of last
            save_visualization_data: If True, save training visualization data
        """
        # Store policy-gradient specific hyperparameters
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.layout_name = layout_name
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.memory_context = memory_context
        self.show_epochs = show_epochs
        self.validation_games = validation_games
        
        # Hyperparameters for logging
        hyperparams = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch,
            'learning_rate': lr,
            'gamma': gamma,
            'lambda': lam,
            'layout': layout_name,
            'memory_context': memory_context,
            'pretrained': resume_from is not None,
            'use_best_checkpoint': use_best_checkpoint
        }
        
        # Initialize base class
        super().__init__(
            training_type='policy_gradient',
            num_epochs=num_epochs,
            hyperparams=hyperparams,
            resume_from=resume_from,
            use_best_checkpoint=use_best_checkpoint,
            save_visualization_data=save_visualization_data
        )
        
        # Policy gradient specific tracking
        self.agent = None
        self.envs = []
        self.wins = 0
        self.total_steps = 0
    
    def create_model(self):
        """Create Actor-Critic network."""
        return ActorCriticNetwork(memory_context=self.memory_context)
    
    def create_optimizer(self, model):
        """Create Adam optimizer."""
        return optim.Adam(model.parameters(), lr=self.lr)
    
    def post_setup(self):
        """Setup agent and environments after model/optimizer creation."""
        # Create agent
        self.agent = RLAgent(self.model, memory_context=self.memory_context)
        
        # Create environments with proper env_id tracking
        self.envs = [
            PacmanEnv(self.agent, self.layout_name, add_extra_ghost=True, env_id=i) 
            for i in range(self.batch_size)
        ]
        
        
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
        """Train for one epoch using GAE with batched forward passes."""
        self.model.train()
        
        # Start visualization for this epoch
        if self.save_visualization_data:
            self.on_visualization_epoch_start(epoch, self.batch_size)
        
        # Storage per environment
        env_data = [
            {
                'log_probs': [],
                'entropies': [],
                'td_errors': [],
                'game_overs': [],
                'episode_steps': 0
            }
            for _ in range(self.batch_size)
        ]
        
        # Get initial states from all environments
        states = [env.game.state for env in self.envs]
        legal_actions = [env.get_legal(state) for state, env in zip(states, self.envs)]
        env_ids = list(range(self.batch_size))
        
        # Batched forward pass for initial states
        probs_batch, values_batch = self.agent.forward_batch(states, env_ids)
        
        # Process each timestep with all environments in parallel
        for step_idx in range(self.steps_per_epoch):
            next_states = []
            rewards = []
            dones = []
            actions_taken = []  # Track actions for visualization
            action_indices = []  # Track action indices for vectorized calculations
            
            # Execute actions in all environments
            for env_idx, (env, probs, value, legal) in enumerate(
                zip(self.envs, probs_batch, values_batch, legal_actions)
            ):
                action, action_idx = self.agent.getAction(legal, probs)
                actions_taken.append((action, action_idx))
                action_indices.append(action_idx)
                
                # Execute action
                next_state, reward, done = env.step(action)
                
                env_data[env_idx]['episode_steps'] += 1
                
                if done:
                    self.total_steps += env_data[env_idx]['episode_steps']
                    env_data[env_idx]['episode_steps'] = 0
                    env.reset()
                    next_state = env.game.state
                
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
            
            # Vectorized entropy and log prob calculations
            batch_entropies = -(probs_batch * torch.log(probs_batch + 1e-10)).sum(dim=1)  # (batch_size,)
            action_indices_tensor = torch.tensor(action_indices, dtype=torch.long)
            batch_log_probs = torch.log(probs_batch[torch.arange(self.batch_size), action_indices_tensor] + 1e-10)
            
            # Store entropy and log probs
            for env_idx in range(self.batch_size):
                env_data[env_idx]['entropies'].append(batch_entropies[env_idx])
                env_data[env_idx]['log_probs'].append(batch_log_probs[env_idx])
            
            # Batched forward pass for next states
            next_probs_batch, next_values_batch = self.agent.forward_batch(next_states, env_ids)
            
            # Vectorized TD error calculation
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)
            td_targets = rewards_tensor + self.gamma * next_values_batch.detach() * (1 - dones_tensor)
            td_errors = td_targets - values_batch
            
            # Store TD errors and record visualization data
            for env_idx in range(self.batch_size):
                env_data[env_idx]['td_errors'].append(td_errors[env_idx])
                env_data[env_idx]['game_overs'].append(dones[env_idx])
                
                # Record visualization data
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
                        'td_target': td_targets[env_idx],
                        'done': dones[env_idx]
                    }
                    self.on_visualization_step(env_idx, step_data)
            
            # Update for next iteration
            states = next_states
            legal_actions = [env.get_legal(state) for state, env in zip(states, self.envs)]
            probs_batch = next_probs_batch
            values_batch = next_values_batch
        
        # Calculate GAE for each environment and aggregate
        all_log_probs = []
        all_advantages = []
        all_entropies = []
        all_td_errors = []
        
        for env_idx in range(self.batch_size):
            # GAE calculation (unchanged)
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
                gae = advantage.detach()
            
            # Reverse advantages to match original order
            advantages.reverse()
            
            # Record advantages for visualization
            if self.save_visualization_data:
                self.on_visualization_advantages(env_idx, advantages)
            
            # Aggregate
            all_log_probs.extend(env_data[env_idx]['log_probs'])
            all_advantages.extend(advantages)
            all_entropies.extend(env_data[env_idx]['entropies'])
            all_td_errors.extend(env_data[env_idx]['td_errors'])
        
        all_log_probs = torch.stack(all_log_probs)
        all_advantages = torch.stack(all_advantages)
        all_entropies = torch.stack(all_entropies)
        all_td_errors = torch.stack(all_td_errors)
        
        # Critic loss: MSE on TD errors
        critic_loss = (all_td_errors ** 2)

        # Actor loss
        actor_loss = -(all_log_probs * all_advantages.detach())

        total_loss = 1.0 * actor_loss + 0.5 * critic_loss - 0.01 * all_entropies

        # Record losses for visualization (per environment, per step)
        if self.save_visualization_data:
            self.on_visualization_batch_losses(actor_loss, critic_loss, all_entropies, total_loss,
                                         self.batch_size, self.steps_per_epoch)

        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy = all_entropies.mean()
        total_loss = total_loss.mean()

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.wins += sum([env.wins for env in self.envs])
        
        # End visualization for this epoch
        if self.save_visualization_data:
            self.on_visualization_epoch_end()
        
        # Return training metrics
        avg_steps = self.total_steps / sum([env.wins + 1 for env in self.envs]) if self.total_steps > 0 else 0
        
        return {
            'Loss/total': total_loss.item(),
            'Loss/actor': actor_loss.item(),
            'Loss/critic': critic_loss.item(),
            'Loss/entropy_bonus': entropy.item(),
            'Performance/total_wins': self.wins,
            'Performance/avg_steps': avg_steps,
        }
    
    def validate(self, epoch):
        """Run validation games."""
        self.model.eval()
        
        val_scores = []
        val_wins = []
        for _ in range(self.validation_games):
            score, won, steps = run_validation_game(self.agent, self.layout_name, with_graphics=False)
            val_scores.append(score)
            val_wins.append(1 if won else 0)
        
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
            score, won, steps = run_validation_game(self.agent, self.layout_name, with_graphics=True)
            
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
    parser = argparse.ArgumentParser(description='Policy Gradient (Actor-Critic) Pacman Training')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Number of parallel environments')
    parser.add_argument('--steps-per-epoch', type=int, default=20, help='Steps per environment per epoch')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Layout name')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.9, help='GAE lambda parameter')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--memory-context', type=int, default=5, help='Number of past positions to remember')
    parser.add_argument('--show-epochs', type=int, default=1, 
                       help='Render validation game every N epochs (0 to disable)')
    parser.add_argument('--validation-games', type=int, default=8, 
                       help='Number of validation games per epoch')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use-best', action='store_true',
                       help='Load best checkpoint instead of last', default=False)
    parser.add_argument('--save-visualization-data', action='store_true',
                       help='Save training data for visualization', default=False)
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run a validation game without training', default=False)
    
    args = parser.parse_args()
    
    # Validation-only mode: load checkpoint and run a single validation game
    if args.validate_only:
        if not args.resume:
            print("Error: --validate-only requires --resume to specify a checkpoint")
            return
        
        print(f"Running validation game from checkpoint: {args.resume}")
        
        # Determine checkpoint path
        checkpoint_name = 'model_best.pth' if args.use_best else 'model_last.pth'
        checkpoint_path = os.path.join(args.resume, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return
        
        # Load checkpoint directly without creating trainer
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Create model and load weights
        model = ActorCriticNetwork(memory_context=args.memory_context)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create agent
        agent = RLAgent(model, memory_context=args.memory_context)
        
        # Run a single validation game with graphics
        print("Starting validation game with graphics...")
        score, won, steps = run_validation_game(agent, args.layout, with_graphics=True)
        
        print(f"\nValidation Result:")
        print(f"  Score: {score}")
        print(f"  Result: {'WON!' if won else 'Lost'}")
        print(f"  Steps: {steps}")
        
        return
    
    # Normal training mode
    trainer = PolicyGradientTrainer(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        layout_name=args.layout,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        memory_context=args.memory_context,
        show_epochs=args.show_epochs,
        validation_games=args.validation_games,
        resume_from=args.resume,
        use_best_checkpoint=args.use_best,
        save_visualization_data=args.save_visualization_data
    )
    
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
