import torch
import torch.optim as optim
import numpy as np
import sys
import os
import argparse
from torch.nn.utils.rnn import pad_sequence

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from trainers.base_trainer import BaseTrainer
from models.policy_gradient_models.llm_actor_critic import LLMActorCriticNetwork
from agents.policy_gradient_agents.llmRlAgent import LLMRLAgent
from core.game_orchestrator import GameOrchestrator

class LLMPolicyGradientTrainer(BaseTrainer):
    """PPO Trainer for LLM Backbone."""
    
    def __init__(self, args):
        super().__init__(
            training_type='llm_policy_gradient',
            args=args
        )
        
        self.show_grid = getattr(args, 'show_grid', False)
        
        self.agent = None
        self.orchestrator = None
        self.wins = 0
        self.total_steps = 0
    
    def create_model(self):
        """Create LLM Actor-Critic network."""
        # Load in 4-bit, backbone frozen
        return LLMActorCriticNetwork(load_in_4bit=True, freeze_backbone=True)
    
    def create_optimizer(self, model):
        """Create Adam optimizer ONLY for the trainable heads."""
        # Filter parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"[Trainer] Optimizing {len(trainable_params)} parameter tensors.")
        return optim.Adam(trainable_params, lr=self.lr)
    
    def post_setup(self):
        """Setup agent and orchestrator after model/optimizer creation."""
        self.agent = LLMRLAgent(self.model, show_grid=self.show_grid)
        
        self.orchestrator = GameOrchestrator(
            agent=self.agent,
            batch_size=self.batch_size,
            train_suite_name=self.train_suite,
            test_suite_name=self.test_suite,
        )
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.orchestrator.set_epoch(epoch)
        
        env_data = [
            {
                'model_inputs': {},
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
        
        states = self.orchestrator.get_all_states()
        legal_actions = self.orchestrator.get_all_legal(states)
        env_ids = list(range(self.batch_size))
        
        # Initial pass
        probs_batch, values_batch, inputs_batch = self.agent.forward_batch(states, env_ids)
        
        for step_idx in range(self.steps_per_epoch):
            next_states = []
            rewards = []
            dones = []
            actions_taken = []
            action_indices = []
            
            # Extract inputs per environment for storage
            for env_idx in range(self.batch_size):
                for k, v in inputs_batch.items():
                    if k not in env_data[env_idx]['model_inputs']:
                        env_data[env_idx]['model_inputs'][k] = []
                    env_data[env_idx]['model_inputs'][k].append(v[env_idx].detach().cpu())
            
            for env_idx, (probs, value, legal) in enumerate(
                zip(probs_batch, values_batch, legal_actions)
            ):
                action, action_idx = self.agent.getAction(legal, probs, env_id=env_idx)
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
            
            # Compute old log probs
            action_indices_tensor = torch.tensor(action_indices, dtype=torch.long)
            old_log_probs = self.model.last_log_probs[
                torch.arange(self.batch_size), action_indices_tensor
            ].detach()
            
            for env_idx in range(self.batch_size):
                env_data[env_idx]['action_indices'].append(action_indices[env_idx])
                env_data[env_idx]['old_log_probs'].append(old_log_probs[env_idx])
                env_data[env_idx]['old_values'].append(values_batch[env_idx].detach())
                env_data[env_idx]['rewards'].append(rewards[env_idx])
            
            # Next pass
            next_probs_batch, next_values_batch, next_inputs_batch = self.agent.forward_batch(next_states, env_ids)
            
            # TD errors
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.model.backbone.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.model.backbone.device)
            td_errors = rewards_tensor + self.gamma * next_values_batch.detach() * (1 - dones_tensor) - values_batch.detach()
            
            for env_idx in range(self.batch_size):
                env_data[env_idx]['td_errors'].append(td_errors[env_idx])
                env_data[env_idx]['game_overs'].append(dones[env_idx])
            
            states = next_states
            legal_actions = self.orchestrator.get_all_legal(states)
            probs_batch = next_probs_batch
            values_batch = next_values_batch
            inputs_batch = next_inputs_batch
        
        # GAE
        all_model_inputs = {}
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
            
            for k, list_of_tensors in env_data[env_idx]['model_inputs'].items():
                if k not in all_model_inputs:
                    all_model_inputs[k] = []
                all_model_inputs[k].extend(list_of_tensors)
                
            all_action_indices.extend(env_data[env_idx]['action_indices'])
            all_old_log_probs.extend(env_data[env_idx]['old_log_probs'])
            all_old_values.extend(env_data[env_idx]['old_values'])
            all_advantages.extend(advantages)
        
        # Pad input sequences since they might have different lengths due to dynamic score insertion
        pad_id = self.model.processor.tokenizer.pad_token_id
        for k, list_of_tensors in all_model_inputs.items():
            if list_of_tensors and list_of_tensors[0].dim() == 1:
                # 1D tensors (e.g. input_ids) need sequence padding
                pad_val = pad_id if k == 'input_ids' else 0
                all_model_inputs[k] = pad_sequence(list_of_tensors, batch_first=True, padding_value=pad_val)
            else:
                # N-D tensors (e.g. pixel_values, image_position_ids) can just be stacked (fixed sizes)
                all_model_inputs[k] = torch.stack(list_of_tensors)
            
        all_action_indices = torch.tensor(all_action_indices, dtype=torch.long)
        all_old_log_probs = torch.stack(all_old_log_probs)
        all_old_values = torch.stack(all_old_values)
        all_advantages = torch.stack(all_advantages)
        
        all_returns = all_advantages + all_old_values
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # PPO Update
        n_samples = all_model_inputs['input_ids'].size(0)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_loss_val = 0.0
        num_updates = 0

        self.model.train()
        
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n_samples)
            
            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                mb_idx = indices[start:end]
                
                mb_inputs = {k: v[mb_idx] for k, v in all_model_inputs.items()}
                
                # Recreate attention mask based on padding for input_ids
                if 'attention_mask' in mb_inputs:
                    mb_inputs['attention_mask'] = (mb_inputs['input_ids'] != pad_id).long()
                
                mb_actions = all_action_indices[mb_idx]
                
                mb_old_log_probs = all_old_log_probs[mb_idx].to(self.model.backbone.device)
                mb_advantages = all_advantages[mb_idx].to(self.model.backbone.device)
                mb_returns = all_returns[mb_idx].to(self.model.backbone.device)
                
                # Forward
                new_probs, new_values = self.model(
                    return_both=True,
                    **mb_inputs
                )
                new_values = new_values.squeeze(-1)
                
                new_log_probs = self.model.last_log_probs[
                    torch.arange(len(mb_idx)), mb_actions
                ]

                # PPO Loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                ratio = torch.clamp(ratio, 1e-3, 10.0)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = ((new_values - mb_returns) ** 2).mean()
                entropy = -(new_probs * self.model.last_log_probs).sum(dim=1).mean()
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients for the heads
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 0.5)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_loss_val += loss.item()
                num_updates += 1

        self.wins += self.orchestrator.total_wins()
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
        self.model.eval()
        results = self.orchestrator.run_validation(n_games=self.validation_games, with_graphics=False)
        val_scores = [r[0] for r in results]
        val_wins = [1 if r[1] else 0 for r in results]
        self.model.train()
        return {
            'Score/score': sum(val_scores) / len(val_scores),
            'Score/won': sum(val_wins) / len(val_wins)
        }
    
    def get_metric_for_checkpoint(self, val_metrics):
        return val_metrics['Score/score'], 'avg_score'
    
    def get_progress_bar_dict(self, train_metrics, val_metrics):
        return {
            'Actor': f"{train_metrics.get('Loss/actor', 0):.4f}",
            'Critic': f"{train_metrics.get('Loss/critic', 0):.2f}",
            'ValScore': f"{val_metrics.get('Score/score', 0):.1f}"
        }
    
    def on_epoch_end(self, epoch, pbar):
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
        end_epoch = self.start_epoch + self.num_epochs
        final_win_rate = f"{self.wins}/{end_epoch * self.batch_size} = {self.wins/(end_epoch * self.batch_size) if (end_epoch * self.batch_size) > 0 else 0:.2%}"
        return {'Final Win Rate': final_win_rate, 'Total Steps': str(self.total_steps)}
    
    def get_additional_checkpoint_data(self):
        return {'wins': self.wins, 'total_steps': self.total_steps, 'val_avg_score': self.best_metric}

def main():
    parser = LLMPolicyGradientTrainer.build_parser()
    parser.add_argument('--show-grid', action='store_true', default=False, help='Show the image representation once')
    
    # Hardcoded defaults for parameters frequently changed by the user
    parser.set_defaults(
        num_epochs=100,
        batch_size=2,          # Keep batch size small for LLM VRAM limits
        steps_per_epoch=16,
        mini_batch_size=2,
        show_epochs=1,         # Show graphics every epoch
        resume=None            # Hardcode a path here if you always want to resume (e.g., 'runs/policy_gradient/model_last.pth')
    )
    
    args = parser.parse_args()
    
    trainer = LLMPolicyGradientTrainer(args)
    
    trainer.setup()
    trainer.train()

if __name__ == '__main__':
    main()
