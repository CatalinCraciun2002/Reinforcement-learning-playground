import torch
import numpy as np
import sys
import os

# Add parent directory to path to import game module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.game import Agent, Directions
from core.image_util import game_state_to_image

class LLMRLAgent(Agent):
    """
    RL Agent that uses a Vision-Language Model.
    Converts PacMan GameState into RGB images and text prompts.
    """
    def __init__(self, model, show_grid=False):
        self.model = model
        self.processor = model.processor
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.show_grid = show_grid
        self.shown_grid = False
        self.history = {} # env_id -> list of dicts: {'img': img, 'score': float, 'action': str}
        
    def _build_inputs(self, state, env_id=0):
        """Build (text, image_list) for a single state including history using a storyboard."""
        from PIL import Image
        history = self.history.get(env_id, [])
        
        prompt = "<|image|>\nYou are seeing a storyboard of the last few frames from left to right.\n"
        all_imgs = []
        
        current_score = state.getScore()
        
        # Process history
        for i, h in enumerate(history):
            all_imgs.append(h['img'])
            
            if h['action'] is not None:
                # The score diff is the difference between the next state's score and this state's score.
                if i + 1 < len(history):
                    next_score = history[i+1]['score']
                else:
                    next_score = current_score
                    
                score_diff = next_score - h['score']
                prompt += f"Frame {i+1}: You took action {h['action']} resulting in score change {score_diff}.\n"
                
        # Current state
        img = game_state_to_image(state, block_size=10)
        all_imgs.append(img)
        
        prompt += (
            f"Frame {len(all_imgs)} (Current): You are PacMan. Ghosts=red, Food=gray, Walls=brown. "
            f"Score: {current_score}. Choose: North, South, East, West, Stop."
        )
        
        # Combine images into a single storyboard (horizontally)
        w, h_img = all_imgs[0].size
        storyboard = Image.new('RGB', (w * len(all_imgs), h_img))
        for i, frame in enumerate(all_imgs):
            storyboard.paste(frame, (w * i, 0))
            
        # Return a list containing the single storyboard image
        return prompt, [storyboard]

    def state_to_inputs(self, state, env_id=0):
        """
        Converts a single state to processor inputs.
        Used by the trainer for storing re-evaluatable states.
        """
        text, images = self._build_inputs(state, env_id)
        
        if self.show_grid and not self.shown_grid:
            images[-1].show()
            self.shown_grid = True
            
        inputs = self.processor(text=[text], images=[images], return_tensors="pt")
        return inputs
        
    def registerInitialState(self, state, env_id=0):
        """Reset the history for this environment."""
        self.history[env_id] = []

    def getAction(self, legal_actions, action_probs, env_id=0):
        """
        Mask illegal actions and sample from remaining probabilities.
        """
        mask = torch.tensor([1.0 if a in legal_actions else 0.0 for a in self.actions])
        
        # Ensure action_probs is on CPU for multinomial
        if action_probs.is_cuda:
            action_probs = action_probs.cpu()
            
        masked = action_probs * mask
        masked = masked / masked.sum() if masked.sum() > 0 else mask / mask.sum()
        action_idx = torch.multinomial(masked, 1).item()

        action = self.actions[action_idx]
        
        if env_id in self.history and len(self.history[env_id]) > 0:
            self.history[env_id][-1]['action'] = action
            
        return action, action_idx

    def forward(self, state, env_id=0):
        """Forward pass for a single state."""
        inputs = self.state_to_inputs(state, env_id)
        probs, value = self.model(
            return_both=True,
            **inputs
        )
        return probs.squeeze(0), value.squeeze(0)
    
    def forward_batch(self, states, env_ids):
        """Batched forward pass for multiple states."""
        texts = []
        batch_images = []
        
        for state, env_id in zip(states, env_ids):
            text, images = self._build_inputs(state, env_id)
            
            if self.show_grid and not self.shown_grid:
                images[-1].show()
                self.shown_grid = True
                
            texts.append(text)
            batch_images.append(images)
            
            # Update history with current step
            if env_id not in self.history:
                self.history[env_id] = []
                
            self.history[env_id].append({
                'img': images[-1],
                'score': state.getScore(),
                'action': None
            })
            
            # Keep max 4 past items (so next time we have 4 past + 1 current = 5 total)
            if len(self.history[env_id]) > 4:
                self.history[env_id].pop(0)
            
        inputs = self.processor(text=texts, images=batch_images, return_tensors="pt", padding=True)
        
        probs_batch, values_batch = self.model(
            return_both=True,
            **inputs
        )
        values_batch = values_batch.squeeze(-1)
        
        # We also need to return the batched inputs so the trainer can store them
        return probs_batch, values_batch, inputs
