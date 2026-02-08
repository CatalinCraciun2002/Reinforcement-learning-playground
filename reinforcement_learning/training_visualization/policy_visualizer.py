"""
Interactive Policy Visualizer
Displays training data step-by-step with game visualization and detailed metrics
"""

import sys
import os
import argparse
import pygame
from pygame.locals import *

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning.epoch_visualizer import load_visualization_data
from display import graphicsUtils
from core.game import Directions


# Colors
BACKGROUND_COLOR = (20, 20, 30)
PANEL_COLOR = (30, 30, 45)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (100, 150, 255)
POSITIVE_COLOR = (50, 200, 50)
NEGATIVE_COLOR = (255, 50, 50)
NEUTRAL_COLOR = (150, 150, 150)


class ControlPanel:
    """Left control panel for selecting epoch/environment/step and viewing metrics"""
    
    def __init__(self, width, height, data):
        """
        Args:
            width: Width of the control panel
            height: Height of the control panel
            data: Loaded visualization data
        """
        self.width = width
        self.height = height
        self.data = data
        self.metadata = data['metadata']
        self.epochs = data['epochs']
        
        # Current selection
        self.current_epoch = 0
        self.current_env = 0
        self.current_step = 0
        
        # Playback
        self.playing = False
        self.playback_speed = 0.5  # seconds per step
        self.last_step_time = 0
        
        # Font
        pygame.font.init()
        self.font_large = pygame.font.SysFont('Courier New', 18, bold=True)
        self.font_medium = pygame.font.SysFont('Courier New', 14)
        self.font_small = pygame.font.SysFont('Courier New', 12)
        
        # Limits
        self.max_epoch = max(self.epochs.keys())
        self.max_env = len(self.epochs[self.current_epoch]['environments']) - 1
        self.max_step = len(self.epochs[self.current_epoch]['environments'][self.current_env]['steps']) - 1
    
    def update_limits(self):
        """Update limits based on current selections"""
        self.max_env = len(self.epochs[self.current_epoch]['environments']) - 1
        self.current_env = min(self.current_env, self.max_env)
        
        self.max_step = len(self.epochs[self.current_epoch]['environments'][self.current_env]['steps']) - 1
        self.current_step = min(self.current_step, self.max_step)
    
    def get_current_step_data(self):
        """Get the currently selected step's data"""
        return self.epochs[self.current_epoch]['environments'][self.current_env]['steps'][self.current_step]
    
    def get_current_losses(self):
        """Get losses for the current epoch"""
        return self.epochs[self.current_epoch]['losses']
    
    def handle_key(self, key):
        """Handle keyboard input"""
        if key == K_SPACE:
            self.playing = not self.playing
        elif key == K_RIGHT:
            self.next_step()
        elif key == K_LEFT:
            self.prev_step()
        elif key == K_UP:
            self.next_env()
        elif key == K_DOWN:
            self.prev_env()
        elif key == K_PAGEUP:
            self.next_epoch()
        elif key == K_PAGEDOWN:
            self.prev_epoch()
        elif key == K_HOME:
            self.current_step = 0
        elif key == K_END:
            self.current_step = self.max_step
    
    def next_step(self):
        if self.current_step < self.max_step:
            self.current_step += 1
        elif self.current_env < self.max_env:
            self.current_env += 1
            self.update_limits()
            self.current_step = 0
    
    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
        elif self.current_env > 0:
            self.current_env -= 1
            self.update_limits()
            self.current_step = self.max_step
    
    def next_env(self):
        if self.current_env < self.max_env:
            self.current_env += 1
            self.update_limits()
    
    def prev_env(self):
        if self.current_env > 0:
            self.current_env -= 1
            self.update_limits()
    
    def next_epoch(self):
        if self.current_epoch < self.max_epoch:
            self.current_epoch += 1
            self.update_limits()
    
    def prev_epoch(self):
        if self.current_epoch > 0:
            self.current_epoch -= 1
            self.update_limits()
    
    def update(self):
        """Update playback if playing"""
        import time
        if self.playing:
            current_time = time.time()
            if current_time - self.last_step_time > self.playback_speed:
                self.next_step()
                self.last_step_time = current_time
                if self.current_step == self.max_step and self.current_env == self.max_env:
                    self.playing = False  # Stop at end
    
    def draw(self, screen):
        """Draw the control panel"""
        # Background
        pygame.draw.rect(screen, PANEL_COLOR, (0, 0, self.width, self.height))
        
        y = 20
        x_margin = 15
        line_height = 25
        
        # Title
        title = self.font_large.render("Training Visualizer", True, HIGHLIGHT_COLOR)
        screen.blit(title, (x_margin, y))
        y += line_height * 2
        
        # Navigation section
        self._draw_section_header(screen, "Navigation", x_margin, y)
        y += line_height
        
        nav_items = [
            f"Epoch: {self.current_epoch}/{self.max_epoch}",
            f"Env: {self.current_env}/{self.max_env}",
            f"Step: {self.current_step}/{self.max_step}"
        ]
        for item in nav_items:
            text = self.font_medium.render(item, True, TEXT_COLOR)
            screen.blit(text, (x_margin + 10, y))
            y += line_height
        
        y += 10
        
        # Step data section
        self._draw_section_header(screen, "Step Data", x_margin, y)
        y += line_height
        
        step_data = self.get_current_step_data()
        
        # Actor info
        y = self._draw_subsection(screen, "Actor:", x_margin, y, line_height)
        
        action = step_data['selected_action']
        y = self._draw_text(screen, f"  Action: {action}", x_margin, y, HIGHLIGHT_COLOR)
        
        # Show all action probabilities
        y = self._draw_text(screen, "  Probabilities:", x_margin, y)
        for act, prob in sorted(step_data['action_probs'].items()):
            color = HIGHLIGHT_COLOR if act == action else TEXT_COLOR
            prob_text = f"    {act}: {prob:.3f}"
            y = self._draw_text(screen, prob_text, x_margin, y, color, self.font_small)
        
        y += 5
        
        # Critic info
        y = self._draw_subsection(screen, "Critic:", x_margin, y, line_height)
        
        critic_items = [
            (f"  Value: {step_data['value']:.2f}", TEXT_COLOR),
            (f"  Next Val: {step_data['next_value']:.2f}", TEXT_COLOR),
        ]
        
        if step_data['td_target'] is not None:
            critic_items.append((f"  TD Target: {step_data['td_target']:.2f}", TEXT_COLOR))
        
        td_error = step_data['td_error']
        td_color = POSITIVE_COLOR if td_error > 0 else NEGATIVE_COLOR if td_error < 0 else NEUTRAL_COLOR
        critic_items.append((f"  TD Error: {td_error:.3f}", td_color))
        
        advantage = step_data['advantage']
        if advantage is not None:
            adv_color = POSITIVE_COLOR if advantage > 0 else NEGATIVE_COLOR if advantage < 0 else NEUTRAL_COLOR
            critic_items.append((f"  Advantage: {advantage:.3f}", adv_color))
        
        for text, color in critic_items:
            y = self._draw_text(screen, text, x_margin, y, color)
        
        y += 5
        
        # Step info
        y = self._draw_subsection(screen, "Step Info:", x_margin, y, line_height)
        
        reward = step_data['reward']
        reward_color = POSITIVE_COLOR if reward > 0 else NEGATIVE_COLOR if reward < 0 else NEUTRAL_COLOR
        
        step_items = [
            (f"  Reward: {reward:.2f}", reward_color),
            (f"  Done: {step_data['done']}", TEXT_COLOR),
        ]
        
        for text, color in step_items:
            y = self._draw_text(screen, text, x_margin, y, color)
        
        y += 10
        
        # Epoch losses
        y = self._draw_section_header(screen, "Epoch Losses", x_margin, y)
        y += line_height
        
        losses = self.get_current_losses()
        if losses:
            loss_items = [
                f"  Actor: {losses['actor']:.4f}",
                f"  Critic: {losses['critic']:.4f}",
                f"  Entropy: {losses['entropy_bonus']:.4f}",
                f"  Total: {losses['total']:.4f}",
            ]
            for item in loss_items:
                text = self.font_small.render(item, True, TEXT_COLOR)
                screen.blit(text, (x_margin + 10, y))
                y += 20
        
        y += 10
        
        # Controls help
        y = self._draw_section_header(screen, "Controls", x_margin, y)
        y += line_height
        
        controls = [
            "SPACE: Play/Pause",
            "←/→: Prev/Next Step",
            "↑/↓: Prev/Next Env",
            "PgUp/Dn: Prev/Next Epoch",
            "HOME/END: First/Last Step",
            "ESC: Quit"
        ]
        
        for control in controls:
            text = self.font_small.render(control, True, NEUTRAL_COLOR)
            screen.blit(text, (x_margin + 5, y))
            y += 18
        
        # Playback indicator
        if self.playing:
            y += 10
            playing_text = self.font_medium.render("▶ PLAYING", True, POSITIVE_COLOR)
            screen.blit(playing_text, (x_margin, y))
    
    def _draw_section_header(self, screen, text, x, y):
        """Draw a section header"""
        rendered = self.font_large.render(text, True, HIGHLIGHT_COLOR)
        screen.blit(rendered, (x, y))
        # Underline
        pygame.draw.line(screen, HIGHLIGHT_COLOR, (x, y + 22), (x + self.width - 30, y + 22), 1)
        return y
    
    def _draw_subsection(self, screen, text, x, y, line_height):
        """Draw a subsection"""
        rendered = self.font_medium.render(text, True, TEXT_COLOR)
        screen.blit(rendered, (x, y))
        return y + line_height
    
    def _draw_text(self, screen, text, x, y, color=None, font=None):
        """Draw a text line"""
        if color is None:
            color = TEXT_COLOR
        if font is None:
            font = self.font_medium
        rendered = font.render(text, True, color)
        screen.blit(rendered, (x, y))
        return y + 22


class GameRenderer:
    """Renders the Pac-Man game state using the existing graphics system"""
    
    def __init__(self, x_offset, y_offset, screen_width, screen_height):
        """
        Args:
            x_offset: X offset from left edge of screen
            y_offset: Y offset from top edge
            screen_width: Available width for game rendering
            screen_height: Available height for game rendering
        """
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Import graphicsUtils components
        from display.graphicsUtils import formatColor
        
        # Colors matching the original game
        self.BACKGROUND_COLOR = (0, 0, 0)
        self.WALL_COLOR = (0, 51, 255)
        self.FOOD_COLOR = (255, 255, 255)
        self.CAPSULE_COLOR = (255, 255, 255)
        self.PACMAN_COLOR = (255, 255, 61)
        self.GHOST_COLORS = [
            (230, 0, 0),      # Red
            (0, 76, 230),     # Blue
            (250, 105, 18),   # Orange
            (26, 191, 179)    # Green
        ]
        self.SCARED_COLOR = (255, 255, 255)
        
        # Will be set when first state is loaded
        self.layout_width = None
        self.layout_height = None
        self.grid_size = None
        
        # Fonts
        self.font = pygame.font.SysFont('Courier New', 20, bold=True)
        self.small_font = pygame.font.SysFont('Courier New', 14)
    
    def draw(self, screen, step_data):
        """Draw the game state"""
        state_data = step_data['state']
        
        # Extract grid dimensions from walls
        walls = state_data['walls']
        if self.layout_width is None:
            self.layout_height = len(walls[0]) if walls and walls[0] else 0  # Note: transposed
            self.layout_width = len(walls)
            
            # Calculate grid size to fit in available space
            max_grid_width = (self.screen_width - 100) / self.layout_width
            max_grid_height = (self.screen_height - 100) / self.layout_height
            self.grid_size = int(min(max_grid_width, max_grid_height, 30))
        
        # Draw score at top
        score_text = f"Score: {state_data['score']}"
        score_surf = self.font.render(score_text, True, (255, 255, 255))
        screen.blit(score_surf, (self.x_offset, self.y_offset - 40))
        
        # Draw win/lose status
        if state_data['is_win']:
            status_surf = self.font.render("WIN!", True, POSITIVE_COLOR)
            screen.blit(status_surf, (self.x_offset + 150, self.y_offset - 40))
        elif state_data['is_lose']:
            status_surf = self.font.render("LOSE", True, NEGATIVE_COLOR)
            screen.blit(status_surf, (self.x_offset + 150, self.y_offset - 40))
        
        # Draw walls
        for x in range(len(walls)):
            for y in range(len(walls[x])):
                if walls[x][y]:
                    self._draw_wall(screen, x, y)
        
        # Draw food
        food = state_data['food']
        for x in range(len(food)):
            for y in range(len(food[x])):
                if food[x][y]:
                    self._draw_food(screen, x, y)
        
        # Draw capsules
        for capsule_pos in state_data['capsules']:
            self._draw_capsule(screen, capsule_pos[0], capsule_pos[1])
        
        # Draw ghosts
        for i, (ghost_pos, scared_timer) in enumerate(state_data['ghost_states']):
            self._draw_ghost(screen, ghost_pos[0], ghost_pos[1], scared_timer > 0, i)
        
        # Draw Pacman (on top)
        pacman_pos = state_data['pacman_pos']
        selected_action = step_data['selected_action']
        self._draw_pacman(screen, pacman_pos[0], pacman_pos[1], selected_action)
        
        # Draw action arrow indicator
        self._draw_action_indicator(screen, pacman_pos[0], pacman_pos[1], selected_action)
    
    def _grid_to_screen(self, grid_x, grid_y):
        """Convert grid coordinates to screen coordinates"""
        # Grid has (0,0) at bottom-left, screen has (0,0) at top-left
        # The to_screen formula from PacmanGraphics is:
        # x = (x + 1) * gridSize
        # y = (height - y) * gridSize
        screen_x = self.x_offset + (grid_x + 1) * self.grid_size
        screen_y = self.y_offset + (self.layout_height - grid_y) * self.grid_size
        return int(screen_x), int(screen_y)
    
    def _draw_wall(self, screen, grid_x, grid_y):
        """Draw a wall cell"""
        x, y = self._grid_to_screen(grid_x, grid_y)
        pygame.draw.rect(screen, self.WALL_COLOR, 
                        (x - self.grid_size//2, y - self.grid_size//2, 
                         self.grid_size, self.grid_size))
    
    def _draw_food(self, screen, grid_x, grid_y):
        """Draw a food pellet"""
        x, y = self._grid_to_screen(grid_x, grid_y)
        radius = max(2, int(self.grid_size * 0.1))
        pygame.draw.circle(screen, self.FOOD_COLOR, (x, y), radius)
    
    def _draw_capsule(self, screen, grid_x, grid_y):
        """Draw a power capsule"""
        x, y = self._grid_to_screen(grid_x, grid_y)
        radius = max(4, int(self.grid_size * 0.25))
        pygame.draw.circle(screen, self.CAPSULE_COLOR, (x, y), radius)
    
    def _draw_ghost(self, screen, grid_x, grid_y, is_scared, ghost_idx):
        """Draw a ghost using a simplified shape"""
        x, y = self._grid_to_screen(grid_x, grid_y)
        radius = max(8, int(self.grid_size * 0.4))
        
        color = self.SCARED_COLOR if is_scared else self.GHOST_COLORS[ghost_idx % len(self.GHOST_COLORS)]
        
        # Draw body (circle)
        pygame.draw.circle(screen, color, (x, y), radius)
        
        # Draw eyes (simple white circles with black pupils)
        eye_radius = max(2, radius // 4)
        pupil_radius = max(1, radius // 6)
        eye_offset_x = radius // 3
        eye_offset_y = -radius // 4
        
        # Left eye
        pygame.draw.circle(screen, (255, 255, 255), 
                          (x - eye_offset_x, y + eye_offset_y), eye_radius)
        pygame.draw.circle(screen, (0, 0, 0), 
                          (x - eye_offset_x, y + eye_offset_y), pupil_radius)
        
        # Right eye
        pygame.draw.circle(screen, (255, 255, 255), 
                          (x + eye_offset_x, y + eye_offset_y), eye_radius)
        pygame.draw.circle(screen, (0, 0, 0), 
                          (x + eye_offset_x, y + eye_offset_y), pupil_radius)
    
    def _draw_pacman(self, screen, grid_x, grid_y, direction):
        """Draw Pacman as a yellow circle"""
        x, y = self._grid_to_screen(grid_x, grid_y)
        radius = max(8, int(self.grid_size * 0.4))
        
        # Draw Pacman as a simple circle (simplified version)
        # Drawing the mouth correctly with angles is complex, so we keep it simple
        pygame.draw.circle(screen, self.PACMAN_COLOR, (x, y), radius)
        
        # Optionally add a small black circle for the eye
        eye_radius = max(1, radius // 8)
        if direction == Directions.EAST:
            eye_x, eye_y = x + radius//4, y - radius//3
        elif direction == Directions.WEST:
            eye_x, eye_y = x - radius//4, y - radius//3
        elif direction == Directions.NORTH:
            eye_x, eye_y = x + radius//4, y - radius//3
        elif direction == Directions.SOUTH:
            eye_x, eye_y = x + radius//4, y + radius//4
        else:  # STOP
            eye_x, eye_y = x + radius//4, y - radius//3
        
        pygame.draw.circle(screen, (0, 0, 0), (eye_x, eye_y), eye_radius)
    
    def _draw_action_indicator(self, screen, grid_x, grid_y, direction):
        """Draw an arrow indicating the selected action"""
        if direction == Directions.STOP:
            return
        
        x, y = self._grid_to_screen(grid_x, grid_y)
        arrow_length = int(self.grid_size * 0.7)
        arrow_color = HIGHLIGHT_COLOR
        
        # Calculate arrow end point
        if direction == Directions.NORTH:
            end_x, end_y = x, y - arrow_length
        elif direction == Directions.SOUTH:
            end_x, end_y = x, y + arrow_length
        elif direction == Directions.EAST:
            end_x, end_y = x + arrow_length, y
        elif direction == Directions.WEST:
            end_x, end_y = x - arrow_length, y
        else:
            return
        
        # Draw arrow line
        pygame.draw.line(screen, arrow_color, (x, y), (end_x, end_y), 3)
        
        # Draw arrowhead
        self._draw_arrow_head(screen, x, y, end_x, end_y, arrow_color)
    
    def _draw_arrow_head(self, screen, start_x, start_y, end_x, end_y, color):
        """Draw an arrow head"""
        import math
        
        # Calculate angle
        dx = end_x - start_x
        dy = end_y - start_y
        angle = math.atan2(dy, dx)
        
        # Arrow head size
        head_length = 8
        head_angle = math.pi / 6  # 30 degrees
        
        # Calculate arrow head points
        point1_x = end_x - head_length * math.cos(angle - head_angle)
        point1_y = end_y - head_length * math.sin(angle - head_angle)
        point2_x = end_x - head_length * math.cos(angle + head_angle)
        point2_y = end_y - head_length * math.sin(angle + head_angle)
        
        pygame.draw.polygon(screen, color, [
            (end_x, end_y),
            (point1_x, point1_y),
            (point2_x, point2_y)
        ])


class PolicyVisualizer:
    """Main visualizer application"""
    
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Path to visualization data directory
        """
        print(f"Loading visualization data from: {data_dir}")
        self.data = load_visualization_data(data_dir)
        print(f"Loaded {len(self.data['epochs'])} epochs")
        
        # Initialize Pygame
        pygame.init()
        
        # Setup display
        self.panel_width = 350
        self.game_width = 800
        self.screen_width = self.panel_width + self.game_width
        self.screen_height = 700
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("RL Training Visualizer")
        
        # Create components
        self.control_panel = ControlPanel(self.panel_width, self.screen_height, self.data)
        self.game_renderer = GameRenderer(
            self.panel_width + 20,  # x_offset
            80,                      # y_offset
            self.game_width - 40,    # screen_width
            self.screen_height - 100 # screen_height
        )
        
        self.clock = pygame.time.Clock()
        self.running = True
    
    def run(self):
        """Main loop"""
        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(30)  # 30 FPS
        
        pygame.quit()
    
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                else:
                    self.control_panel.handle_key(event.key)
    
    def _update(self):
        """Update state"""
        self.control_panel.update()
    
    def _draw(self):
        """Draw everything"""
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw control panel
        self.control_panel.draw(self.screen)
        
        # Draw vertical separator
        pygame.draw.line(
            self.screen, 
            HIGHLIGHT_COLOR, 
            (self.panel_width, 0), 
            (self.panel_width, self.screen_height), 
            2
        )
        
        # Draw game
        step_data = self.control_panel.get_current_step_data()
        self.game_renderer.draw(self.screen, step_data)
        
        # Update display
        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description='Visualize RL training data')
    parser.add_argument('data_dir', help='Path to visualization data directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory not found: {args.data_dir}")
        return 1
    
    visualizer = PolicyVisualizer(args.data_dir)
    visualizer.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
