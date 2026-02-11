
import display.graphicsUtils as graphicsUtils

class InGameTrajectoryVisualizer:
    """
    Visualizes trajectories directly on the game grid using dots.
    Supports multiple paths (e.g., all ghosts) and priorities.
    """
    def __init__(self, display):
        self.display = display
        self.gridSize = display.gridSize
        self.dots = {} # (x, y) -> dot_id

    def to_screen(self, pos):
        (x, y) = pos
        screen_x = (x + 1) * self.gridSize
        screen_y = (self.display.height - y) * self.gridSize
        return (screen_x, screen_y)

    def update(self, ghost_paths, capsule_path, food_path, scared_paths, state):
        canvas = graphicsUtils._canvas
        if not canvas: return
        
        # Clear old dots
        for dot_id in self.dots.values():
            canvas.delete(dot_id)
        self.dots = {}

        # Priority application (Reverse order of priority for stacking)
        
        # Priority 4: Scared Ghosts (Green)
        for path in scared_paths:
            for pos in path:
                self._add_dot(pos, '#00FF00', alpha=0.3)
            
        # Priority 3: Food (Translucent Blue)
        for pos in food_path:
            self._add_dot(pos, '#0000FF', alpha=0.3)

        # Priority 2: Capsules / Power Ups (Yellow)
        for pos in capsule_path:
            self._add_dot(pos, '#FFFF00', alpha=0.6) # Slightly more opaque than food
            
        # Priority 1: Active Ghosts (Red)
        has_food = state.getFood()
        for path in ghost_paths:
            for pos in path:
                x, y = pos
                if has_food[x][y]:
                    self._add_dot(pos, '#FF0000', alpha=1.0) # Opaque Red
                else:
                    self._add_dot(pos, '#FF0000', alpha=0.4) # Translucent Red

    def _add_dot(self, pos, color, alpha=1.0):
        canvas = graphicsUtils._canvas
        if not canvas: return
        
        screen_pos = self.to_screen(pos)
        r = 3 # Dot radius
        
        stipple = ''
        if alpha < 0.4:
            stipple = 'gray25'
        elif alpha < 1.0:
            stipple = 'gray50'
            
        # Always overwrite old dot if it's the same cell (priority is handled by order)
        if pos in self.dots:
            canvas.delete(self.dots[pos])
            
        dot = canvas.create_oval(
            screen_pos[0]-r, screen_pos[1]-r, 
            screen_pos[0]+r, screen_pos[1]+r,
            fill=color, outline='', stipple=stipple
        )
        canvas.tag_raise(dot)
        self.dots[pos] = dot

    def reset(self):
        canvas = graphicsUtils._canvas
        if not canvas: return
        for dot_id in self.dots.values():
            canvas.delete(dot_id)
        self.dots = {}
