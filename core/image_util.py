import numpy as np
from PIL import Image

def game_state_to_image(state, block_size=10):
    """
    Converts a PacMan GameState into an RGB PIL Image.
    
    Args:
        state: GameState object
        block_size: Size of each grid cell in pixels
        
    Returns:
        PIL.Image
    """
    walls = state.getWalls()
    width = walls.width
    height = walls.height
    
    # Initialize empty grid (Black)
    # Shape: (height, width, 3)
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Draw Walls (Brown: 139, 69, 19)
    for x in range(width):
        for y in range(height):
            if walls[x][y]:
                # Note: Pacman coordinates have (0,0) at bottom-left usually, 
                # but grids are accessed as grid[y][x]. 
                # We need to map (x, y) to matrix indices.
                # Usually in graphics display, y=0 is the bottom. 
                # We will map y=0 to the bottom of the image.
                img_y = height - 1 - y
                grid[img_y, x] = [139, 69, 19]
    
    # 2. Draw Food (Gray: 128, 128, 128)
    food = state.getFood()
    for x in range(width):
        for y in range(height):
            if food[x][y]:
                img_y = height - 1 - y
                grid[img_y, x] = [128, 128, 128]
                
    # 3. Draw Capsules/Power Pellets (White: 255, 255, 255)
    for x, y in state.getCapsules():
        img_y = height - 1 - int(y)
        grid[img_y, int(x)] = [255, 255, 255]
        
    # 4. Draw Ghosts (Red: 255, 0, 0)
    for ghost_pos in state.getGhostPositions():
        x, y = ghost_pos
        img_y = height - 1 - int(y)
        grid[img_y, int(x)] = [255, 0, 0]
        
    # 5. Draw Pacman (Yellow: 255, 255, 0)
    pac_x, pac_y = state.getPacmanPosition()
    img_y = height - 1 - int(pac_y)
    grid[img_y, int(pac_x)] = [255, 255, 0]
    
    # Scale up the image
    # We use nearest neighbor interpolation to keep the blocks sharp
    img = Image.fromarray(grid, 'RGB')
    if block_size > 1:
        img = img.resize((width * block_size, height * block_size), Image.Resampling.NEAREST)
        
    return img

def display_game_state(state, block_size=20):
    """
    Displays the game state image for debugging/assessment.
    """
    img = game_state_to_image(state, block_size)
    img.show()
