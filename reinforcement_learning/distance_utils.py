
from core.game import Actions

def closestTarget(pos, target, walls, is_grid=True):
    """
    BFS to find distance to closest target.
    """
    path = findPath(pos, target, walls, is_grid)
    return len(path) - 1 if path else None

def findPath(pos, target, walls, is_grid=True):
    """
    BFS to find the shortest path (list of nodes) to the closest target.
    """
    # pos might be float, convert to nearest grid point
    start_x, start_y = int(pos[0] + 0.5), int(pos[1] + 0.5)
    fringe = [((start_x, start_y), [])]
    expanded = set()
    
    while fringe:
        (pos_x, pos_y), path = fringe.pop(0)
        
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        
        current_path = path + [(pos_x, pos_y)]
        
        # Check if we hit a target
        if is_grid:
            if target[pos_x][pos_y]: return current_path
        else:
            # Ghost positions are floats usually, check proximity
            for tx, ty in target:
                if (pos_x, pos_y) == (int(tx+0.5), int(ty+0.5)):
                    return current_path
            
        # Spread out
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append(((nbr_x, nbr_y), current_path))
            
    return []

def findAllPaths(pos, targets, walls):
    """
    Finds paths to EACH of the targets provided in 'targets' list.
    """
    paths = []
    for target_pos in targets:
        # Wrap target in a list to use findPath effectively
        path = findPath(pos, [target_pos], walls, is_grid=False)
        if path:
            paths.append(path)
    return paths

def get_path_to_food(state):
    return findPath(state.getPacmanPosition(), state.getFood(), state.getWalls())

def get_path_to_capsules(state):
    return findPath(state.getPacmanPosition(), state.getCapsules(), state.getWalls(), is_grid=False)

def get_paths_to_all_active_ghosts(state):
    active_ghosts = [g.getPosition() for g in state.getGhostStates() if g.scaredTimer == 0]
    return findAllPaths(state.getPacmanPosition(), active_ghosts, state.getWalls())

def get_paths_to_all_scared_ghosts(state):
    scared_ghosts = [g.getPosition() for g in state.getGhostStates() if g.scaredTimer > 0]
    return findAllPaths(state.getPacmanPosition(), scared_ghosts, state.getWalls())

def get_path_to_scared_ghost(state):
    # Keep for compatibility, return nearest
    scared_ghosts = [g.getPosition() for g in state.getGhostStates() if g.scaredTimer > 0]
    if not scared_ghosts: return []
    return findPath(state.getPacmanPosition(), scared_ghosts, state.getWalls(), is_grid=False)

def get_path_to_ghost(state):
    # Keep for compatibility, return nearest
    active_ghosts = [g.getPosition() for g in state.getGhostStates() if g.scaredTimer == 0]
    if not active_ghosts: return []
    return findPath(state.getPacmanPosition(), active_ghosts, state.getWalls(), is_grid=False)

def get_distance_to_food(state):
    return closestTarget(state.getPacmanPosition(), state.getFood(), state.getWalls())

def get_distance_to_scared_ghost(state):
    scared_ghosts = [g.getPosition() for g in state.getGhostStates() if g.scaredTimer > 0]
    if not scared_ghosts: return None
    return closestTarget(state.getPacmanPosition(), scared_ghosts, state.getWalls(), is_grid=False)

def get_distance_to_ghost(state):
    active_ghosts = [g.getPosition() for g in state.getGhostStates() if g.scaredTimer == 0]
    if not active_ghosts: return None
    return closestTarget(state.getPacmanPosition(), active_ghosts, state.getWalls(), is_grid=False)
