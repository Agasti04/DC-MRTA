# navigation.py

import heapq

def a_star_navigation(start, goal, obstacles):
    """
    A simple A* pathfinding algorithm on a 2D grid with rectangular obstacles.
    - start, goal: (x, y) tuples
    - obstacles: list of (ox, oy, width, height) rectangles
    """
    def in_obstacle(cell):
        x, y = cell
        for ox, oy, w, h in obstacles:
            if ox <= x < ox + w and oy <= y < oy + h:
                return True
        return False

    def heuristic(a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if in_obstacle(neighbor):
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
    return []  # no path found

def orca_collision_avoidance(robots, robot_positions, time_step):
    """
    ORCA stub: shift each robot one unit forward in x.
    Real ORCA would consider both x,y velocities and obstacles.
    """
    new_positions = {}
    for r in robots:
        x, y = robot_positions[r]
        new_positions[r] = (x + 1, y)
    return new_positions
