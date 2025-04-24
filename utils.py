# utils.py

import random

def calculate_distance(p1, p2):
    """Euclidean distance."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def initialize_environment(grid_size, num_obstacles, num_tasks, num_agents):
    # random agent + task locations
    agents = [
        (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        for _ in range(num_agents)
    ]
    tasks = [
        (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        for _ in range(num_tasks)
    ]

    # two obstacle columns
    center = grid_size // 2
    col1_x, col2_x = center - 30, center + 30

    rows = num_obstacles // 2
    vspace = grid_size // (rows + 1)
    h = vspace // 2
    w = 3

    obstacles = []
    for i in range(rows):
        y = (i + 1) * vspace
        obstacles.append((col1_x, y, w, h))
        obstacles.append((col2_x, y, w, h))

    return agents, tasks, obstacles
