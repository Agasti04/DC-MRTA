# task_allocation.py

from utils import calculate_distance

def greedy_task_allocation(robot_positions, task_positions):
    """
    Assigns each robot the closest remaining task (by Euclidean distance).
    robot_positions: dict robot_id -> (x,y)
    task_positions: list of (x,y)
    """
    assignment = {}
    remaining = set(range(len(task_positions)))
    for r, pos in robot_positions.items():
        best = min(remaining, key=lambda i: calculate_distance(pos, task_positions[i]))
        assignment[r] = best
        remaining.remove(best)
    return assignment

def nearest_task_allocation(robot_position, task_positions, tasks_done, current_assignments):
    """
    Returns the index of the nearest task that is neither done nor already assigned.
    """
    candidates = [
        i for i in range(len(task_positions))
        if i not in tasks_done and i not in current_assignments.values()
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda i: calculate_distance(robot_position, task_positions[i]))
