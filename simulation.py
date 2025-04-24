# simulation.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from task_allocation import greedy_task_allocation
from navigation import a_star_navigation, orca_collision_avoidance
from utils import initialize_environment

def run_simulation(n_robots, n_tasks, grid=150, obs_count=20):
    agents, tasks, obstacles = initialize_environment(grid, obs_count, n_tasks, n_robots)
    # filter out tasks that land inside an obstacle
    tasks = [t for t in tasks
             if not any(ox <= t[0] < ox+w and oy <= t[1] < oy+h
                        for ox, oy, w, h in obstacles)]
    if len(tasks) < n_tasks:
        raise ValueError("Too many obstacles; fewer valid task spots.")

    robot_pos = {f"r{i}": agents[i] for i in range(n_robots)}
    assignment = greedy_task_allocation(robot_pos, tasks)
    print("Assignment:", assignment)

    robots = list(robot_pos.keys())
    paths = {r: [] for r in robots}

    fig, ax = plt.subplots()
    ax.set_xlim(0, grid)
    ax.set_ylim(0, grid)
    ax.set_aspect('equal')

    def update(frame):
        ax.clear()
        ax.set_xlim(0, grid)
        ax.set_ylim(0, grid)
        ax.set_aspect('equal')

        # draw obstacles
        for ox, oy, w, h in obstacles:
            ax.add_patch(plt.Rectangle((ox, oy), w, h, color='gray', alpha=0.6))

        done = True
        for r in robots:
            goal = tasks[assignment[r]]
            path = a_star_navigation(robot_pos[r], goal, obstacles)
            if path:
                step = min(frame, len(path)-1)
                robot_pos[r] = path[step]
                paths[r].append(robot_pos[r])
                xs, ys = zip(*paths[r])
                ax.plot(xs, ys, 'b-')
                if robot_pos[r] != goal:
                    done = False
            ax.plot(*robot_pos[r], 'go')

        newp = orca_collision_avoidance(robots, robot_pos, time_step=0.25)
        for r in robots:
            x, y = newp[r]
            hit   = any(ox <= x < ox+w and oy <= y < oy+h for ox, oy, w, h in obstacles)
            clash = any((x,y) == robot_pos[o] and o != r for o in robots)
            if hit or clash:
                newp[r] = agents[int(r[1:])]  # reset to origin
        robot_pos.update(newp)

        if done:
            plt.close(fig)

    anim = animation.FuncAnimation(fig, update, frames=100, interval=200, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_simulation(5, 5)
