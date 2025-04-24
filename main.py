# main.py

import os
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import count

import numpy as np
from PIL import Image  # Pillow

from navigation import a_star_navigation, orca_collision_avoidance
from utils      import initialize_environment, calculate_distance

def start_simulation(num_agents, num_tasks, interval_ms):
    cwd = os.getcwd()
    print(f"Working directory: {cwd}\n")

    grid, obs_count = 150, 20

    # 1) Build environment with enough valid tasks
    while True:
        agents, tasks_raw, obstacles = initialize_environment(grid, obs_count, num_tasks, num_agents)
        valid_tasks = [
            t for t in tasks_raw
            if not any(ox <= t[0] < ox + w and oy <= t[1] < oy + h
                       for ox, oy, w, h in obstacles)
        ]
        if len(valid_tasks) >= num_tasks:
            tasks = valid_tasks[:num_tasks]
            break

    # 2) Simulation state
    robot_pos     = {f"r{i}": agents[i] for i in range(num_agents)}
    tasks_pending = set(range(len(tasks)))
    tasks_done    = set()
    robot_goal    = {r: None for r in robot_pos}
    paths         = {r: [robot_pos[r]] for r in robot_pos}

    # 3) Save initial snapshot
    fig0, ax0 = plt.subplots()
    ax0.set_xlim(0, grid); ax0.set_ylim(0, grid); ax0.set_aspect('equal')
    for ox, oy, w, h in obstacles:
        ax0.add_patch(plt.Rectangle((ox, oy), w, h, color='gray', alpha=0.8))
    for tx, ty in tasks:
        ax0.plot(tx, ty, 'x', color='red', markersize=10)
    for pos in agents:
        ax0.plot(*pos, 'o', color='blue')
    start_path = os.path.join(cwd, "start.png")
    fig0.savefig(start_path)
    plt.close(fig0)
    print(f"Saved start.png → {start_path}")

    # 4) Live animation + frame capture
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid); ax.set_ylim(0, grid); ax.set_aspect('equal')

    frames = []
    took_end = False

    def update(frame):
        nonlocal took_end

        ax.clear()
        ax.set_xlim(0, grid); ax.set_ylim(0, grid); ax.set_aspect('equal')

        # draw obstacles
        for ox, oy, w, h in obstacles:
            ax.add_patch(plt.Rectangle((ox, oy), w, h, color='gray', alpha=0.8))

        # draw tasks
        for i, (tx, ty) in enumerate(tasks):
            color = 'green' if i in tasks_done else 'red'
            ax.plot(tx, ty, 'x', color=color, markersize=10)

        # assign & move robots
        for r in robot_pos:
            if robot_goal[r] is None and tasks_pending:
                nxt = min(tasks_pending, key=lambda i: calculate_distance(robot_pos[r], tasks[i]))
                tasks_pending.remove(nxt)
                robot_goal[r] = nxt
                paths[r] = [robot_pos[r]]
                print(f"{r} assigned to task {nxt}")

            goal_idx = robot_goal[r]
            if goal_idx is not None:
                goal = tasks[goal_idx]
                path = a_star_navigation(robot_pos[r], goal, obstacles)
                if path:
                    step = min(frame, len(path)-1)
                    robot_pos[r] = path[step]
                    paths[r].append(robot_pos[r])
                    xs, ys = zip(*paths[r])
                    ax.plot(xs, ys, '-', color='blue', alpha=0.6)
                ax.plot(*robot_pos[r], 'o', color='blue')

                if robot_pos[r] == goal and goal_idx not in tasks_done:
                    tasks_done.add(goal_idx)
                    robot_goal[r] = None
                    print(f"{r} finished task {goal_idx}")

        # ORCA collision handling
        newp = orca_collision_avoidance(list(robot_pos), robot_pos, time_step=0.25)
        for r in newp:
            x, y = newp[r]
            hit   = any(ox <= x < ox + w and oy <= y < oy + h for ox, oy, w, h in obstacles)
            clash = any((x, y) == robot_pos[o] and o != r for o in robot_pos)
            if hit or clash:
                newp[r] = agents[int(r[1:])]
        robot_pos.update(newp)

        # capture frame
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        h, w = fig.canvas.get_width_height()[::-1]
        frames.append(img.reshape((h, w, 3)))

        # on last task, save end.png and stop
        if len(tasks_done) == len(tasks) and not took_end:
            end_path = os.path.join(cwd, "end.png")
            fig.savefig(end_path)
            print(f"Saved end.png → {end_path}")
            took_end = True
            anim.event_source.stop()

    anim = animation.FuncAnimation(
        fig, update,
        frames=count(), interval=interval_ms,
        repeat=False, cache_frame_data=False
    )

    # show live
    plt.show()

    # 5) Write out GIF via Pillow
    gif_path = os.path.join(cwd, "simulation.gif")
    fps = max(1, int(1000/interval_ms))
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=interval_ms, loop=0
    )
    print(f"Saved simulation.gif → {gif_path}")


def setup_ui():
    root = tk.Tk()
    root.title("DC-MRTA Simulation")

    tk.Label(root, text="Number of Agents:").pack(pady=(10,0))
    agents_var = tk.IntVar(value=5)
    tk.Spinbox(root, from_=1, to=20, textvariable=agents_var, width=5).pack()

    tk.Label(root, text="Number of Tasks:").pack(pady=(10,0))
    tasks_var = tk.IntVar(value=10)
    tk.Spinbox(root, from_=1, to=50, textvariable=tasks_var, width=5).pack()

    def on_start():
        na = agents_var.get()
        nt = tasks_var.get()
        speed = simpledialog.askinteger(
            "Simulation Speed",
            "Enter speed (ms per frame):",
            parent=root, minvalue=10, maxvalue=1000, initialvalue=200
        )
        if speed is None:
            root.destroy()
            return
        root.withdraw()
        start_simulation(na, nt, speed)
        root.destroy()

    tk.Button(root, text="Start Simulation", command=on_start).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    setup_ui()
