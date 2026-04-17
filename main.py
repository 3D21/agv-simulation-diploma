import matplotlib.pyplot as plt
import numpy as np
import math

from environment import MapGrid
from robot import AMR
from navigation.a_star import AStar
from navigation.dwa import DWA
from fleet.coordinator import TrafficCoordinator


def get_local_goal(robot, path, lookahead_dist=1.2):
    if not path: return (robot.x, robot.y)
    min_dist = float('inf')
    closest_idx = 0
    for i, p in enumerate(path):
        dist = math.hypot(robot.x - p[0], robot.y - p[1])
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    target_idx = closest_idx
    dist_ahead = 0.0
    while target_idx < len(path) - 1 and dist_ahead < lookahead_dist:
        target_idx += 1
        p1 = path[target_idx - 1]
        p2 = path[target_idx]
        dist_ahead += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return path[target_idx]


def block_area(sim_map, temp_blocked, x_m, y_m, radius_m, cost_value):
    ix = int(x_m / sim_map.resolution)
    iy = int(y_m / sim_map.resolution)
    cells = int(radius_m / sim_map.resolution)
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            ny, nx = iy + dy, ix + dx
            if 0 <= ny < sim_map.grid_height and 0 <= nx < sim_map.grid_width:
                if (ny, nx) not in temp_blocked:
                    temp_blocked[(ny, nx)] = sim_map.grid[ny, nx]
                sim_map.grid[ny, nx] = cost_value


def main():
    print("Инициализация цеха...")
    sim_map = MapGrid("config.yaml")

    sim_map.add_rect_obstacle(0, 0, sim_map.width_m, 0.5)
    sim_map.add_rect_obstacle(0, sim_map.height_m - 0.5, sim_map.width_m, 0.5)
    sim_map.add_rect_obstacle(0, 0, 0.5, sim_map.height_m)
    sim_map.add_rect_obstacle(sim_map.width_m - 0.5, 0, 0.5, sim_map.height_m)
    sim_map.add_rect_obstacle(5.0, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(8.5, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(15.0, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(5.0, 17.0, 12.0, 1.5)

    print("Инициализация флота...")
    fleet = [
        AMR(id=1, start_pose=(2.0, 2.0, 1.57)),
        AMR(id=2, start_pose=(4.0, 2.0, 1.57)),
        AMR(id=3, start_pose=(6.0, 2.0, 1.57)),
        AMR(id=4, start_pose=(8.0, 2.0, 1.57))
    ]

    global_planner = AStar(sim_map)
    local_planner = DWA()
    coordinator = TrafficCoordinator(fleet, sim_map)

    coordinator.assign_tasks()
    fig, ax = plt.subplots(figsize=(12, 8))

    while True:
        coordinator.update_tasks()
        coordinator.assign_tasks()

        # --- 0. ДИНАМИЧЕСКИЕ ПРИОРИТЕТЫ ---
        for robot in fleet:
            robot.priority = robot.id
            # Если едем на зарядку, вычитаем 100.
            # Робот 4 превратится в -96, что меньше (важнее) чем 1.
            if coordinator.robot_states[robot.id] == "TO_CHARGE":
                robot.priority -= 100

        for robot in fleet:
            if robot.goal is None:
                robot.path = []
                robot.update_pose(0, 0, local_planner.dt)
                continue

            if not hasattr(robot, 'plan_timer'): robot.plan_timer = 0
            robot.plan_timer += 1

            needs_replanning = not robot.path or robot.plan_timer > 10

            # --- ГЛОБАЛЬНОЕ ПЛАНИРОВАНИЕ ---
            if needs_replanning:
                robot.plan_timer = 0
                temp_blocked = {}

                for other in fleet:
                    if other.id != robot.id:
                        block_area(sim_map, temp_blocked, other.x, other.y, 0.8, 100)

                        # Бронирование коридоров теперь зависит от ДИНАМИЧЕСКОГО ПРИОРИТЕТА
                        if other.priority < robot.priority and other.path:
                            for p in other.path[::4]:
                                block_area(sim_map, temp_blocked, p[0], p[1], 0.5, 100)

                block_area(sim_map, temp_blocked, robot.x, robot.y, 0.4, 0)
                new_path = global_planner.plan((robot.x, robot.y), robot.goal)

                for (ny, nx), old_val in temp_blocked.items():
                    sim_map.grid[ny, nx] = old_val

                if new_path:
                    robot.path = new_path
                else:
                    robot.path = []

            # --- ЛОКАЛЬНОЕ УПРАВЛЕНИЕ ---
            if robot.path:
                local_goal = get_local_goal(robot, robot.path, lookahead_dist=1.2)
                v, omega, _ = local_planner.compute_velocity(robot, sim_map, local_goal, fleet)
            else:
                v, omega = 0.0, 0.0

            robot.update_pose(v, omega, dt=local_planner.dt)

        # --- ОТРИСОВКА ---
        ax.clear()
        sim_map.draw(ax)

        # Рисуем все 4 станции зарядки явно поверх карты (светло-желтые зоны)
        for st in coordinator.charging_stations:
            circle = plt.Circle(st, 0.8, color='gold', alpha=0.4, zorder=1)
            ax.add_patch(circle)

        colors = ['blue', 'orange', 'purple', 'cyan']
        for i, robot in enumerate(fleet):
            robot.draw(ax, color=colors[i])
            if robot.path:
                # Если едет на зарядку - рисуем жирный красный пунктир маршрута!
                if coordinator.robot_states[robot.id] == "TO_CHARGE":
                    path_style = {'color': 'red', 'linewidth': 2.5, 'linestyle': '--', 'alpha': 0.8}
                else:
                    path_style = {'color': colors[i], 'linewidth': 1.5, 'linestyle': '-', 'alpha': 0.6}

                path_x = [p[0] for p in robot.path]
                path_y = [p[1] for p in robot.path]
                ax.plot(path_x, path_y, **path_style)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.pause(0.01)


if __name__ == '__main__':
    main()