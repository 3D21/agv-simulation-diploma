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


def show_dashboard(fleet, history):
    """Генерация аналитического дашборда после окончания симуляции"""
    print("\nГенерация аналитического отчета...")
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 10))
    fig.canvas.manager.set_window_title('Аналитический Дашборд КФС')
    colors = {1: 'blue', 2: 'orange', 3: 'purple', 4: 'cyan'}

    # 1. Линейный график: Жизненный цикл батарей (Ambulance Routing)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("График разряда/заряда батарей во времени", fontweight='bold')
    ax1.set_xlabel("Время (кадры симуляции)")
    ax1.set_ylabel("Заряд батареи (%)")
    for r in fleet:
        ax1.plot(history[r.id]['battery'], label=f"Робот {r.id}", color=colors[r.id], linewidth=2)
    ax1.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Критический заряд (20%)')
    ax1.legend()

    # 2. Столбчатая диаграмма: Производительность (Throughput)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("Производительность (Доставлено деталей)", fontweight='bold')
    ax2.set_ylabel("Количество деталей")
    cycles = [history[r.id]['cycles'] for r in fleet]
    bars = ax2.bar([f"Робот {r.id}" for r in fleet], cycles, color=[colors[r.id] for r in fleet])
    ax2.bar_label(bars)  # Выводим цифры над столбцами

    # 3. Круговые диаграммы: Эффективность (Utilization)
    state_colors = ['#2ca02c', '#d62728', '#ff7f0e', '#7f7f7f']  # Зеленый, Красный, Оранжевый, Серый
    for idx, r in enumerate(fleet):
        ax_pie = plt.subplot(2, 4, 5 + idx)
        states = history[r.id]['states']
        labels = ['В движении\n(Работа)', 'Ожидание\n(Пробки)', 'Зарядка', 'Простой\n(Нет задач)']
        sizes = [states['MOVING'], states['WAITING'], states['CHARGING'], states['IDLE']]

        # Фильтруем нули, чтобы график был красивым
        labels_filt = [l for s, l in zip(sizes, labels) if s > 0]
        sizes_filt = [s for s in sizes if s > 0]
        colors_filt = [c for s, c in zip(sizes, state_colors) if s > 0]

        ax_pie.pie(sizes_filt, labels=labels_filt, colors=colors_filt, autopct='%1.1f%%', startangle=140)
        ax_pie.set_title(f"Робот {r.id}", color=colors[r.id], fontweight='bold')

    plt.suptitle("Сводный отчет о работе флота AMR", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


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

    # --- ИНИЦИАЛИЗАЦИЯ СБОРА МЕТРИК ---
    history = {
        r.id: {
            'battery': [],
            'states': {'MOVING': 0, 'WAITING': 0, 'CHARGING': 0, 'IDLE': 0},
            'cycles': 0
        } for r in fleet
    }
    prev_states = {r.id: "IDLE" for r in fleet}

    coordinator.assign_tasks()
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title('Симуляция Цеха. ЗАКРОЙТЕ ОКНО для вывода аналитики')

    print("\n" + "=" * 50)
    print("ВНИМАНИЕ: Чтобы посмотреть аналитический дашборд,")
    print("просто ЗАКРОЙТЕ ОКНО СИМУЛЯЦИИ (нажмите крестик)!")
    print("=" * 50 + "\n")

    # Симуляция крутится, пока окно открыто
    while plt.fignum_exists(fig.number):
        coordinator.update_tasks()
        coordinator.assign_tasks()

        # Динамические приоритеты
        for robot in fleet:
            robot.priority = robot.id
            if coordinator.robot_states[robot.id] == "TO_CHARGE":
                robot.priority -= 100

        for robot in fleet:
            # --- СБОР МЕТРИК ---
            history[robot.id]['battery'].append(robot.battery)

            curr_state = coordinator.robot_states[robot.id]
            # Подсчет доставленных деталей: если статус сменился с доставки на ожидание ИЛИ новую погрузку
            if prev_states[robot.id] == "TO_DELIVERY" and curr_state in ["IDLE", "TO_LOAD"]:
                history[robot.id]['cycles'] += 1
            prev_states[robot.id] = curr_state

            # Подсчет времени состояний
            if curr_state in ["CHARGING", "TO_CHARGE"]:
                history[robot.id]['states']['CHARGING'] += 1
            elif curr_state == "IDLE":
                history[robot.id]['states']['IDLE'] += 1
            else:
                if abs(robot.v) < 0.05 and abs(robot.omega) < 0.05 and robot.path:
                    history[robot.id]['states']['WAITING'] += 1  # Стоим в пробке
                else:
                    history[robot.id]['states']['MOVING'] += 1  # Полезная работа
            # --------------------

            if robot.goal is None:
                robot.path = []
                robot.update_pose(0, 0, local_planner.dt)
                continue

            if not hasattr(robot, 'plan_timer'): robot.plan_timer = 0
            robot.plan_timer += 1

            needs_replanning = not robot.path or robot.plan_timer > 10

            # Глобальное планирование
            if needs_replanning:
                robot.plan_timer = 0
                temp_blocked = {}

                for other in fleet:
                    if other.id != robot.id:
                        block_area(sim_map, temp_blocked, other.x, other.y, 0.8, 100)
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

            # Локальное управление
            if robot.path:
                local_goal = get_local_goal(robot, robot.path, lookahead_dist=1.2)
                v, omega, _ = local_planner.compute_velocity(robot, sim_map, local_goal, fleet)
            else:
                v, omega = 0.0, 0.0

            robot.update_pose(v, omega, dt=local_planner.dt)

        # Отрисовка
        ax.clear()
        sim_map.draw(ax)

        for st in coordinator.charging_stations:
            circle = plt.Circle(st, 0.8, color='gold', alpha=0.4, zorder=1)
            ax.add_patch(circle)

        colors = ['blue', 'orange', 'purple', 'cyan']
        for i, robot in enumerate(fleet):
            robot.draw(ax, color=colors[i])
            if robot.path:
                if coordinator.robot_states[robot.id] in ["TO_CHARGE", "CHARGING"]:
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

    # --- ВЫЗОВ ДАШБОРДА ПОСЛЕ ЗАКРЫТИЯ ОКНА ---
    show_dashboard(fleet, history)


if __name__ == '__main__':
    main()