import numpy as np
import heapq
import matplotlib.pyplot as plt


class AStar:
    def __init__(self, map_grid):
        self.map = map_grid
        # 8 направлений движения: (dx, dy, стоимость)
        # Осевые переходы стоят 1.0, диагональные 1.414 (корень из 2)
        self.motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414), (1, -1, 1.414)
        ]

    def heuristic(self, node, goal):
        """
        Октильная эвристика из раздела 2.5.3 твоей ВКР.
        Оптимальна для 8-связной сетки.
        """
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        return (1.414 - 1.0) * min(dx, dy) + max(dx, dy)

    def plan(self, start_m, goal_m):
        """
        Ищет путь от start_m (x, y в метрах) до goal_m (x, y в метрах).
        Возвращает список координат пути [(x1,y1), (x2,y2), ...]
        """
        # Переводим метры в индексы сетки (ячейки)
        start_idx = (int(start_m[0] / self.map.resolution), int(start_m[1] / self.map.resolution))
        goal_idx = (int(goal_m[0] / self.map.resolution), int(goal_m[1] / self.map.resolution))

        # Проверка: если цель внутри препятствия, путь не найти
        if self.map.grid[goal_idx[1], goal_idx[0]] >= 100:
            #print("Ошибка: Цель находится внутри препятствия!")
            return []

        # Очередь с приоритетом: (f_score, (x, y))
        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))

        # Словари для хранения стоимостей и родителей (откуда пришли)
        g_score = {start_idx: 0.0}
        came_from = {}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            # Если достигли цели - восстанавливаем путь
            if current == goal_idx:
                path = []
                while current in came_from:
                    # Переводим обратно в метры для робота
                    path.append((current[0] * self.map.resolution, current[1] * self.map.resolution))
                    current = came_from[current]
                path.append(start_m)
                path.reverse()
                return path

            # Проверяем соседей
            for motion in self.motions:
                neighbor = (current[0] + motion[0], current[1] + motion[1])

                # Проверка выхода за границы карты
                if (neighbor[0] < 0 or neighbor[0] >= self.map.grid_width or
                        neighbor[1] < 0 or neighbor[1] >= self.map.grid_height):
                    continue

                # Считываем стоимость ячейки из карты (0 - свободно, 50 - инфляция, 100 - стена)
                cell_cost = self.map.grid[neighbor[1], neighbor[0]]
                if cell_cost >= 100:
                    continue  # В стену ехать нельзя

                # Если это зона инфляции, добавляем штраф, чтобы робот предпочитал центр прохода
                penalty = 5.0 if cell_cost == 50 else 0.0

                # Считаем новую стоимость пути до соседа
                tentative_g = g_score[current] + motion[2] + penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f_score, neighbor))

        print("Путь не найден!")
        return []


# === БЛОК ПРОВЕРКИ ===
if __name__ == "__main__":
    import sys
    import os

    # Добавляем корневую папку в пути импорта, чтобы найти environment.py
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment import MapGrid

    print("Инициализация карты и планировщика...")
    sim_map = MapGrid()

    # Добавляем препятствия как в основном файле
    sim_map.add_rect_obstacle(0, 0, sim_map.width_m, 0.5)
    sim_map.add_rect_obstacle(0, sim_map.height_m - 0.5, sim_map.width_m, 0.5)
    sim_map.add_rect_obstacle(0, 0, 0.5, sim_map.height_m)
    sim_map.add_rect_obstacle(sim_map.width_m - 0.5, 0, 0.5, sim_map.height_m)
    sim_map.add_rect_obstacle(5.0, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(8.5, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(15.0, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(5.0, 17.0, 12.0, 1.5)

    planner = AStar(sim_map)

    # Пробуем построить путь от зоны погрузки (слева) до склада ГП (справа)
    start_point = (2.0, 8.0)
    goal_point = (22.0, 8.0)

    print(f"Поиск пути от {start_point} до {goal_point}...")
    path = planner.plan(start_point, goal_point)

    # Отрисовка
    fig, ax = plt.subplots(figsize=(12, 8))  # Немного увеличим ширину окна для легенды
    sim_map.draw(ax)

    # Рисуем путь, если он найден
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, color='cyan', linewidth=2, label='Глобальный путь A*')
        ax.plot(start_point[0], start_point[1], 'go', markersize=8, label='Старт')
        ax.plot(goal_point[0], goal_point[1], 'ro', markersize=8, label='Финиш')

    # --- УМНАЯ ОТРИСОВКА ЛЕГЕНДЫ ---
    # 1. Убираем дубликаты
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # 2. Выносим легенду за пределы графика (вправо)
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Поджимаем сам график, чтобы легенда влезла в окно
    plt.tight_layout()
    plt.show()