import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import os


class MapGrid:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл {config_path} не найден!")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.width_m = config['map']['width']
        self.height_m = config['map']['height']
        self.resolution = config['map']['resolution']

        self.circumscribed_radius = config['robot']['circumscribed_radius']
        self.d_safe = 0.10
        self.r_inflate = self.circumscribed_radius + self.d_safe

        self.grid_width = int(self.width_m / self.resolution)
        self.grid_height = int(self.height_m / self.resolution)

        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)

        # НОВОЕ: Считываем координаты технологических точек
        self.loading_points = config['points']['loading']
        self.workstations = config['points']['workstations']
        self.delivery_points = config['points']['delivery']
        self.charging_points = config['points']['charging']

    def add_rect_obstacle(self, x_m, y_m, w_m, h_m):
        # --- 1. Рассчитываем и наносим зону инфляции ---
        ix_start = max(0, int((x_m - self.r_inflate) / self.resolution))
        iy_start = max(0, int((y_m - self.r_inflate) / self.resolution))
        ix_end = min(self.grid_width, int((x_m + w_m + self.r_inflate) / self.resolution))
        iy_end = min(self.grid_height, int((y_m + h_m + self.r_inflate) / self.resolution))

        view = self.grid[iy_start:iy_end, ix_start:ix_end]
        view[view < 100] = 50

        # --- 2. Рассчитываем и наносим глухое препятствие ---
        ox_start = max(0, int(x_m / self.resolution))
        oy_start = max(0, int(y_m / self.resolution))
        ox_end = min(self.grid_width, int((x_m + w_m) / self.resolution))
        oy_end = min(self.grid_height, int((y_m + h_m) / self.resolution))

        self.grid[oy_start:oy_end, ox_start:ox_end] = 100

    def draw(self, ax):
        """Отрисовывает costmap и технологические точки"""
        ax.imshow(self.grid, cmap='Greys', origin='lower',
                  extent=[0, self.width_m, 0, self.height_m],
                  vmin=0, vmax=100)

        # НОВОЕ: Отрисовка технологических точек (как на 3D картинках)
        point_radius = 0.5

        for pt in self.loading_points:
            ax.add_patch(patches.Circle(pt, point_radius, color='red', alpha=0.5, label='Погрузка'))

        for pt in self.workstations:
            ax.add_patch(patches.Circle(pt, point_radius, color='green', alpha=0.5, label='Станция'))

        for pt in self.delivery_points:
            ax.add_patch(patches.Circle(pt, point_radius, color='blue', alpha=0.5, label='Склад ГП'))

        for pt in self.charging_points:
            ax.add_patch(patches.Circle(pt, point_radius, color='yellow', alpha=0.5, label='Зарядка'))

        ax.set_xlim(0, self.width_m)
        ax.set_ylim(0, self.height_m)
        ax.set_xlabel('Ось X [метры]')
        ax.set_ylabel('Ось Y [метры]')
        ax.set_title('Global Costmap и Технологические зоны')
        ax.grid(True, color='gray', alpha=0.2, linestyle='--')


# === БЛОК ПРОВЕРКИ ===
if __name__ == "__main__":
    sim_map = MapGrid("config.yaml")

    sim_map.add_rect_obstacle(0, 0, sim_map.width_m, 0.5)
    sim_map.add_rect_obstacle(0, sim_map.height_m - 0.5, sim_map.width_m, 0.5)
    sim_map.add_rect_obstacle(0, 0, 0.5, sim_map.height_m)
    sim_map.add_rect_obstacle(sim_map.width_m - 0.5, 0, 0.5, sim_map.height_m)

    sim_map.add_rect_obstacle(5.0, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(8.5, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(15.0, 5.0, 2.0, 10.0)
    sim_map.add_rect_obstacle(5.0, 17.0, 12.0, 1.5)

    fig, ax = plt.subplots(figsize=(10, 8))
    sim_map.draw(ax)

    # Убираем дубликаты из легенды
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.show()