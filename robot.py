import numpy as np
import matplotlib.patches as patches
import yaml
import os


class AMR:
    def __init__(self, id, start_pose, config_path="config.yaml"):
        self.id = id

        # Поза: x (метры), y (метры), theta (радианы)
        self.x = start_pose[0]
        self.y = start_pose[1]
        self.theta = start_pose[2]

        # Текущие скорости
        self.v = 0.0  # Линейная скорость
        self.omega = 0.0  # Угловая скорость

        # Глобальный путь (будет назначаться планировщиком A*)
        self.path = []
        self.is_waiting = False  # Флаг для диспетчера (остановка)

        # Читаем габариты из конфига
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.length = config['robot']['length']
            self.width = config['robot']['width']
            self.max_v = config['robot']['max_speed']
            self.max_w = config['robot']['max_yaw_rate']
        else:
            # Дефолтные значения, если конфиг не найден
            self.length = 1.0
            self.width = 0.7
            self.max_v = 1.0
            self.max_w = 1.0

    def update_pose(self, v, omega, dt):
        """
        Кинематика дифференциально-приводного робота.
        Обновляет координаты за шаг времени dt.
        """
        # Ограничиваем скорости максимальными значениями (физика робота)
        self.v = np.clip(v, -self.max_v, self.max_v)
        self.omega = np.clip(omega, -self.max_w, self.max_w)

        # Пересчитываем координаты
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt

        # Нормализуем угол, чтобы он всегда был от -pi до pi
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def draw(self, ax, color='blue'):
        """Отрисовка прямоугольного робота и вектора направления на графике."""
        # Находим координаты левого нижнего угла прямоугольника с учетом поворота
        dx = -self.length / 2
        dy = -self.width / 2

        corner_x = self.x + dx * np.cos(self.theta) - dy * np.sin(self.theta)
        corner_y = self.y + dx * np.sin(self.theta) + dy * np.cos(self.theta)

        # Рисуем корпус (прямоугольник)
        rect = patches.Rectangle((corner_x, corner_y), self.length, self.width,
                                 angle=np.degrees(self.theta),
                                 linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)

        # Рисуем "нос" (красная линия, показывающая, куда смотрит робот)
        nose_x = self.x + (self.length / 2) * np.cos(self.theta)
        nose_y = self.y + (self.length / 2) * np.sin(self.theta)
        ax.plot([self.x, nose_x], [self.y, nose_y], color='red', linewidth=2)

        # Пишем номер (ID) робота по центру
        ax.text(self.x, self.y, str(self.id), color='white', weight='bold',
                ha='center', va='center', fontsize=8)