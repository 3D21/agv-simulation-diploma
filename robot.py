import numpy as np
import matplotlib.patches as patches
import yaml
import os


class AMR:
    def __init__(self, id, start_pose, config_path="config.yaml"):
        self.id = id
        self.x = start_pose[0]
        self.y = start_pose[1]
        self.theta = start_pose[2]

        self.v = 0.0
        self.omega = 0.0
        self.path = []

        # НОВОЕ: У каждого робота свой начальный заряд (от 40% до 85%)
        # Это нужно, чтобы они разряжались по очереди!
        self.battery = 100.0 - (id * 15.0)

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.length = config['robot']['length']
            self.width = config['robot']['width']
            self.max_v = config['robot']['max_speed']
            self.max_w = config['robot']['max_yaw_rate']
        else:
            self.length = 1.0
            self.width = 0.7
            self.max_v = 1.0
            self.max_w = 1.0

    def update_pose(self, v, omega, dt):
        self.v = np.clip(v, -self.max_v, self.max_v)
        self.omega = np.clip(omega, -self.max_w, self.max_w)

        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def draw(self, ax, color='blue'):
        dx = -self.length / 2
        dy = -self.width / 2

        corner_x = self.x + dx * np.cos(self.theta) - dy * np.sin(self.theta)
        corner_y = self.y + dx * np.sin(self.theta) + dy * np.cos(self.theta)

        # Рисуем корпус
        rect = patches.Rectangle((corner_x, corner_y), self.length, self.width,
                                 angle=np.degrees(self.theta),
                                 linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)

        # Рисуем нос
        nose_x = self.x + (self.length / 2) * np.cos(self.theta)
        nose_y = self.y + (self.length / 2) * np.sin(self.theta)
        ax.plot([self.x, nose_x], [self.y, nose_y], color='red', linewidth=2)

        # Номер робота
        ax.text(self.x, self.y, str(self.id), color='white', weight='bold', ha='center', va='center', fontsize=8)

        # НОВОЕ: Индикатор батареи (зеленый > 20%, красный < 20%)
        batt_color = 'red' if self.battery < 20.0 else 'green'
        ax.text(self.x, self.y + 0.8, f"{int(self.battery)}%", color=batt_color,
                weight='bold', ha='center', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))