import numpy as np
import math


class DWA:
    def __init__(self):
        self.max_v = 1.2
        self.max_yaw = 1.5
        self.dt = 0.1

    def compute_velocity(self, robot, map_grid, local_goal, fleet):
        # 1. ЗАЩИТА ОТ ФИЗИЧЕСКИХ АВАРИЙ С УЧЕТОМ "МИГАЛОК"
        for other in fleet:
            if other.id != robot.id:
                dist = math.hypot(robot.x - other.x, robot.y - other.y)
                if dist < 0.85:
                    # Теперь мы сравниваем ДИНАМИЧЕСКИЙ приоритет, а не ID!
                    if robot.priority > other.priority:
                        return 0.0, 0.0, None

        # 2. РУЛЕНИЕ НА ТОЧКУ
        dx = local_goal[0] - robot.x
        dy = local_goal[1] - robot.y
        target_angle = math.atan2(dy, dx)

        error_angle = target_angle - robot.theta
        error_angle = (error_angle + math.pi) % (2 * math.pi) - math.pi
        dist_to_goal = math.hypot(dx, dy)

        if abs(error_angle) > 0.5:
            v = 0.0
            omega = np.sign(error_angle) * self.max_yaw
        else:
            v = self.max_v if dist_to_goal > 0.5 else 0.6
            omega = error_angle * 2.5

        v = np.clip(v, 0, self.max_v)
        omega = np.clip(omega, -self.max_yaw, self.max_yaw)

        # 3. ЗАЩИТА ОТ СТЕН
        sim_x, sim_y, sim_theta = robot.x, robot.y, robot.theta
        hit_wall = False
        traj = []

        for _ in range(int(1.0 / self.dt)):
            sim_x += v * math.cos(sim_theta) * self.dt
            sim_y += v * math.sin(sim_theta) * self.dt
            sim_theta += omega * self.dt
            traj.append([sim_x, sim_y, sim_theta])

            ix = int(sim_x / map_grid.resolution)
            iy = int(sim_y / map_grid.resolution)

            if ix < 0 or ix >= map_grid.grid_width or iy < 0 or iy >= map_grid.grid_height:
                hit_wall = True;
                break
            if map_grid.grid[iy, ix] >= 100:
                hit_wall = True;
                break

        if hit_wall:
            v = 0.0
            omega = self.max_yaw if error_angle > 0 else -self.max_yaw

        return v, omega, np.array(traj) if traj else None