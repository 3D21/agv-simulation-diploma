import math


class TrafficCoordinator:
    def __init__(self, fleet, map_grid):
        self.fleet = fleet
        self.map = map_grid
        self.robot_states = {r.id: "IDLE" for r in fleet}

        # Расставляем 4 станции зарядки вдоль пустого нижнего прохода
        self.charging_stations = [
            (2.0, 2.0),  # Левая (старая)
            (8.0, 2.0),  # Центрально-левая
            (14.0, 2.0),  # Центрально-правая
            (20.0, 2.0)  # Правая
        ]

    def assign_tasks(self):
        for robot in self.fleet:
            # --- 1. ПРИОРИТЕТНОЕ ПРЕРЫВАНИЕ: НИЗКИЙ ЗАРЯД ---
            if robot.battery <= 20.0 and self.robot_states[robot.id] not in ["TO_CHARGE", "CHARGING"]:
                print(f"[Диспетчер] ВНИМАНИЕ! Робот {robot.id} разряжен! Включает мигалки и ищет розетку.")
                self.robot_states[robot.id] = "TO_CHARGE"

                # Ищем ближайшую СВОБОДНУЮ станцию зарядки
                taken_stations = [r.goal for r in self.fleet if self.robot_states[r.id] in ["TO_CHARGE", "CHARGING"]]
                best_station = self.charging_stations[0]
                min_dist = float('inf')

                for st in self.charging_stations:
                    if st not in taken_stations:
                        dist = math.hypot(robot.x - st[0], robot.y - st[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_station = st

                robot.goal = best_station
                robot.path = []  # Сбрасываем рабочий маршрут
                continue

            # --- 2. РАЗДАЧА РАБОТЫ ---
            if self.robot_states[robot.id] == "IDLE" and robot.battery > 20.0:
                target_idx = (robot.id - 1) % len(self.map.loading_points)
                robot.goal = self.map.loading_points[target_idx]
                self.robot_states[robot.id] = "TO_LOAD"

    def update_tasks(self):
        for robot in self.fleet:
            # Симуляция расхода батареи (ограничиваем на 0%, чтобы не уходил в минус)
            if self.robot_states[robot.id] != "CHARGING":
                robot.battery -= 0.10
                if robot.battery < 0: robot.battery = 0.0
            else:
                robot.battery += 1.0
                if robot.battery >= 100.0:
                    robot.battery = 100.0
                    self.robot_states[robot.id] = "IDLE"
                    robot.goal = None
                continue

            if robot.goal is None:
                continue

            dist = math.hypot(robot.x - robot.goal[0], robot.y - robot.goal[1])
            if dist < 0.3:
                current_state = self.robot_states[robot.id]

                if current_state == "TO_CHARGE":
                    self.robot_states[robot.id] = "CHARGING"
                    robot.goal = None
                    robot.path = []
                elif current_state == "TO_LOAD":
                    target_idx = (robot.id - 1) % len(self.map.workstations)
                    robot.goal = self.map.workstations[target_idx]
                    self.robot_states[robot.id] = "TO_WORKSTATION"
                    robot.path = []
                elif current_state == "TO_WORKSTATION":
                    robot.goal = self.map.delivery_points[0]
                    self.robot_states[robot.id] = "TO_DELIVERY"
                    robot.path = []
                elif current_state == "TO_DELIVERY":
                    self.robot_states[robot.id] = "IDLE"
                    robot.goal = None
                    robot.path = []