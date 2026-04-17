import math


class TrafficCoordinator:
    def __init__(self, fleet, map_grid):
        self.fleet = fleet
        self.map = map_grid

        # Статусы заданий для роботов:
        # IDLE (свободен), TO_LOAD (на погрузку),
        # TO_WORKSTATION (на станцию), TO_DELIVERY (на склад)
        self.robot_states = {r.id: "IDLE" for r in fleet}

    def assign_tasks(self):
        """Раздача первичных заданий свободным роботам"""
        for robot in self.fleet:
            if self.robot_states[robot.id] == "IDLE":
                # Распределяем роботов по разным точкам погрузки (по их ID)
                target_idx = (robot.id - 1) % len(self.map.loading_points)
                robot.goal = self.map.loading_points[target_idx]
                self.robot_states[robot.id] = "TO_LOAD"
                print(f"[Диспетчер] Робот {robot.id} получил задание: ехать на ПОГРУЗКУ.")

    def update_tasks(self):
        """Проверяет, доехал ли робот, и переключает его на следующий этап"""
        for robot in self.fleet:
            if robot.goal is None:
                continue

            # Проверяем расстояние до цели
            dist = math.hypot(robot.x - robot.goal[0], robot.y - robot.goal[1])
            if dist < 0.3:  # Доехали (допуск 30 см)
                current_state = self.robot_states[robot.id]

                if current_state == "TO_LOAD":
                    # Загрузились -> едем на рабочую станцию
                    target_idx = (robot.id - 1) % len(self.map.workstations)
                    robot.goal = self.map.workstations[target_idx]
                    self.robot_states[robot.id] = "TO_WORKSTATION"
                    robot.path = []  # Сбрасываем старый путь, чтобы A* построил новый
                    print(f"[Диспетчер] Робот {robot.id} загружен. Едет на СТАНЦИЮ.")

                elif current_state == "TO_WORKSTATION":
                    # Собрали деталь -> едем на склад готовой продукции
                    robot.goal = self.map.delivery_points[0]
                    self.robot_states[robot.id] = "TO_DELIVERY"
                    robot.path = []
                    print(f"[Диспетчер] Робот {robot.id} обработал деталь. Едет на СКЛАД ГП.")

                elif current_state == "TO_DELIVERY":
                    # Доставили -> цикл завершен, робот снова свободен
                    self.robot_states[robot.id] = "IDLE"
                    robot.goal = None
                    robot.path = []
                    print(f"[Диспетчер] Робот {robot.id} завершил цикл доставки!")