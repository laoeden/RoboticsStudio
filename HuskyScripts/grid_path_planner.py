"""Small local grid planner for Husky lidar navigation."""

import heapq

import numpy as np


def plan_path(
    robot_x: float,
    robot_y: float,
    goal: np.ndarray | None,
    obstacle_points: np.ndarray,
    resolution: float = 0.2,
    robot_clearance: float = 0.45,
) -> np.ndarray:
    if goal is None:
        return np.empty((0, 2), dtype=np.float32)

    start_cell = (int(round(robot_x / resolution)), int(round(robot_y / resolution)))
    goal_cell = (int(round(float(goal[0]) / resolution)), int(round(float(goal[1]) / resolution)))
    obstacle_cells: set[tuple[int, int]] = set()
    inflation = int(np.ceil(robot_clearance / resolution))
    for point in obstacle_points:
        cell_x = int(round(float(point[0]) / resolution))
        cell_y = int(round(float(point[1]) / resolution))
        for offset_x in range(-inflation, inflation + 1):
            for offset_y in range(-inflation, inflation + 1):
                if offset_x * offset_x + offset_y * offset_y <= inflation * inflation:
                    obstacle_cells.add((cell_x + offset_x, cell_y + offset_y))
    obstacle_cells.discard(start_cell)
    obstacle_cells.discard(goal_cell)

    margin = int(np.ceil(2.0 / resolution))
    min_x = min(start_cell[0], goal_cell[0]) - margin
    max_x = max(start_cell[0], goal_cell[0]) + margin
    min_y = min(start_cell[1], goal_cell[1]) - margin
    max_y = max(start_cell[1], goal_cell[1]) + margin
    if obstacle_cells:
        min_x = min(min_x, min(cell[0] for cell in obstacle_cells) - margin)
        max_x = max(max_x, max(cell[0] for cell in obstacle_cells) + margin)
        min_y = min(min_y, min(cell[1] for cell in obstacle_cells) - margin)
        max_y = max(max_y, max(cell[1] for cell in obstacle_cells) + margin)

    neighbors = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
    )
    frontier = [(0.0, start_cell)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far = {start_cell: 0.0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_cell:
            path_cells = [current]
            while current in came_from:
                current = came_from[current]
                path_cells.append(current)
            path_cells.reverse()
            return np.asarray(path_cells, dtype=np.float32) * resolution
        for dx, dy, step_cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (min_x <= neighbor[0] <= max_x and min_y <= neighbor[1] <= max_y):
                continue
            if neighbor in obstacle_cells:
                continue
            new_cost = cost_so_far[current] + step_cost
            if new_cost >= cost_so_far.get(neighbor, float('inf')):
                continue
            cost_so_far[neighbor] = new_cost
            heuristic = np.hypot(goal_cell[0] - neighbor[0], goal_cell[1] - neighbor[1])
            heapq.heappush(frontier, (new_cost + heuristic, neighbor))
            came_from[neighbor] = current
    return np.empty((0, 2), dtype=np.float32)
