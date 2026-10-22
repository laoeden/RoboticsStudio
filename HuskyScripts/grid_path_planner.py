"""Small local grid planner for Husky lidar navigation."""

import heapq

import numpy as np

# Obstacles are always kept at least this far from the planned path. The larger
# robot_clearance is a preference that is relaxed near the start and the goal (see
# plan_path), so a robot parked inside it can still drive out and a goal close to an
# obstacle can still be reached.
HARD_CLEARANCE = 0.6


def plan_path(
    robot_x: float,
    robot_y: float,
    goal: np.ndarray | None,
    obstacle_points: np.ndarray,
    resolution: float = 0.2,
    robot_clearance: float = 1.0,
) -> np.ndarray:
    if goal is None:
        return np.empty((0, 2), dtype=np.float32)

    start_cell = (int(round(robot_x / resolution)), int(round(robot_y / resolution)))
    goal_cell = (int(round(float(goal[0]) / resolution)), int(round(float(goal[1]) / resolution)))
    obstacle_cells: set[tuple[int, int]] = set()
    core_cells: set[tuple[int, int]] = set()
    inflation = int(np.ceil(robot_clearance / resolution))
    core_inflation = min(inflation, int(np.ceil(HARD_CLEARANCE / resolution)))
    for point in obstacle_points:
        cell_x = int(round(float(point[0]) / resolution))
        cell_y = int(round(float(point[1]) / resolution))
        for offset_x in range(-inflation, inflation + 1):
            for offset_y in range(-inflation, inflation + 1):
                distance_squared = offset_x * offset_x + offset_y * offset_y
                if distance_squared <= inflation * inflation:
                    obstacle_cells.add((cell_x + offset_x, cell_y + offset_y))
                if distance_squared <= core_inflation * core_inflation:
                    core_cells.add((cell_x + offset_x, cell_y + offset_y))
    # A start or goal inside the clearance margin would otherwise be walled in by it.
    # Open the margin (but never the hard core) around such a cell.
    reach_squared = (inflation + 1) ** 2
    for anchor in (start_cell, goal_cell):
        if anchor in obstacle_cells:
            obstacle_cells = {
                cell for cell in obstacle_cells
                if cell in core_cells
                or (cell[0] - anchor[0]) ** 2 + (cell[1] - anchor[1]) ** 2 > reach_squared
            }
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
