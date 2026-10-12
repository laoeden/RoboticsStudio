#!/usr/bin/env python3
"""View Husky odometry and manage goals with a matplotlib map."""

import argparse
import math
import matplotlib.pyplot as plt
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class GoalVisualizer(Node):
    def __init__(self, namespace: str, frame: str, goal_tolerance: float) -> None:
        super().__init__('goal_visualizer', namespace=namespace)
        self.frame = frame
        self.goal_tolerance = goal_tolerance
        self.position = None
        self.start_position = None
        self.path_x = []
        self.path_y = []
        self.goals = []
        self.active_goal = None
        self.pending_goals = []
        self.goal_publisher = self.create_publisher(PointStamped, 'goal', 10)
        self.command_publisher = self.create_publisher(String, 'mission_command', 10)
        self.create_subscription(Odometry, 'odometry', self.on_odometry, 10)

    def on_odometry(self, message: Odometry) -> None:
        position = message.pose.pose.position
        if not all(math.isfinite(value) for value in (position.x, position.y)):
            return
        self.position = (position.x, position.y)
        if self.start_position is None:
            self.start_position = self.position
        self.path_x.append(position.x)
        self.path_y.append(position.y)
        if (
            self.active_goal is not None
            and math.hypot(
                position.x - self.active_goal[0],
                position.y - self.active_goal[1],
            ) <= self.goal_tolerance
        ):
            self.get_logger().info(
                f'Goal reached ({self.active_goal[0]:.2f}, '
                f'{self.active_goal[1]:.2f}).'
            )
            self.active_goal = None
            self.publish_next_goal()

    def publish_goal(self, x: float, y: float) -> None:
        self.goals.append((x, y))
        self.pending_goals.append((x, y))
        self.publish_next_goal()

    def publish_next_goal(self) -> None:
        if self.active_goal is not None or not self.pending_goals:
            return
        x, y = self.pending_goals.pop(0)
        message = PointStamped()
        message.header.frame_id = self.frame
        message.point.x = x
        message.point.y = y
        self.goal_publisher.publish(message)
        self.active_goal = (x, y)
        self.get_logger().info(f'Published goal ({x:.2f}, {y:.2f}).')

    def clear_goal(self) -> None:
        message = String()
        message.data = 'CLEAR_GOAL'
        self.command_publisher.publish(message)
        self.goals.clear()
        self.pending_goals.clear()
        self.active_goal = None
        self.get_logger().info('Cleared the active Husky goal.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='husky1')
    parser.add_argument('--frame', default='husky1/odom')
    parser.add_argument('--goal-tolerance', type=float, default=0.2)
    args = parser.parse_args()
    if args.goal_tolerance <= 0:
        parser.error('--goal-tolerance must be positive')

    rclpy.init(args=[])
    node = GoalVisualizer(args.namespace, args.frame, args.goal_tolerance)

    figure, axis = plt.subplots()
    axis.set_title(
        'Husky goals: left-click queue (FIFO), right/middle-click clear'
    )
    axis.set_xlabel('X (m)')
    axis.set_ylabel('Y (m)')
    axis.grid(True)
    (path_line,) = axis.plot([], [], 'b-', label='odometry')
    robot_point, = axis.plot([], [], 'bo', label='Husky')
    goal_points, = axis.plot([], [], 'rx', markersize=10, label='goals')
    axis.legend()

    def click(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            node.publish_goal(event.xdata, event.ydata)
        elif event.button in (2, 3):
            node.clear_goal()

    figure.canvas.mpl_connect('button_press_event', click)

    try:
        map_initialized = False
        while plt.fignum_exists(figure.number):
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.start_position is not None and not map_initialized:
                start_x, start_y = node.start_position
                axis.set_xlim(start_x - 25.0, start_x + 25.0)
                axis.set_ylim(start_y - 25.0, start_y + 25.0)
                map_initialized = True
            if node.path_x:
                path_line.set_data(node.path_x, node.path_y)
            if node.position is not None:
                robot_point.set_data([node.position[0]], [node.position[1]])
            visible_goals = (
                [node.active_goal] if node.active_goal is not None
                else []
            ) + node.pending_goals
            if visible_goals:
                goal_points.set_data(
                    [goal[0] for goal in visible_goals],
                    [goal[1] for goal in visible_goals],
                )
            else:
                goal_points.set_data([], [])
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.05)
    finally:
        node.clear_goal()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
