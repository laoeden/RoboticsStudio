from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot = LaunchConfiguration('robot')
    goal_distance = LaunchConfiguration('goal_distance')
    goal_timeout = LaunchConfiguration('goal_timeout')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            default_value='husky1',
            choices=['husky1', 'parrot1'],
            description='Robot namespace to explore.',
        ),
        DeclareLaunchArgument(
            'goal_distance',
            default_value='2.5',
            description='Minimum distance from the robot for a frontier goal.',
        ),
        DeclareLaunchArgument(
            'goal_timeout',
            default_value='20.0',
            description='Seconds before a stalled Nav2 goal is cancelled.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulator time.',
        ),
        Node(
            package='41068_ignition_bringup',
            executable='voxel_nav_explorer.py',
            namespace=robot,
            name='voxel_nav_explorer',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
            arguments=[
                '--robot', robot,
                '--goal-distance', goal_distance,
                '--goal-timeout', goal_timeout,
            ],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
            ],
        ),
    ])