# Drone coordinate movement

`DroneMotion.py` is a ROS 2 node for the existing simplified Gazebo Parrot.
It reads `/parrot1/odometry`, accepts `PointStamped` goals on `/parrot1/goal`,
and publishes body-frame `Twist` commands on `/parrot1/cmd_vel` at 20 Hz.
The drone model now publishes 3D odometry. This controller uses that simulator
feedback directly, rather than the filtered `/parrot1/odom` estimate.

## Run

Build the changed simulation package from the RoboticsStudio directory:

```bash
cd /home/jaiden/git/RoboticsStudio
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select 41068_ignition_bringup
source install/setup.bash
ros2 launch 41068_ignition_bringup 41068_ignition_parrot.launch.py world:=simple_trees nav2:=false slam:=false
```

Start a fresh simulation after rebuilding so it loads the 3D odometry change.
Press Play if Gazebo is paused. In a second terminal:

In a second terminal
```bash
source /opt/ros/humble/setup.bash
cd /home/jaiden/git/RoboticsStudio/DroneStuff
python3 DroneMotion.py
```

In a third terminal, inspect the current position first:

```bash
source /opt/ros/humble/setup.bash
cd /home/jaiden/git/RoboticsStudio/DroneStuff
ros2 topic echo /parrot1/odometry --once
python3 send_goal.py 0 0 3
```

Coordinates are absolute metres in `parrot1_odom`, the simulator's world-fixed
odometry frame. They are not relative movement offsets or height above terrain.
For a vertical climb, use the current odometry x and y and a higher z; `(0, 0, 3)`
is only an example destination. Choose a point clear of the terrain.
The controller reports acceptance and `Arrived` in its terminal. After arrival,
send another coordinate, for example `python3 send_goal.py 5 2 3`.
The sender reports delivery only, not acceptance or arrival.

To run a named sequence, use `sequence_goals.py`. Waypoints are absolute
positions and are sent in the order they appear:

```bash
python3 sequence_goals.py \
  --waypoint takeoff 2 0 2 \
  --waypoint travel 6 2 2 \
  --waypoint landing 6 2 0.8
```

The script waits for each waypoint to be reached before publishing the next
one. Use `--tolerance` and `--timeout` to adjust arrival distance and the
maximum time allowed for each movement.

## Behaviour and tuning

- Speed decreases with distance and is capped at 1 m/s; arrival tolerance is 0.15 m.
- A new valid goal replaces the previous goal.
- Missing odometry for 1 second or a goal taking 120 seconds cancels movement.
  These timeouts use wall time, including when Gazebo is paused. Send a fresh goal
  after resolving the problem.
- Ctrl+C sends a stop before the controller exits. Force-killing the process cannot
  send that stop; the simulator's velocity plugin retains its last command.
- Run one command controller at a time: disable Nav2 and stop keyboard teleoperation.
- Movement follows a straight line. There is no obstacle avoidance, terrain
  following, automatic takeoff/landing, or realistic rotor control. The existing
  model uses simplified flight physics and has collisions disabled.

Example tuning:

```bash
python3 DroneMotion.py --ros-args -p max_speed:=0.5 -p tolerance:=0.1 -p goal_timeout:=180.0
```

Run the controller checks:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```
