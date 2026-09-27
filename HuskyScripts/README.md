# Husky movement

`HuskyMotion.py` moves the simulated Husky to absolute XY coordinates. It
reads `/husky1/odometry`, accepts `PointStamped` goals on `/husky1/goal`, and
publishes differential-drive commands on `/husky1/cmd_vel`.

The Husky is a ground vehicle: the goal's `z` value is ignored. The controller
turns toward each goal and then drives forward; it does not perform obstacle
avoidance.

## Run

From the RoboticsStudio directory, start the simulation and then run:

```bash
source /opt/ros/humble/setup.bash
cd /home/aphri/robotStudio/RoboticsStudio
python3 HuskyScripts/HuskyMotion.py
```

The simulation must have the Husky enabled. To inspect its current position:

```bash
ros2 topic echo /husky1/odometry --once
```

## Named movement sequence

The shared sequence runner can send named waypoints to the Husky separately
from the Parrot. Waypoints are absolute positions in `husky1_odom`; X and Y
are used and Z is ignored:

```bash
python3 DroneStuff/sequence_goals.py \
  --robot husky \
  --waypoint leave_base 0 0 0 \
  --waypoint travel 6 2 0 \
  --waypoint return 0 0 0
```

The runner waits for each waypoint to be reached before sending the next one.
Use `--tolerance` and `--timeout` to adjust arrival distance and the maximum
time allowed for each movement.

## Tuning

Controller parameters can be overridden with ROS arguments, for example:

```bash
python3 HuskyScripts/HuskyMotion.py \
  --ros-args -p max_speed:=0.5 -p position_tolerance:=0.1
```

Stop the controller with Ctrl+C so it publishes a final zero velocity.