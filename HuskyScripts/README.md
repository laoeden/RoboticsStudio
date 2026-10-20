# Husky movement

`HuskyMotion.py` moves the simulated Husky to absolute XY coordinates. It
reads `/husky1/odometry`, accepts `PointStamped` goals on `/husky1/goal`, and
publishes differential-drive commands on `/husky1/cmd_vel`.

The Husky is a ground vehicle: the goal's `z` value is ignored. Goals are
absolute XY coordinates in the odometry frame. The controller uses odometry
XY and yaw for travel and heading. GPS remains available for optional
diagnostics, but is not required for goal acceptance, path following,
progress detection, or waypoint completion.

While moving, it monitors `/husky1/scan`. Any valid lidar return within 45
degrees of the front and within 1 metre cancels the active goal and publishes
an obstacle-stop event. A Husky sequence receives that event and cancels all
remaining waypoints.

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

For a conservative single-waypoint test from the usual Husky spawn near
`(-5, 0)`, run the `run_farmer.bash` launch first, start `HuskyMotion.py` in a
second terminal, then send:

```bash
python3 DroneStuff/sequence_goals.py \
  --robot husky \
  --waypoint test_forward -4.0 0.0 0.0 \
  --tolerance 0.2 \
  --timeout 60
```

The Husky speed defaults are intentionally limited to `0.5 m/s`. While turning
outside the `heading_tolerance`, it commands at least `0.8 rad/s` so small
heading corrections do not stall against terrain contact; it stops turning
inside the tolerance. Turning uses `2.0` angular gain and a `1.0 rad/s`
maximum. Override these only for a controlled test:

```bash
python3 HuskyScripts/HuskyMotion.py \
  --ros-args -p max_speed:=0.8 -p max_angular_speed:=0.8
```

## Tuning

Controller parameters can be overridden with ROS arguments, for example:

```bash
python3 HuskyScripts/HuskyMotion.py \
  --ros-args -p max_speed:=0.5 -p position_tolerance:=0.1
```

If a controlled rotation test demonstrates that the odometry yaw is aligned
differently, override the conversion, for example:

```bash
python3 HuskyScripts/HuskyMotion.py \
  --ros-args -p odom_to_world_heading:=90.0
```

The obstacle settings can also be changed:

```bash
python3 HuskyScripts/HuskyMotion.py \
  --ros-args -p obstacle_distance:=1.5 -p front_angle:=0.785398
```

Stop the controller with Ctrl+C so it publishes a final zero velocity.

The Husky uses Ignition's `VelocityControl` system on the model `cmd_vel`
topic. This prioritizes reliable movement to world-coordinate goals over
physically accurate wheel torque and skid-steer traction. GPS and odometry
remain enabled for world-position feedback and diagnostics.

Obstacle-aware path planning is enabled by default. `HuskyMotion.py` converts
lidar returns into world coordinates and uses `grid_path_planner.plan_path` to
select and replan paths around detected obstacles. Set
`obstacle_stopping_enabled:=false` only when intentionally testing direct
goal travel without obstacle handling.
The default robot clearance is `0.45 m`; when no route is available from the
current scan, the controller holds position and retries planning instead of
cancelling the goal.
Recent YOLO grass world positions are retained for `5.0` seconds. When the
active goal matches a recent grass position, lidar returns within `0.8 m` of
that grass position are excluded from A* planning, while the raw lidar
front-collision safety check remains active.

The wheel model uses deliberately high wheel friction (`mu1=100.0`,
`mu2=100.0`) as a diagnostic. These values have little effect on the
`VelocityControl` body motion and can be reduced if wheel-driven physics is
not being tested. Restart the simulator after changing wheel, chassis, or
plugin Xacro files; restarting only the controller does not reload the robot
model.

The controller monitors measured odometry progress. If the Husky has not
reduced its distance to the active goal by `minimum_progress` within
`progress_timeout` seconds, it briefly reverses and turns in the opposite
direction before retrying the path. It makes at most two recovery attempts;
if both fail, it cancels the goal instead of reversing indefinitely. Tune or
disable this recovery with
`stuck_recovery_duration`, `stuck_recovery_speed`,
`stuck_recovery_turn_speed`, and `stuck_recovery_enabled`.

## Visual goal manager

Run this alongside `HuskyMotion.py` to view odometry and manage goals:

```bash
python3 HuskyScripts/goal_visualizer.py
```

Left-click appends a goal to a FIFO queue. The visualizer publishes only the
first goal, waits until odometry reaches it, and then publishes the next goal,
so goals are visited in placement order. Right-click or middle-click clears
the active and queued goals and stops the controller. The map uses the
`husky1/odom` frame and displays a fixed 50 m by 50 m area centered on the
Husky's position when the visualizer starts.

## Keyboard teleoperation

For an intuitive physics test, stop any other node publishing
`/husky1/cmd_vel`, then run:

```bash
python3 HuskyScripts/keyboard_teleop.py
```

Hold `W`/Up to drive forward, `S`/Down to reverse, and `A`/Left or `D`/Right
to rotate. Press Space to stop and `Q` or `X` to quit. Commands ramp smoothly
and automatically return to zero when key repeats stop, so releasing a key
does not leave the Husky moving indefinitely. Adjust speeds with
`--linear-speed` and `--angular-speed`.

## Rotation diagnostic

To measure whether the simulator applies the expected sign and magnitude for
`cmd_vel.angular.z`, stop `HuskyMotion.py` first, then run:

```bash
python3 HuskyScripts/diagnose_husky_motion.py
```

The tool sends only angular commands, records odometry yaw and displacement,
and tests positive then negative rotation. A positive command should produce a
positive yaw change. Use this test to determine a command-sign or simulator
frame problem before tuning `angular_gain` or `max_angular_speed`.