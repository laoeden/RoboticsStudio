#!/bin/bash

cd "$(dirname "$0")"

source /opt/ros/humble/setup.bash
source install/setup.bash

echo "Starting FARMER simulation..."

# 1. Start Gazebo with Husky + Parrot
env LIBGL_ALWAYS_SOFTWARE=1 ros2 launch \
41068_ignition_bringup 41068_ignition.launch.py \
world:=worldterrian \
husky:=True \
parrot:=True &

SIM_PID=$!

# Give Gazebo time to start
sleep 8

# 2. Start drone coordinate controller
python3 DroneStuff/DroneMotion.py &

DRONE_PID=$!

# Start Husky coordinate controller
python3 HuskyScripts/HuskyMotion.py &

HUSKY_PID=$!

sleep 2

# 3. Start camera risk detector
python3 DroneStuff/camera_risk.py &

CAMERA_RISK_PID=$!

# 4. Start FARMER control station
python3 UI/farmer_ui.py &

UI_PID=$!

cleanup() {
	trap - INT TERM EXIT
	kill "$UI_PID" "$CAMERA_RISK_PID" "$HUSKY_PID" "$DRONE_PID" "$SIM_PID" 2>/dev/null || true
	wait 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "FARMER launched."
echo "Gazebo PID: $SIM_PID"
echo "Drone controller PID: $DRONE_PID"
echo "Husky controller PID: $HUSKY_PID"
echo "Camera risk PID: $CAMERA_RISK_PID"
echo "UI PID: $UI_PID"

# Keep launcher alive
wait

