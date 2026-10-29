#!/bin/bash

cd "$(dirname "$0")"

source /opt/ros/humble/setup.bash
source install/setup.bash

PYTHON="$PWD/yolo_env/bin/python"
if [[ ! -x "$PYTHON" ]]; then
	echo "Missing Python environment: $PYTHON" >&2
	exit 1
fi

# Pass RandomWorld for a fresh layout on every simulation launch.
WORLD="${1:-worldterrian}"

echo "Starting FARMER simulation ($WORLD)..."

#./run_farmer.bash RandomWorld
#this run it with the random world

# 1. Start Gazebo with Husky + Parrot
env LIBGL_ALWAYS_SOFTWARE=1 ros2 launch \
41068_ignition_bringup 41068_ignition.launch.py \
world:="$WORLD" \
husky:=True \
parrot:=True &

SIM_PID=$!

# Give Gazebo time to start
sleep 8

# 2. Start drone coordinate controller
# Cruise at 2 m/s so survey sweeps and transits finish in reasonable time.
"$PYTHON" DroneStuff/DroneMotion.py --ros-args -p max_speed:=2.0 &

DRONE_PID=$!

# Start Husky coordinate controller
"$PYTHON" HuskyScripts/HuskyMotion.py &

HUSKY_PID=$!

# Remove grass patches as the Husky approaches them.
"$PYTHON" HuskyScripts/grassRemover.py --ros-args -p world_file:="$PWD/world/$WORLD.sdf" &

GRASS_PID=$!

# Battery simulation for the Husky (charging pad at the origin)
"$PYTHON" Mission/battery_node.py &

BATTERY_PID=$!

sleep 2

# 3. Start camera risk detector
#python3 DroneStuff/camera_risk.py &

#CAMERA_RISK_PID=$!

# Start YOLO inference on the drone camera
"$PYTHON" infer_yolo_camera.py &

CAMERA_INFERENCE_PID=$!

# 4. Start FARMER control station
"$PYTHON" UI/farmer_ui.py &

UI_PID=$!

cleanup() {
	trap - INT TERM EXIT
	kill "$UI_PID" "$CAMERA_INFERENCE_PID" "$BATTERY_PID" "$GRASS_PID" "$HUSKY_PID" "$DRONE_PID" "$SIM_PID" 2>/dev/null || true
	wait 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "FARMER launched."
echo "Gazebo PID: $SIM_PID"
echo "Drone controller PID: $DRONE_PID"
echo "Husky controller PID: $HUSKY_PID"
echo "Grass remover PID: $GRASS_PID"
echo "Battery node PID: $BATTERY_PID"
echo "Camera inference PID: $CAMERA_INFERENCE_PID"
echo "UI PID: $UI_PID"

# Keep launcher alive
wait

####### NEED TO INSTALL YOLO ENVIRONMENT #######
# python3 -m venv yolo_env
# source yolo_env/bin/activate
# pip install -r requirements-yolo.txt