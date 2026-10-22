import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from UI.grass_pinpoints import GrassPinpoints
from collections import deque
from HuskyScripts.grid_path_planner import plan_path
from Mission.coverage_planner import CAMERA_DROP, Rect, SurveyConfig, footprint, plan_survey
from Mission.survey_mission import (
    MissionConfig, MissionIO, MissionState, SurveyMission, Telemetry,
)

import tkinter as tk
from tkinter import ttk
from datetime import datetime
import math
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

from geometry_msgs.msg import PointStamped

# =========================================================
# FARMER UI
# Fire Aware Robotic Mower for Environmental Risk
# =========================================================

BG = "#0a0d0a"
PANEL = "#111611"
PANEL_2 = "#171d17"
GREEN = "#8fbf63"
GREEN_BRIGHT = "#b6e67f"
TEXT = "#d7ddd4"
TEXT_DIM = "#7f8b7c"
RED = "#d85c50"
AMBER = "#d3aa55"
BORDER = "#394438"


root = tk.Tk()
root.title("FARMER Control Station")
root.geometry("1600x850")
root.configure(bg=BG)

# =========================================================
# ROS 2 SETUP
# =========================================================

rclpy.init()

ros_node = Node("farmer_control_station")
bridge = CvBridge()
latest_yolo_image = None
frame_stamps = deque(maxlen=60)   # stamps of Parrot camera frames not yet counted as survey coverage
latest_husky_depth = None
depth_frame_count = 0
drone_odom_sample = None
odom_pin_samples = deque(maxlen=100)
last_pinpoint_stamp = None
grass_pinpoints = GrassPinpoints(REPO_ROOT / "UI" / "dry_grass_pinpoints.json")
pinpoint_items = {}
camera_risk_lock = threading.Lock()
camera_risk_wakeup = threading.Event()
camera_risk_stop = threading.Event()
camera_risk_husky_position = None
camera_risk_pending_goals = []
husky_dispatch_enabled = False
camera_risk_active_goal = None
camera_risk_goal_published = 0
camera_risk_goal_tolerance = 0.2
camera_risk_duplicate_radius = 2.0
camera_risk_completed_goals = []
camera_risk_logs = queue.Queue()
yolo_detection_queue = queue.Queue()
MAIN_LIDAR_MAX_POINTS = 150
MAIN_MAP_UPDATE_PERIOD_MS = 500
DEPTH_DISPLAY_PERIOD_MS = 200
DEPTH_DISPLAY_SCALE = 0.5
HUSKY_GRASS_EXCLUSION_RADIUS = 3.5


def message_time(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9


ugv_x = None
ugv_y = None
ugv_heading = None

latest_lidar_scan = None
latest_husky_goal = None

uav_x = None
uav_y = None
uav_z = None
uav_odom_history = []
spiral_active = False
spiral_targets = []
spiral_index = 0
estop_active = False

# Survey mission (see the SURVEY MISSION section below).
survey_mode_active = False      # True while a survey owns the camera-risk pipeline
mission_events = queue.Queue()  # ROS callbacks -> UI thread

def odometry_callback(msg):
    global ugv_x, ugv_y, ugv_heading, camera_risk_husky_position

    ugv_x = msg.pose.pose.position.x
    ugv_y = msg.pose.pose.position.y

    q = msg.pose.pose.orientation

    ugv_heading = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )
    with camera_risk_lock:
        camera_risk_husky_position = (ugv_x, ugv_y)
    camera_risk_wakeup.set()
    
odom_sub = ros_node.create_subscription(
    Odometry,
    "/husky1/odometry",
    odometry_callback,
    10
)

battery_level = None
battery_low_flag = False
charging_flag = False

def battery_callback(msg):
    global battery_level
    battery_level = msg.data

def battery_low_callback(msg):
    global battery_low_flag
    battery_low_flag = msg.data

def charging_callback(msg):
    global charging_flag
    charging_flag = msg.data

battery_sub = ros_node.create_subscription(Float32, "/husky1/battery", battery_callback, 10)
battery_low_sub = ros_node.create_subscription(Bool, "/husky1/battery_low", battery_low_callback, 10)
charging_sub = ros_node.create_subscription(Bool, "/husky1/charging", charging_callback, 10)

def lidar_callback(msg):
    global latest_lidar_scan
    latest_lidar_scan = msg

lidar_sub = ros_node.create_subscription(
    LaserScan,
    "/husky1/scan",
    lidar_callback,
    10
)

def husky_goal_callback(msg):
    global latest_husky_goal
    latest_husky_goal = (
        float(msg.point.x),
        float(msg.point.y)
    )


def husky_obstacle_stop_callback(msg):
    global camera_risk_active_goal, latest_husky_goal
    if not msg.data:
        return
    with camera_risk_lock:
        blocked_goal = camera_risk_active_goal
        camera_risk_active_goal = None
    if blocked_goal is not None:
        if latest_husky_goal == tuple(blocked_goal[:2]):
            latest_husky_goal = None
        camera_risk_logs.put((
            "GOAL",
            f"Skipped blocked Husky goal at "
            f"({blocked_goal[0]:.2f}, {blocked_goal[1]:.2f}); "
            "continuing with next goal.",
        ))
    camera_risk_wakeup.set()


husky_obstacle_stop_sub = ros_node.create_subscription(
        Bool,
        "/husky1/obstacle_stop",
        husky_obstacle_stop_callback,
        10,
)


husky_goal_sub = ros_node.create_subscription(
    PointStamped,
    "/husky1/goal",
    husky_goal_callback,
    10
)

def drone_odometry_callback(msg):
    global uav_x, uav_y, uav_z, drone_odom_sample, uav_odom_history

    uav_x = msg.pose.pose.position.x
    uav_y = msg.pose.pose.position.y
    uav_z = msg.pose.pose.position.z
    drone_odom_sample = (msg, message_time(msg))
    odom_pin_samples.append(drone_odom_sample)

    odom_time = (
        msg.header.stamp.sec
        + msg.header.stamp.nanosec * 1e-9
    )

    uav_odom_history.append(
        (odom_time, uav_x, uav_y, uav_z)
    )

    if len(uav_odom_history) > 200:
        uav_odom_history.pop(0)


drone_odom_sub = ros_node.create_subscription(
    Odometry,
    "/parrot1/odometry",
    drone_odometry_callback,
    10
)


def yolo_world_position_callback(msg):
    yolo_detection_queue.put((
        np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float32),
        msg.header.frame_id,
        message_time(msg),
    ))


yolo_world_position_sub = ros_node.create_subscription(
    PointStamped,
    "/parrot1/yolo/world_position",
    yolo_world_position_callback,
    10,
)


def yolo_image_callback(msg):
    global latest_yolo_image

    try:
        latest_yolo_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception as error:
        ros_node.get_logger().error(
            f"YOLO image conversion failed: {error}"
        )


yolo_image_sub = ros_node.create_subscription(
    RosImage,
    "/parrot1/yolo/image",
    yolo_image_callback,
    10,
)


def husky_depth_callback(msg):
    global latest_husky_depth, depth_frame_count

    try:
        if msg.encoding == "32FC1":
            dtype = np.float32
            scale = 1.0
        elif msg.encoding == "16UC1":
            dtype = np.uint16
            scale = 0.001
        else:
            raise ValueError(
                f"Unsupported depth encoding: {msg.encoding!r}"
            )
        bytes_per_pixel = np.dtype(dtype).itemsize
        row_bytes = msg.width * bytes_per_pixel
        if msg.step < row_bytes:
            raise ValueError(
                f"Invalid depth step {msg.step} for width {msg.width}"
            )
        depth = np.frombuffer(msg.data, dtype=dtype).reshape(
            msg.height, msg.step // bytes_per_pixel
        )[:, :msg.width].astype(np.float32, copy=True)
        latest_husky_depth = depth * scale
        depth_frame_count += 1
    except (TypeError, ValueError) as error:
        ros_node.get_logger().error(
            f"Husky depth conversion failed: {error}",
            throttle_duration_sec=2.0,
        )


def parrot_frame_callback(msg):
    # Only the capture time is needed: the survey marks coverage from the drone pose at
    # that moment, so the image itself is never decoded here (YOLO has its own copy).
    frame_stamps.append(message_time(msg))


parrot_frame_sub = ros_node.create_subscription(
    RosImage,
    "/parrot1/camera/image",
    parrot_frame_callback,
    10,
)


husky_depth_sub = ros_node.create_subscription(
    RosImage,
    "/husky1/camera/depth/image",
    husky_depth_callback,
    10,
)

def hazard_removed_callback(msg):
    mission_events.put(("removed", float(msg.point.x), float(msg.point.y)))


def husky_status_callback(msg):
    mission_events.put(("status", msg.data))


hazard_removed_sub = ros_node.create_subscription(
    PointStamped,
    "/hazard_removed",
    hazard_removed_callback,
    10
)

husky_status_sub = ros_node.create_subscription(
    String,
    "/husky1/goal_status",
    husky_status_callback,
    10
)

def ros_spin():
    rclpy.spin(ros_node)

ros_thread = threading.Thread(
    target=ros_spin,
    daemon=True,
    name="ROS_THREAD"
)

ros_thread.start()


mission_command_pub = ros_node.create_publisher(
    String,
    "/mission_command",
    10
)

drone_goal_pub = ros_node.create_publisher(
    PointStamped,
    "/parrot1/goal",
    10
)

husky_goal_pub = ros_node.create_publisher(
    PointStamped,
    "/husky1/goal",
    10
)


def publish_command(command):
    msg = String()
    msg.data = command
    mission_command_pub.publish(msg)

    ros_node.get_logger().info(f"Published command: {command}")


def publish_drone_goal(x, y, z):
    msg = PointStamped()
    msg.header.frame_id = "parrot1_odom"
    msg.point.x = x
    msg.point.y = y
    msg.point.z = z
    drone_goal_pub.publish(msg)
    ros_node.get_logger().info(f"Sent drone goal: ({x}, {y}, {z})")

def send_drone_goal():
    try:
        x = float(drone_x_entry.get())
        y = float(drone_y_entry.get())
        z = float(drone_z_entry.get())
    except ValueError:
        ros_node.get_logger().error("Drone coordinates must be numbers")
        return

    publish_drone_goal(x, y, z)

def publish_husky_goal(x, y):
    msg = PointStamped()
    msg.header.frame_id = "husky1_odom"
    msg.point.x = x
    msg.point.y = y
    msg.point.z = 0.0

    husky_goal_pub.publish(msg)

    ros_node.get_logger().info(
        f"Sent Husky goal: ({x}, {y})"
    )


def send_husky_goal():
    try:
        x = float(husky_x_entry.get())
        y = float(husky_y_entry.get())
    except ValueError:
        ros_node.get_logger().error("Husky coordinates must be numbers")
        return

    publish_husky_goal(x, y)


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def add_log(source, message):
    log.config(state="normal")
    log.insert(tk.END, f"[{timestamp()}] [{source}] {message}\n")
    log.see(tk.END)
    log.config(state="disabled")

def start_mission():
    global spiral_active, spiral_targets, spiral_index

    if mission.active:
        add_log("SYSTEM", "A survey is running; abort it before starting the spiral")
        return

    pause_husky_dispatch()
    publish_command("STOP")
    publish_command("START")

################ SPIRAL ##################
    spiral_center_x = 0.0 # this can later be adjusted through the UI or mission parameters
    spiral_center_y = 0.0

    spiral_radius = 15
    spiral_expainsion_rate = 0.7
    spiral_steps = round(spiral_radius / spiral_expainsion_rate)

    spiral_targets = [
        (
            spiral_center_x + 0.7 * step * math.cos(step * math.pi / 4),
            spiral_center_y + 0.7 * step * math.sin(step * math.pi / 4),
            10.0,
        )
        for step in range(spiral_steps)
    ]
    spiral_index = 0
    spiral_active = True
    publish_drone_goal(*spiral_targets[spiral_index])

    status_value.config(text="ACTIVE", fg=GREEN_BRIGHT)
    mission_value.config(text="EXECUTING")
    add_log("SYSTEM", "MISSION STARTED")


def pause_husky_dispatch():
    """Keep collected targets for a later manual start."""
    global husky_dispatch_enabled, camera_risk_active_goal
    with camera_risk_lock:
        husky_dispatch_enabled = False
        if camera_risk_active_goal is not None:
            camera_risk_pending_goals.insert(0, camera_risk_active_goal)
            camera_risk_active_goal = None


def start_husky():
    """Release collected coordinates independently of the detection source."""
    global husky_dispatch_enabled
    if estop_active:
        return
    if mission.active:
        add_log("UGV", "A survey is running; abort it before starting the Husky")
        return
    if spiral_active:
        add_log("UGV", "Wait for the drone survey to finish before starting Husky")
        return
    with camera_risk_lock:
        if husky_dispatch_enabled:
            return
        if not camera_risk_pending_goals:
            add_log("UGV", "No collected coordinates available yet")
            return
        husky_dispatch_enabled = True
        target_count = len(camera_risk_pending_goals)
    camera_risk_wakeup.set()
    status_value.config(text="ACTIVE", fg=GREEN_BRIGHT)
    mission_value.config(text="HUSKY EXECUTING")
    add_log("UGV", f"Husky started with {target_count} queued coordinates")


def stop_mission():
    global spiral_active

    mission.abort("operator stop")
    pause_husky_dispatch()
    publish_command("STOP")
    spiral_active = False

    status_value.config(text="HALTED", fg=RED)
    mission_value.config(text="STOPPED")
    add_log("SYSTEM", "MISSION STOPPED")


def return_to_base():
    global spiral_active

    mission.abort("return to base requested")
    pause_husky_dispatch()
    publish_command("RETURN_TO_BASE")
    spiral_active = False

    publish_drone_goal(0.0, 0.0, 10.0)

    status_value.config(text="RTB", fg=AMBER)
    mission_value.config(text="RETURN TO BASE")
    add_log("UGV", "RETURN TO BASE COMMAND SENT")


def emergency_stop():
    global estop_active, spiral_active

    estop_active = not estop_active

    controls = [
        start_button,
        start_husky_button,
        stop_button,
        rtb_button,
        send_goal_button,
        drone_x_entry,
        drone_y_entry,
        drone_z_entry,
        send_husky_goal_button,
        husky_x_entry,
        husky_y_entry
    ]

    if estop_active:
        pause_husky_dispatch()
        publish_command("EMERGENCY_STOP")
        spiral_active = False
        set_draw_mode(False)
        mission.abort("emergency stop", hover=False)

        for control in controls:
            control.config(state="disabled")

        status_value.config(text="E-STOP", fg=RED)
        mission_value.config(text="EMERGENCY STOP")

        estop_button.config(text="[ RESET E-STOP ]")

        add_log("SYSTEM", "EMERGENCY STOP ACTIVATED")

    else:
        publish_command("EMERGENCY_STOP_RESET")

        for control in controls:
            control.config(state="normal")

        status_value.config(text="READY", fg=GREEN_BRIGHT)
        mission_value.config(text="STANDBY")

        estop_button.config(text="[ EMERGENCY STOP ]")

        add_log("SYSTEM", "EMERGENCY STOP RESET")

# =========================================================
# SURVEY MISSION
# Draw an area -> drone surveys it -> Husky investigates and
# clears the POIs -> both return to base.
# The logic lives in Mission/ (coverage_planner, survey_mission);
# this section only connects it to ROS and the display.
# =========================================================

WORLD_HALF_EXTENT = 25.0          # the terrain mesh spans +/-25 m
WORLD_RECT = Rect(
    -WORLD_HALF_EXTENT, -WORLD_HALF_EXTENT,
    WORLD_HALF_EXTENT, WORLD_HALF_EXTENT,
)
MIN_SURVEY_SIDE = 4.0             # smallest area that can be drawn (m)
SURVEY_CFG = SurveyConfig()
# Hover a little longer than the default after the sweep: YOLO inference on a software-rendered
# simulation can lag the camera by a few seconds, and late sightings still need to become pins.
MISSION_CFG = MissionConfig(survey=SURVEY_CFG, settle_time=6.0)

POI_COLOURS = {
    "pending": AMBER,
    "active": "#5fb3d8",
    "cleared": GREEN_BRIGHT,
    "false_positive": TEXT_DIM,
    "unreachable": RED,
    "visited": GREEN,
}
POI_TAGS = {
    "active": "VISITING",
    "cleared": "CLEARED",
    "false_positive": "FALSE POSITIVE",
    "unreachable": "UNREACHABLE",
    "visited": "VISITED",
}

survey_area = None                # Rect chosen on the map, or None
survey_plan = None                # preview of the sweep for that area
survey_drag = {"start": None}
draw_mode_on = False
square_lock = tk.BooleanVar(value=True)
coverage_overlay = {"photo": None, "version": -1, "drawn_at": 0.0}
last_mission_state = None

map_view = {"ox": 405.0, "oy": 315.0, "scale": 10.0}


def world_to_map(x, y):
    """Convert ROS world coordinates (m) to environment display pixels."""
    return (
        map_view["ox"] + x * map_view["scale"],
        map_view["oy"] - y * map_view["scale"],
    )


def map_to_world(px, py):
    return (
        (px - map_view["ox"]) / map_view["scale"],
        (map_view["oy"] - py) / map_view["scale"],
    )


class UiMissionIO(MissionIO):
    """Connects SurveyMission to the ROS publishers, pins and event log."""

    def prepare_survey(self, area):
        global survey_mode_active, camera_risk_active_goal, husky_dispatch_enabled
        with camera_risk_lock:
            # Earlier results in this area are re-detected by the new survey.
            grass_pinpoints.discard_for_resurvey(area.bounds)
            camera_risk_pending_goals.clear()
            camera_risk_completed_goals.clear()
            camera_risk_active_goal = None
            husky_dispatch_enabled = False
        survey_mode_active = True        # stop the camera tab dispatching the Husky
        publish_command("CLEAR_GOAL")
        auto_pinpoints.set(True)

    def end_survey(self):
        global survey_mode_active
        survey_mode_active = False

    def drone_goal(self, x, y, z):
        publish_drone_goal(float(x), float(y), float(z))

    def husky_goal(self, x, y):
        publish_husky_goal(float(x), float(y))

    def command(self, text):
        publish_command(text)

    def log(self, source, message):
        add_log(source, message)

    def list_pois(self, area):
        with camera_risk_lock:
            return [
                {"id": pin["id"], "x": pin["x"], "y": pin["y"]}
                for pin in grass_pinpoints.open_in(area.bounds)
            ]

    def mark_poi(self, poi_id, status):
        with camera_risk_lock:
            grass_pinpoints.set_status(poi_id, status)


mission = SurveyMission(UiMissionIO(), MISSION_CFG)


def mission_telemetry():
    drone = None
    if uav_x is not None and uav_y is not None and uav_z is not None:
        drone = (uav_x, uav_y, uav_z)
    husky = None
    if ugv_x is not None and ugv_y is not None:
        husky = (ugv_x, ugv_y)
    try:
        remover_online = ros_node.count_publishers("/hazard_removed") > 0
    except Exception:       # ROS is shutting down (Ctrl+C): the window is about to close anyway
        remover_online = False
    return Telemetry(
        drone=drone,
        husky=husky,
        battery=battery_level,
        battery_low=battery_low_flag,
        remover_online=remover_online,
    )


def format_duration(seconds):
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


# ---------- map drawing ----------

def stack_map_layers():
    """Static map at the bottom, then coverage, then the survey path."""
    for tag in ("survey_overlay", "coverage_overlay", "map_static"):
        try:
            canvas.tag_lower(tag)
        except tk.TclError:
            pass


def redraw_map_static():
    canvas.delete("map_static")
    half = WORLD_HALF_EXTENT
    for i in range(-int(half), int(half) + 1, 5):
        colour = "#2a3a28" if i == 0 else "#1b241a"
        canvas.create_line(
            *world_to_map(i, -half), *world_to_map(i, half),
            fill=colour, tags="map_static",
        )
        canvas.create_line(
            *world_to_map(-half, i), *world_to_map(half, i),
            fill=colour, tags="map_static",
        )
        if i % 10 == 0 and i != 0:
            canvas.create_text(
                *world_to_map(i, -half - 1.6), text=f"{i}",
                fill=TEXT_DIM, font=("DejaVu Sans Mono", 7), tags="map_static",
            )
            canvas.create_text(
                *world_to_map(-half - 1.8, i), text=f"{i}",
                fill=TEXT_DIM, font=("DejaVu Sans Mono", 7), tags="map_static",
            )
    canvas.create_rectangle(
        *world_to_map(-half, half), *world_to_map(half, -half),
        outline=GREEN, dash=(8, 5), width=2, tags="map_static",
    )
    # Charging pad / base at the origin (2 m x 2 m).
    canvas.create_rectangle(
        *world_to_map(-1, 1), *world_to_map(1, -1),
        outline="#5fb3d8", width=2, tags="map_static",
    )
    canvas.create_text(
        *world_to_map(0, 2.2), text="BASE",
        fill="#5fb3d8", font=("DejaVu Sans Mono", 8), tags="map_static",
    )
    canvas.create_text(
        15, 15, text="GRID REF: FIELD_A1", anchor="nw",
        fill=TEXT_DIM, font=("DejaVu Sans Mono", 8), tags="map_static",
    )
    stack_map_layers()


def on_map_resize(event):
    if event.width < 100 or event.height < 100:
        return
    margin = 28
    scale = min(event.width - 2 * margin, event.height - 2 * margin) / (
        2 * WORLD_HALF_EXTENT
    )
    scale = max(scale, 1.0)
    # Centred across, but pinned to the top: the MAIN tab can be taller than the
    # window, and the map must stay in view without scrolling.
    map_view.update(
        ox=event.width / 2.0,
        oy=margin + WORLD_HALF_EXTENT * scale,
        scale=scale,
    )
    redraw_map_static()
    draw_survey_overlay()
    coverage_overlay["version"] = -1


def current_plan():
    """The mission's own plan once it has started, otherwise the preview."""
    if mission.plan is not None and mission.area == survey_area:
        return mission.plan
    return survey_plan


def draw_survey_overlay():
    canvas.delete("survey_overlay")
    area = survey_area
    if area is None:
        return
    x0, y0 = world_to_map(area.xmin, area.ymax)
    x1, y1 = world_to_map(area.xmax, area.ymin)
    canvas.create_rectangle(
        x0, y0, x1, y1, outline=AMBER, width=2, dash=(6, 3),
        tags="survey_overlay",
    )
    canvas.create_text(
        x0 + 4, y0 + 4, anchor="nw", fill=AMBER,
        font=("DejaVu Sans Mono", 8, "bold"),
        text=f"SURVEY AREA {area.width:.0f} x {area.height:.0f} m",
        tags="survey_overlay",
    )
    plan = current_plan()
    if plan is not None and plan.waypoints:
        coords = []
        for waypoint in plan.waypoints:
            coords.extend(world_to_map(waypoint[0], waypoint[1]))
        if len(coords) >= 4:
            canvas.create_line(
                *coords, fill=GREEN_BRIGHT, width=2, arrow=tk.LAST,
                tags="survey_overlay",
            )
        for index, waypoint in enumerate(plan.waypoints):
            x, y = world_to_map(waypoint[0], waypoint[1])
            canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill=GREEN_BRIGHT, outline="",
                tags="survey_overlay",
            )
        x, y = world_to_map(plan.waypoints[0][0], plan.waypoints[0][1])
        canvas.create_text(
            x + 8, y - 8, text="START", fill=GREEN_BRIGHT,
            font=("DejaVu Sans Mono", 7, "bold"), tags="survey_overlay",
        )
    stack_map_layers()


def draw_survey_drag(rect):
    canvas.delete("survey_drag")
    x0, y0 = world_to_map(rect.xmin, rect.ymax)
    x1, y1 = world_to_map(rect.xmax, rect.ymin)
    canvas.create_rectangle(
        x0, y0, x1, y1, outline="#5fb3d8", width=2, dash=(4, 3),
        tags="survey_drag",
    )
    canvas.create_text(
        x0 + 4, y0 + 4, anchor="nw", fill="#5fb3d8",
        font=("DejaVu Sans Mono", 8, "bold"),
        text=f"{rect.width:.1f} x {rect.height:.1f} m", tags="survey_drag",
    )


def draw_coverage_overlay():
    """Tint the survey area: green where the camera has been, red where not."""
    snapshot = mission.coverage_snapshot()
    if snapshot is None or survey_area is None:
        return
    version, seen, area = snapshot
    if area != survey_area:
        return
    now = time.monotonic()
    if version == coverage_overlay["version"]:
        return
    if now - coverage_overlay["drawn_at"] < 0.5:
        return
    coverage_overlay["version"] = version
    coverage_overlay["drawn_at"] = now
    x0, y0 = world_to_map(area.xmin, area.ymax)
    x1, y1 = world_to_map(area.xmax, area.ymin)
    width = max(1, int(round(x1 - x0)))
    height = max(1, int(round(y1 - y0)))
    rgba = np.zeros(seen.shape + (4,), dtype=np.uint8)
    rgba[seen] = (143, 191, 99, 90)
    rgba[~seen] = (216, 92, 80, 45)
    image = PILImage.fromarray(rgba).resize((width, height), PILImage.NEAREST)
    photo = ImageTk.PhotoImage(image)
    canvas.delete("coverage_overlay")
    canvas.create_image(
        x0, y0, anchor="nw", image=photo, tags="coverage_overlay"
    )
    coverage_overlay["photo"] = photo     # keep a reference or Tk drops the image
    stack_map_layers()


# ---------- drawing the area ----------

def set_draw_mode(enabled):
    global draw_mode_on
    draw_mode_on = enabled
    draw_area_button.config(
        text="[ CANCEL DRAWING ]" if enabled else "[ DRAW AREA ]"
    )
    canvas.config(cursor="crosshair" if enabled else "")
    survey_drag["start"] = None
    canvas.delete("survey_drag")
    if enabled:
        survey_status_label.config(
            text="Drag on the environment display to draw the survey area."
        )


def toggle_draw_mode():
    if mission.active or estop_active:
        return
    set_draw_mode(not draw_mode_on)


def drag_rect(event):
    """Rectangle from the press point to the pointer, optionally forced square."""
    start_x, start_y = survey_drag["start"]
    half = WORLD_HALF_EXTENT
    end_x, end_y = map_to_world(event.x, event.y)
    end_x = min(max(end_x, -half), half)
    end_y = min(max(end_y, -half), half)
    if square_lock.get():
        dx, dy = end_x - start_x, end_y - start_y
        side = max(abs(dx), abs(dy))
        sign_x = 1.0 if dx >= 0 else -1.0
        sign_y = 1.0 if dy >= 0 else -1.0
        room_x = (half - start_x) if sign_x > 0 else (start_x + half)
        room_y = (half - start_y) if sign_y > 0 else (start_y + half)
        side = min(side, room_x, room_y)
        end_x, end_y = start_x + sign_x * side, start_y + sign_y * side
    return Rect.from_corners(start_x, start_y, end_x, end_y)


def on_area_press(event):
    if not draw_mode_on or mission.active:
        return
    x, y = map_to_world(event.x, event.y)
    if not WORLD_RECT.contains(x, y):
        return
    survey_drag["start"] = (x, y)


def on_area_drag(event):
    if survey_drag["start"] is None:
        return
    draw_survey_drag(drag_rect(event))


def on_area_release(event):
    if survey_drag["start"] is None:
        return
    rect = drag_rect(event)
    canvas.delete("survey_drag")
    survey_drag["start"] = None
    if rect.width < MIN_SURVEY_SIDE or rect.height < MIN_SURVEY_SIDE:
        survey_status_label.config(
            text=f"Area too small: each side must be at least {MIN_SURVEY_SIDE:.0f} m."
        )
        return
    set_survey_area(rect)
    set_draw_mode(False)


def set_survey_area(rect):
    global survey_area, survey_plan
    survey_area = rect
    start = None
    if uav_x is not None and uav_y is not None:
        start = (uav_x, uav_y)
    survey_plan = plan_survey(rect, SURVEY_CFG, start)
    coverage_overlay["version"] = -1
    canvas.delete("coverage_overlay")
    draw_survey_overlay()
    plan = survey_plan
    add_log(
        "SURVEY",
        f"Area {rect.width:.0f} x {rect.height:.0f} m selected: {plan.rows} rows, "
        f"{len(plan.waypoints)} waypoints, altitude {plan.flight_z:.1f} m, "
        f"ETA {format_duration(plan.eta_s)}",
    )
    for warning in plan.warnings:
        add_log("SURVEY", f"WARNING: {warning}")
    survey_status_label.config(
        text="Area ready. Press SURVEY + INVESTIGATE to start."
    )


def clear_survey_area():
    global survey_area, survey_plan
    if mission.active:
        return
    survey_area = None
    survey_plan = None
    set_draw_mode(False)
    canvas.delete("survey_overlay")
    canvas.delete("coverage_overlay")
    coverage_overlay["version"] = -1
    survey_status_label.config(text="Draw an area on the map to begin.")
    add_log("SURVEY", "Survey area cleared")


# ---------- starting and stopping ----------

def start_survey():
    if estop_active:
        add_log("SURVEY", "Cannot start: emergency stop is active")
        return
    if mission.active:
        return
    if survey_area is None:
        add_log("SURVEY", "Draw a survey area first")
        return
    if spiral_active:
        add_log("SURVEY", "Stop the spiral mission before starting a survey")
        return
    if uav_x is None or uav_y is None or uav_z is None:
        add_log("SURVEY", "Waiting for drone odometry")
        return
    husky = (ugv_x, ugv_y) if ugv_x is not None and ugv_y is not None else None
    set_draw_mode(False)
    if mission.start(survey_area, time.monotonic(), (uav_x, uav_y, uav_z), husky):
        coverage_overlay["version"] = -1
        draw_survey_overlay()
        refresh_survey_labels()


def abort_survey():
    mission.abort("operator abort")


# ---------- keeping the display up to date ----------

def refresh_survey_labels():
    plan = current_plan() if survey_area is not None else None
    if survey_area is None or plan is None:
        area_label.config(text="NONE")
        eta_label.config(text="--:--")
        waypoints_label.config(text="--")
        return
    area_label.config(text=f"{survey_area.width:.0f}x{survey_area.height:.0f} m")
    eta_label.config(text=format_duration(plan.eta_s))
    waypoints_label.config(text=str(len(plan.waypoints)))


def refresh_survey_ui():
    global last_mission_state
    status = mission.status()
    state = status["state"]
    active = mission.active
    blocked = estop_active
    idle_ok = not active and not blocked
    draw_area_button.config(state="normal" if idle_ok else "disabled")
    clear_area_button.config(
        state="normal" if idle_ok and survey_area is not None else "disabled"
    )
    survey_start_button.config(
        state="normal" if idle_ok and survey_area is not None else "disabled"
    )
    survey_abort_button.config(state="normal" if active else "disabled")

    if state != last_mission_state:
        last_mission_state = state
        if not estop_active:
            if active:
                status_value.config(text="SURVEY", fg=GREEN_BRIGHT)
            elif state == MissionState.COMPLETE:
                status_value.config(text="READY", fg=GREEN_BRIGHT)
            elif state == MissionState.ABORTED:
                status_value.config(text="HALTED", fg=RED)
        if state != MissionState.IDLE:
            mission_value.config(text=state.value)
        draw_survey_overlay()
        refresh_survey_labels()

    progress_label.config(text=f"{status['progress'] * 100:.0f}%")
    if status["pois_found"]:
        hazards_label.config(
            text=f"{status['pois_cleared']}/{status['pois_total']}"
        )
    elif state == MissionState.IDLE:
        hazards_label.config(text="--")
    mode_label.config(text="SURVEY/INVESTIGATE" if active else "STANDBY")

    if state != MissionState.IDLE:
        lines = [state.value]
        if state in (MissionState.TAKEOFF, MissionState.SWEEP,
                     MissionState.GAP_FILL, MissionState.SETTLE):
            lines.append(
                f"Seen {status['coverage'] * 100:.0f}% | waypoint "
                f"{min(status['waypoint'] + 1, max(status['waypoints'], 1))}"
                f"/{status['waypoints']}"
            )
        elif status["pois_found"]:
            lines.append(
                f"POIs: {status['pois_cleared']} cleared, "
                f"{status['pois_false']} false, "
                f"{status['pois_unreachable']} unreachable, "
                f"{status['pois_open']} to go"
            )
        if status["reason"]:
            lines.append(status["reason"])
        survey_status_label.config(text="\n".join(lines))

    if active or state != MissionState.IDLE:
        draw_coverage_overlay()


def service_mission():
    """Run once per UI tick: feed ROS events to the mission and step it."""
    while True:
        try:
            event = mission_events.get_nowait()
        except queue.Empty:
            break
        if event[0] == "removed":
            mission.on_hazard_removed(event[1], event[2])
        else:
            mission.on_husky_status(event[1])
    mission.update(time.monotonic(), mission_telemetry())
    refresh_survey_ui()


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    root,
    bg=PANEL,
    highlightbackground=BORDER,
    highlightthickness=1
)
header.pack(fill="x", padx=8, pady=8)

title_frame = tk.Frame(header, bg=PANEL)
title_frame.pack(side="left", padx=15, pady=10)

tk.Label(
    title_frame,
    text="FARMER",
    bg=PANEL,
    fg=TEXT,
    font=("DejaVu Sans Mono", 26, "bold")
).pack(anchor="w")

tk.Label(
    title_frame,
    text="FIRE AWARE ROBOTIC MOWER // ENVIRONMENTAL RISK PLATFORM",
    bg=PANEL,
    fg=GREEN,
    font=("DejaVu Sans Mono", 9)
).pack(anchor="w")

status_header = tk.Frame(header, bg=PANEL)
status_header.pack(side="right", padx=20)

tk.Label(
    status_header,
    text="LINK: ONLINE",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="e")

tk.Label(
    status_header,
    text="ROS2 HUMBLE",
    bg=PANEL,
    fg=TEXT_DIM,
    font=("DejaVu Sans Mono", 9)
).pack(anchor="e")

# =========================================================
# TABBED INTERFACE
# =========================================================

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))

main_tab = tk.Frame(notebook, bg=BG)
lidar_tab = tk.Frame(notebook, bg=BG)
camera_risk_tab = tk.Frame(notebook, bg=BG)

notebook.add(main_tab, text="MAIN")
notebook.add(lidar_tab, text="HUSKY LIDAR")
notebook.add(camera_risk_tab, text="CAMERA RISK")

# =========================================================
# HUSKY LIDAR TAB
# =========================================================

LIDAR_MAX_RANGE = 15.0
LIDAR_COLLISION_DISTANCE = 1.0
LIDAR_FRONT_ANGLE = math.radians(45.0)

lidar_figure = Figure(
    figsize=(13, 6),
    dpi=100
)

lidar_axis = lidar_figure.add_subplot(
    121,
    projection="polar"
)

lidar_map_axis = lidar_figure.add_subplot(122)

# -------------------------
# Polar lidar display
# -------------------------

lidar_axis.set_theta_zero_location("E")
lidar_axis.set_theta_direction(1)

lidar_axis.set_title(
    "Husky 360 lidar | front +/-45 deg | collision distance 1.0 m"
)

lidar_axis.set_rmax(LIDAR_MAX_RANGE)
lidar_axis.set_rlabel_position(135)

sector_angles = np.linspace(
    -LIDAR_FRONT_ANGLE,
    LIDAR_FRONT_ANGLE,
    100
)

lidar_axis.fill_between(
    sector_angles,
    0.0,
    LIDAR_COLLISION_DISTANCE,
    color="red",
    alpha=0.18,
    label="collision zone"
)

lidar_axis.plot(
    sector_angles,
    np.full_like(
        sector_angles,
        LIDAR_COLLISION_DISTANCE
    ),
    color="red",
    linewidth=1.5,
    label="1.0 m boundary"
)

lidar_axis.plot(
    [0.0, 0.0],
    [0.0, LIDAR_MAX_RANGE],
    color="black",
    linewidth=1
)

lidar_points, = lidar_axis.plot(
    [],
    [],
    ".",
    color="steelblue",
    markersize=3,
    label="lidar"
)

lidar_front_points, = lidar_axis.plot(
    [],
    [],
    ".",
    color="darkorange",
    markersize=5,
    label="front sector"
)

lidar_axis.legend(
    loc="upper right",
    bbox_to_anchor=(1.25, 1.1)
)

# -------------------------
# World map display
# -------------------------

lidar_map_axis.set_title(
    "Current lidar obstacles and planned path"
)

lidar_map_axis.set_xlabel("world X (m)")
lidar_map_axis.set_ylabel("world Y (m)")

lidar_map_axis.set_aspect(
    "equal",
    adjustable="datalim"
)

lidar_map_axis.grid(
    True,
    alpha=0.3
)

lidar_map_points, = lidar_map_axis.plot(
    [],
    [],
    ".",
    color="slateblue",
    markersize=4,
    label="current lidar"
)

lidar_goal_points, = lidar_map_axis.plot(
    [],
    [],
    "x",
    color="crimson",
    markersize=10,
    label="latest goal"
)

lidar_path_points, = lidar_map_axis.plot(
    [],
    [],
    "-",
    color="seagreen",
    linewidth=2,
    label="planned path"
)

lidar_robot_point, = lidar_map_axis.plot(
    [],
    [],
    "o",
    color="black",
    markersize=7,
    label="Husky"
)

lidar_robot_heading, = lidar_map_axis.plot(
    [],
    [],
    color="black",
    linewidth=2
)

lidar_grass_exclusion = Circle(
    (0.0, 0.0),
    HUSKY_GRASS_EXCLUSION_RADIUS,
    fill=False,
    edgecolor="#d3aa55",
    linestyle="--",
    linewidth=2,
    label="Grass exclusion",
    visible=False,
)
lidar_map_axis.add_patch(lidar_grass_exclusion)

lidar_status = lidar_figure.text(
    0.02,
    0.02,
    "Waiting for /husky1/scan..."
)

lidar_map_axis.legend(
    loc="upper right"
)

lidar_canvas = FigureCanvasTkAgg(
    lidar_figure,
    master=lidar_tab
)

lidar_canvas.get_tk_widget().pack(
    fill="both",
    expand=True,
    padx=8,
    pady=8
)

lidar_figure.tight_layout(
    rect=[0.0, 0.04, 1.0, 1.0]
)

lidar_canvas.draw()

# =========================================================
# CAMERA RISK TAB
# =========================================================

camera_risk_main = tk.Frame(camera_risk_tab, bg=BG)
camera_risk_main.pack(fill="both", expand=True, padx=10, pady=10)

camera_risk_main.grid_columnconfigure(0, weight=1)
camera_risk_main.grid_columnconfigure(1, weight=0)
camera_risk_main.grid_rowconfigure(0, weight=1)

camera_risk_display = tk.Label(
    camera_risk_main,
    text="CAMERA RISK FEED WAITING",
    bg="#050805",
    fg=TEXT_DIM,
    font=("DejaVu Sans Mono", 12)
)
camera_risk_display.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 10)
)

camera_risk_controls = tk.Frame(
    camera_risk_main,
    bg=PANEL,
    width=320,
    highlightbackground=BORDER,
    highlightthickness=1
)
camera_risk_controls.grid(
    row=0,
    column=1,
    sticky="ns"
)
camera_risk_controls.grid_propagate(False)

tk.Label(
    camera_risk_controls,
    text="YOLO DETECTION CONTROLS",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 11, "bold")
).pack(anchor="w", padx=15, pady=15)

auto_pinpoints = tk.BooleanVar(value=True)
tk.Checkbutton(
    camera_risk_controls, text="Automatically pin dry grass", variable=auto_pinpoints,
    bg=PANEL, fg=TEXT, selectcolor=PANEL_2, activebackground=PANEL,
    activeforeground=TEXT,
).pack(anchor="w", padx=15, pady=(8, 0))
pinpoint_status = tk.Label(
    camera_risk_controls, text=f"Saved grass pins: {len(grass_pinpoints.pins)}",
    bg=PANEL, fg=AMBER, wraplength=280, justify="left",
)
pinpoint_status.pack(anchor="w", padx=15, pady=5)

# =========================================================
# MAIN AREA
# =========================================================

# Scrollable MAIN tab
main_canvas = tk.Canvas(
    main_tab,
    bg=BG,
    highlightthickness=0
)

main_scrollbar = tk.Scrollbar(
    main_tab,
    orient="vertical",
    command=main_canvas.yview
)

main = tk.Frame(
    main_canvas,
    bg=BG
)

main_window = main_canvas.create_window(
    (0, 0),
    window=main,
    anchor="nw"
)

main_canvas.configure(
    yscrollcommand=main_scrollbar.set
)

main_canvas.pack(
    side="left",
    fill="both",
    expand=True
)

main_scrollbar.pack(
    side="right",
    fill="y"
)

def update_main_scroll_region(event=None):
    main_canvas.configure(
        scrollregion=main_canvas.bbox("all")
    )

def resize_main_width(event):
    main_canvas.itemconfigure(
        main_window,
        width=event.width
    )

main.bind(
    "<Configure>",
    update_main_scroll_region
)

main_canvas.bind(
    "<Configure>",
    resize_main_width
)

main.grid_columnconfigure(1, weight=1)
main.grid_rowconfigure(0, weight=1)

def scroll_main(event):
    main_canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )

def scroll_main_up(event):
    main_canvas.yview_scroll(-1, "units")

def scroll_main_down(event):
    main_canvas.yview_scroll(1, "units")

main_canvas.bind_all("<MouseWheel>", scroll_main)
main_canvas.bind_all("<Button-4>", scroll_main_up)
main_canvas.bind_all("<Button-5>", scroll_main_down)


# =========================================================
# LEFT PANEL
# =========================================================

left = tk.Frame(
    main,
    bg=PANEL,
    width=250,
    highlightbackground=BORDER,
    highlightthickness=1
)
left.grid(row=0, column=0, sticky="ns", padx=(0, 5))

tk.Label(
    left,
    text="SYSTEM STATUS",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 11, "bold")
).pack(anchor="w", padx=15, pady=(15, 10))


status_box = tk.Frame(
    left,
    bg=PANEL_2,
    highlightbackground=BORDER,
    highlightthickness=1
)
status_box.pack(fill="x", padx=12, pady=5)

tk.Label(
    status_box,
    text="SYSTEM STATUS",
    bg=PANEL_2,
    fg=TEXT_DIM,
    font=("DejaVu Sans Mono", 8)
).pack(anchor="w", padx=10, pady=(8, 0))

status_value = tk.Label(
    status_box,
    text="READY",
    bg=PANEL_2,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 18, "bold")
)
status_value.pack(anchor="w", padx=10, pady=(0, 8))


def info_row(parent, label, value):
    row = tk.Frame(parent, bg=PANEL)
    row.pack(fill="x", padx=15, pady=6)

    tk.Label(
        row,
        text=label,
        bg=PANEL,
        fg=TEXT_DIM,
        font=("DejaVu Sans Mono", 9)
    ).pack(side="left")

    value_label = tk.Label(
        row,
        text=value,
        bg=PANEL,
        fg=TEXT,
        font=("DejaVu Sans Mono", 9, "bold")
    )
    value_label.pack(side="right")

    return value_label


battery_label = info_row(left, "BATTERY", "-- %")
progress_label = info_row(left, "PROGRESS", "0%")
hazards_label = info_row(left, "HAZARDS", "--")
info_row(left, "LOCALISATION", "FIXED")
info_row(left, "LINK", "GOOD")

ugv_x_label = info_row(left, "UGV X", "-- m")
ugv_y_label = info_row(left, "UGV Y", "-- m")
uav_x_label = info_row(left, "UAV X", "-- m")
uav_y_label = info_row(left, "UAV Y", "-- m")
uav_z_label = info_row(left, "UAV Z", "-- m")


tk.Frame(left, bg=BORDER, height=1).pack(
    fill="x",
    padx=12,
    pady=12
)

tk.Label(
    left,
    text="MISSION STATE",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", padx=15)

mission_value = tk.Label(
    left,
    text="STANDBY",
    bg=PANEL,
    fg=TEXT,
    font=("DejaVu Sans Mono", 10)
)
mission_value.pack(anchor="w", padx=15, pady=(3, 15))


def tactical_button(text, command, colour=TEXT):
    button = tk.Button(
        left,
        text=text,
        command=command,
        bg=PANEL_2,
        fg=colour,
        activebackground=GREEN,
        activeforeground="black",
        relief="flat",
        bd=0,
        highlightbackground=BORDER,
        highlightthickness=1,
        font=("DejaVu Sans Mono", 9, "bold")
    )
    button.pack(fill="x", padx=12, pady=5, ipady=8)
    return button


start_button = tactical_button("[ START DRONE M ]", start_mission, GREEN_BRIGHT)
start_husky_button = tactical_button("[ START HUSKY M]", start_husky, GREEN_BRIGHT)
stop_button = tactical_button("[ STOP ]", stop_mission, AMBER)
rtb_button = tactical_button("[ RETURN TO BASE ]", return_to_base)
estop_button = tactical_button("[ EMERGENCY STOP ]", emergency_stop, RED)

tk.Frame(left, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(10, 6))

tk.Label(
    left,
    text="SURVEY AREA",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", padx=15)

draw_area_button = tactical_button("[ DRAW AREA ]", toggle_draw_mode, GREEN_BRIGHT)

tk.Checkbutton(
    left, text="Lock to square", variable=square_lock,
    bg=PANEL, fg=TEXT, selectcolor=PANEL_2, activebackground=PANEL,
    activeforeground=TEXT,
).pack(anchor="w", padx=15)

clear_area_button = tactical_button("[ CLEAR AREA ]", clear_survey_area)
survey_start_button = tactical_button(
    "[ SURVEY + INVESTIGATE ]", start_survey, GREEN_BRIGHT
)
survey_abort_button = tactical_button("[ ABORT SURVEY ]", abort_survey, AMBER)

survey_status_label = tk.Label(
    left,
    text="Draw an area on the map to begin.",
    bg=PANEL,
    fg=TEXT_DIM,
    wraplength=220,
    justify="left",
    font=("DejaVu Sans Mono", 8)
)
survey_status_label.pack(anchor="w", padx=15, pady=(2, 10))

drone_target_frame = tk.Frame(left, bg=PANEL)
drone_target_frame.pack(fill="x", padx=15, pady=10)

tk.Label(
    drone_target_frame,
    text="DRONE TARGET",
    bg=PANEL,
    fg=TEXT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", pady=(0, 5))

drone_x_entry = tk.Entry(drone_target_frame, width=7)
drone_x_entry.pack(side="left", padx=2)
drone_x_entry.insert(0, "0")

drone_y_entry = tk.Entry(drone_target_frame, width=7)
drone_y_entry.pack(side="left", padx=2)
drone_y_entry.insert(0, "0")

drone_z_entry = tk.Entry(drone_target_frame, width=7)
drone_z_entry.pack(side="left", padx=2)
drone_z_entry.insert(0, "2")

send_goal_button = tk.Button(
    drone_target_frame,
    text="SEND GOAL",
    command=send_drone_goal
)
send_goal_button.pack(side="left", padx=5)

husky_target_frame = tk.Frame(left, bg=PANEL)
husky_target_frame.pack(fill="x", padx=15, pady=10)

tk.Label(
    husky_target_frame,
    text="GROUND ROBOT TARGET",
    bg=PANEL,
    fg=TEXT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", pady=(0, 5))

husky_x_entry = tk.Entry(husky_target_frame, width=7)
husky_x_entry.pack(side="left", padx=2)
husky_x_entry.insert(0, "0")

husky_y_entry = tk.Entry(husky_target_frame, width=7)
husky_y_entry.pack(side="left", padx=2)
husky_y_entry.insert(0, "0")

send_husky_goal_button = tk.Button(
    husky_target_frame,
    text="SEND GOAL",
    command=send_husky_goal
)

send_husky_goal_button.pack(side="left", padx=5)

# =========================================================
# CENTER PANEL
# =========================================================

center = tk.Frame(main, bg=BG)
center.grid(row=0, column=1, sticky="nsew", padx=5)

center.grid_rowconfigure(0, weight=1)
center.grid_columnconfigure(0, weight=1)
center.grid_columnconfigure(1, weight=1)


map_panel = tk.Frame(
    center,
    bg=PANEL,
    highlightbackground=BORDER,
    highlightthickness=1
)
map_panel.grid(row=0, column=0, sticky="nsew")

tk.Label(
    map_panel,
    text="ENVIRONMENT DISPLAY",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", padx=12, pady=8)


canvas = tk.Canvas(
    map_panel,
    bg="#0c120c",
    highlightthickness=0
)
canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))


camera_panel = tk.Frame(
    center,
    bg=PANEL,
    highlightbackground=BORDER,
    highlightthickness=1
)
camera_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

tk.Label(
    camera_panel,
    text="HUSKY DEPTH CAMERA",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", padx=12, pady=8)

depth_camera_label = tk.Label(
    camera_panel,
    text="DEPTH FEED WAITING",
    bg="#050805",
    fg=TEXT_DIM,
    font=("DejaVu Sans Mono", 10),
    width=52,
    height=24
)
depth_camera_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))


# Static map layers (grid, world boundary, base) are drawn in world coordinates
# by redraw_map_static() and redrawn whenever the canvas is resized.
canvas.bind("<Configure>", on_map_resize)
canvas.bind("<ButtonPress-1>", on_area_press)
canvas.bind("<B1-Motion>", on_area_drag)
canvas.bind("<ButtonRelease-1>", on_area_release)
redraw_map_static()


# UGV
ugv_marker = canvas.create_oval(
    390, 300,
    420, 330,
    fill=GREEN_BRIGHT,
    outline=TEXT
)

ugv_marker_label = canvas.create_text(
    405, 350,
    text="UGV-01",
    fill=TEXT,
    font=("DejaVu Sans Mono", 9, "bold")
)


# UAV
uav_marker = canvas.create_polygon(
    530, 160,
    540, 170,
    530, 180,
    520, 170,
    fill=TEXT,
    outline=GREEN
)

uav_marker_label = canvas.create_text(
    530, 195,
    text="UAV-01",
    fill=TEXT,
    font=("DejaVu Sans Mono", 9)
)

# Camera footprint of the Parrot (smallest case: over the highest terrain).
uav_fov_item = canvas.create_rectangle(0, 0, 0, 0, outline="#5fb3d8", dash=(3, 3))


# =========================================================
# LOG PANEL
# =========================================================

log_panel = tk.Frame(
    center,
    bg=PANEL,
    height=150,
    highlightbackground=BORDER,
    highlightthickness=1
)
log_panel.grid(row=1, column=0, sticky="ew", pady=(8, 0))
log_panel.pack_propagate(False)

tk.Label(
    log_panel,
    text="EVENT LOG",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 9, "bold")
).pack(anchor="w", padx=10, pady=(7, 3))

log = tk.Text(
    log_panel,
    bg="#050805",
    fg=TEXT,
    insertbackground=TEXT,
    relief="flat",
    font=("DejaVu Sans Mono", 8),
    state="disabled"
)
log.pack(fill="both", expand=True, padx=8, pady=(0, 8))


# =========================================================
# RIGHT PANEL
# =========================================================

right = tk.Frame(
    main,
    bg=PANEL,
    width=240,
    highlightbackground=BORDER,
    highlightthickness=1
)
right.grid(row=0, column=2, sticky="ns", padx=(5, 0))

tk.Label(
    right,
    text="MISSION DATA",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 11, "bold")
).pack(anchor="w", padx=15, pady=(15, 10))


def right_info(label, value):
    frame = tk.Frame(right, bg=PANEL)
    frame.pack(fill="x", padx=15, pady=6)

    tk.Label(
        frame,
        text=label,
        bg=PANEL,
        fg=TEXT_DIM,
        font=("DejaVu Sans Mono", 9)
    ).pack(side="left")

    value_label = tk.Label(
        frame,
        text=value,
        bg=PANEL,
        fg=TEXT,
        font=("DejaVu Sans Mono", 9, "bold")
    )
    value_label.pack(side="right")
    return value_label


area_label = right_info("AREA", "NONE")
mode_label = right_info("MODE", "STANDBY")
eta_label = right_info("ETA", "--:--")
waypoints_label = right_info("WAYPOINTS", "--")


tk.Frame(right, bg=BORDER, height=1).pack(
    fill="x",
    padx=12,
    pady=12
)


tk.Label(
    right,
    text="ASSET STATUS",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", padx=15)

tk.Label(
    right,
    text=(
        "UGV-01   ONLINE\n"
        "UAV-01   ONLINE\n"
        "LIDAR    ACTIVE\n"
        "CAMERA   ACTIVE\n"
        "MOWER    STANDBY\n"
        "LOCALISATION FIXED\n"
    ),
    justify="left",
    bg=PANEL,
    fg=TEXT,
    font=("DejaVu Sans Mono", 9)
).pack(anchor="w", padx=15, pady=10)


tk.Frame(right, bg=BORDER, height=1).pack(
    fill="x",
    padx=12,
    pady=12
)

tk.Label(
    right,
    text="HAZARD STATUS",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", padx=15)

tk.Label(
    right,
    text=(
        "RISK LEVEL: LOW\n"
        "DETECTIONS: 0\n"
        "ALERT STATE: CLEAR"
    ),
    justify="left",
    bg=PANEL,
    fg=TEXT,
    font=("DejaVu Sans Mono", 9)
).pack(anchor="w", padx=15, pady=10)


def clear_grass_pinpoints():
    global last_pinpoint_stamp, camera_risk_active_goal
    if mission.active:
        add_log("GRASS", "Cannot clear pinpoints while a survey is running")
        return
    try:
        with camera_risk_lock:
            grass_pinpoints.clear()
            camera_risk_pending_goals.clear()
            camera_risk_active_goal = None
            camera_risk_completed_goals.clear()
    except OSError as error:
        pinpoint_status.config(text=f"Could not clear saved pins: {error}")
        add_log("GRASS", f"Could not clear saved pins: {error}")
        return
    auto_pinpoints.set(False)
    last_pinpoint_stamp = None
    for marker, label in pinpoint_items.values():
        canvas.delete(marker)
        canvas.delete(label)
    pinpoint_items.clear()
    pinpoint_status.config(text="Pinning paused | Saved: 0")
    add_log("GRASS", "Cleared all pinpoints and saved coordinates; pinning paused")


tk.Button(
    right, text="CLEAR PINPOINTS", command=clear_grass_pinpoints,
    bg=PANEL_2, fg=AMBER, activebackground=PANEL, activeforeground=TEXT,
    highlightbackground=BORDER, font=("DejaVu Sans Mono", 9, "bold"),
    pady=10,
).pack(side="bottom", fill="x", padx=12, pady=15)


# =========================================================
# FOOTER
# =========================================================

footer = tk.Frame(
    root,
    bg="#060806",
    height=28
)
footer.pack(fill="x")
footer.pack_propagate(False)

tk.Label(
    footer,
    text="FARMER C2 // ROBOTICS STUDIO 1",
    bg="#060806",
    fg=TEXT_DIM,
    font=("DejaVu Sans Mono", 8)
).pack(side="left", padx=12)

tk.Label(
    footer,
    text="SYS READY | TELEMETRY ACTIVE | LINK SECURE",
    bg="#060806",
    fg=GREEN,
    font=("DejaVu Sans Mono", 8)
).pack(side="right", padx=12)


# =========================================================
# INITIAL LOG
# =========================================================

add_log("SYSTEM", "FARMER CONTROL STATION INITIALISED")
add_log("SYSTEM", "ROS2 COMMUNICATION ONLINE")
add_log("UGV", "UGV-01 READY")
add_log("UAV", "UAV-01 READY")
add_log("SYSTEM", "AWAITING OPERATOR COMMAND")


def on_close():
    if rclpy.ok():
        rclpy.shutdown()

    ros_node.destroy_node()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)


def advance_spiral():
    global spiral_active, spiral_index

    if not spiral_active or uav_x is None or uav_y is None or uav_z is None:
        return

    target_x, target_y, target_z = spiral_targets[spiral_index]
    distance = math.sqrt(
        (uav_x - target_x) ** 2
        + (uav_y - target_y) ** 2
        + (uav_z - target_z) ** 2
    )
    if distance > 0.2:
        return

    if spiral_index == len(spiral_targets) - 1:
        spiral_active = False
        add_log("UAV", "SPIRAL COMPLETE")
        mission_value.config(text="SURVEY COMPLETE")
        add_log("UGV", "Coordinates collected  press START HUSKY when ready")
        return

    spiral_index += 1
    publish_drone_goal(*spiral_targets[spiral_index])

def get_synced_uav_position():
    return uav_x, uav_y, uav_z


def update_telemetry_ui():
    advance_spiral()
    service_mission()

    display_uav_x, display_uav_y, display_uav_z = get_synced_uav_position()

    if battery_level is not None:
        suffix = " CHG" if charging_flag else (" LOW" if battery_low_flag else "")
        battery_label.config(
            text=f"{battery_level:.0f}%{suffix}",
            fg=RED if battery_low_flag else (GREEN_BRIGHT if charging_flag else TEXT),
        )

    if ugv_x is not None:
        ugv_x_label.config(text=f"{ugv_x:.2f} m")

    if ugv_y is not None:
        ugv_y_label.config(text=f"{ugv_y:.2f} m")

    if uav_x is not None:
        uav_x_label.config(text=f"{uav_x:.2f} m")

    if uav_y is not None:
        uav_y_label.config(text=f"{uav_y:.2f} m")

    if uav_z is not None:
        uav_z_label.config(text=f"{uav_z:.2f} m")

    # Move UGV marker
    if ugv_x is not None and ugv_y is not None:
        map_x, map_y = world_to_map(
            ugv_x,
            ugv_y
        )

        canvas.coords(
            ugv_marker,
            map_x - 15, map_y - 15,
            map_x + 15, map_y + 15
        )

        canvas.coords(
            ugv_marker_label,
            map_x, map_y + 35
        )

    # Move UAV marker
    if display_uav_x is not None and display_uav_y is not None:
        map_x, map_y = world_to_map(
            display_uav_x,
            display_uav_y
        )

        canvas.coords(
            uav_marker,
            map_x, map_y - 10,
            map_x + 10, map_y,
            map_x, map_y + 10,
            map_x - 10, map_y
        )

        canvas.coords(
            uav_marker_label,
            map_x, map_y + 25
        )

        if display_uav_z is not None:
            # Ground footprint of the downward camera (smallest case).
            agl = max(display_uav_z - CAMERA_DROP - SURVEY_CFG.terrain_max, 0.5)
            span_x, span_y = footprint(agl)
            half_w = span_x / 2.0 * map_view["scale"]
            half_h = span_y / 2.0 * map_view["scale"]
            canvas.coords(
                uav_fov_item,
                map_x - half_w, map_y - half_h,
                map_x + half_w, map_y + half_h
            )

    draw_grass_pinpoints()

    root.after(100, update_telemetry_ui)


def update_main_lidar_overlay():
    canvas.delete("lidar_overlay")
    canvas.delete("goal_overlay")
    canvas.delete("husky_goal_overlay")
    canvas.delete("grass_exclusion_overlay")
    canvas.delete("path_overlay")

    if (
        latest_lidar_scan is not None
        and ugv_x is not None
        and ugv_y is not None
        and ugv_heading is not None
    ):
        ranges = np.asarray(latest_lidar_scan.ranges, dtype=np.float32)
        angles = (
            latest_lidar_scan.angle_min
            + np.arange(len(ranges), dtype=np.float32)
            * latest_lidar_scan.angle_increment
        )
        valid = (
            np.isfinite(ranges)
            & (ranges >= latest_lidar_scan.range_min)
            & (
                ranges
                <= min(latest_lidar_scan.range_max, LIDAR_MAX_RANGE)
            )
        )
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size > MAIN_LIDAR_MAX_POINTS:
            valid_indices = valid_indices[::
                int(np.ceil(valid_indices.size / MAIN_LIDAR_MAX_POINTS))
            ]

        cos_yaw = math.cos(ugv_heading)
        sin_yaw = math.sin(ugv_heading)
        for index in valid_indices:
            local_x = float(ranges[index] * math.cos(angles[index]))
            local_y = float(ranges[index] * math.sin(angles[index]))
            world_x = ugv_x + cos_yaw * local_x - sin_yaw * local_y
            world_y = ugv_y + sin_yaw * local_x + cos_yaw * local_y
            map_x, map_y = world_to_map(world_x, world_y)
            canvas.create_oval(
                map_x - 2,
                map_y - 2,
                map_x + 2,
                map_y + 2,
                fill="#62a8d1",
                outline="",
                tags="lidar_overlay",
            )

    if latest_husky_goal is not None:
        goal_x, goal_y = world_to_map(
            latest_husky_goal[0],
            latest_husky_goal[1],
        )
        canvas.create_polygon(
            goal_x,
            goal_y - 10,
            goal_x + 10,
            goal_y,
            goal_x,
            goal_y + 10,
            goal_x - 10,
            goal_y,
            fill=RED,
            outline=TEXT,
            width=2,
            tags="goal_overlay",
        )
        canvas.create_text(
            goal_x,
            goal_y + 18,
            text="HUSKY GOAL",
            fill=RED,
            font=("DejaVu Sans Mono", 8, "bold"),
            tags="goal_overlay",
        )

        if (
            ugv_x is not None
            and ugv_y is not None
        ):
            obstacle_points = np.empty((0, 2), dtype=np.float32)
            if (
                latest_lidar_scan is not None
                and ugv_heading is not None
            ):
                ranges = np.asarray(
                    latest_lidar_scan.ranges,
                    dtype=np.float32,
                )
                angles = (
                    latest_lidar_scan.angle_min
                    + np.arange(len(ranges), dtype=np.float32)
                    * latest_lidar_scan.angle_increment
                )
                valid = (
                    np.isfinite(ranges)
                    & (ranges >= latest_lidar_scan.range_min)
                    & (
                        ranges
                        <= min(latest_lidar_scan.range_max, LIDAR_MAX_RANGE)
                    )
                )
                if np.any(valid):
                    local_x = ranges[valid] * np.cos(angles[valid])
                    local_y = ranges[valid] * np.sin(angles[valid])
                    cos_yaw = math.cos(ugv_heading)
                    sin_yaw = math.sin(ugv_heading)
                    obstacle_points = np.column_stack((
                        ugv_x + cos_yaw * local_x - sin_yaw * local_y,
                        ugv_y + sin_yaw * local_x + cos_yaw * local_y,
                    )).astype(np.float32)
            with camera_risk_lock:
                active_grass_goal = (
                    None
                    if camera_risk_active_goal is None
                    else camera_risk_active_goal.copy()
                )
            if (
                active_grass_goal is not None
                and math.hypot(
                    latest_husky_goal[0] - active_grass_goal[0],
                    latest_husky_goal[1] - active_grass_goal[1],
                ) <= HUSKY_GRASS_EXCLUSION_RADIUS
                and obstacle_points.size
            ):
                grass_goal = np.asarray(active_grass_goal[:2], dtype=np.float32)
                obstacle_points = obstacle_points[
                    np.linalg.norm(obstacle_points - grass_goal, axis=1)
                    > HUSKY_GRASS_EXCLUSION_RADIUS
                ]

            path = plan_path(
                ugv_x,
                ugv_y,
                np.asarray(latest_husky_goal, dtype=np.float32),
                obstacle_points,
            )
            if path.size:
                path_coordinates = []
                for path_x, path_y in path:
                    path_coordinates.extend(world_to_map(path_x, path_y))
                canvas.create_line(
                    *path_coordinates,
                    fill="#58c98a",
                    width=3,
                    dash=(8, 4),
                    tags="path_overlay",
                )

    with camera_risk_lock:
        pending_goals = [goal.copy() for goal in camera_risk_pending_goals]
        active_goal = (
            None
            if camera_risk_active_goal is None
            else camera_risk_active_goal.copy()
        )
    if active_goal is not None:
        exclusion_x, exclusion_y = world_to_map(
            active_goal[0],
            active_goal[1],
        )
        exclusion_radius = HUSKY_GRASS_EXCLUSION_RADIUS * map_view["scale"]
        canvas.create_oval(
            exclusion_x - exclusion_radius,
            exclusion_y - exclusion_radius,
            exclusion_x + exclusion_radius,
            exclusion_y + exclusion_radius,
            outline=AMBER,
            dash=(6, 4),
            width=2,
            tags="grass_exclusion_overlay",
        )
        canvas.create_text(
            exclusion_x,
            exclusion_y + exclusion_radius + 12,
            text=f"GRASS EXCLUSION {HUSKY_GRASS_EXCLUSION_RADIUS:.1f} m",
            fill=AMBER,
            font=("DejaVu Sans Mono", 8, "bold"),
            tags="grass_exclusion_overlay",
        )
    for goal in pending_goals:
        goal_x, goal_y = world_to_map(goal[0], goal[1])
        canvas.create_oval(
            goal_x - 5,
            goal_y - 5,
            goal_x + 5,
            goal_y + 5,
            fill=AMBER,
            outline=TEXT,
            tags="husky_goal_overlay",
        )
    if active_goal is not None:
        goal_x, goal_y = world_to_map(active_goal[0], active_goal[1])
        canvas.create_polygon(
            goal_x,
            goal_y - 9,
            goal_x + 9,
            goal_y,
            goal_x,
            goal_y + 9,
            goal_x - 9,
            goal_y,
            fill=GREEN_BRIGHT,
            outline=TEXT,
            width=2,
            tags="husky_goal_overlay",
        )
        canvas.create_text(
            goal_x,
            goal_y - 16,
            text="ACTIVE GRASS",
            fill=GREEN_BRIGHT,
            font=("DejaVu Sans Mono", 8, "bold"),
            tags="husky_goal_overlay",
        )
    root.after(MAIN_MAP_UPDATE_PERIOD_MS, update_main_lidar_overlay)


def update_depth_heatmap():
    rendered_depth = False
    if latest_husky_depth is not None:
        try:
            height, width = latest_husky_depth.shape
            display_size = (
                max(1, int(width * DEPTH_DISPLAY_SCALE)),
                max(1, int(height * DEPTH_DISPLAY_SCALE)),
            )
            depth = cv2.resize(
                latest_husky_depth,
                display_size,
                interpolation=cv2.INTER_NEAREST,
            )
            valid = np.isfinite(depth) & (depth > 0.0)
            if np.any(valid):
                near = 0.4
                far = 10.0
                normalized_depth = np.zeros_like(depth, dtype=np.uint8)
                normalized_depth[valid] = np.clip(
                    (far - depth[valid]) * 255.0 / (far - near),
                    0,
                    255,
                ).astype(np.uint8)
                heatmap = cv2.applyColorMap(
                    normalized_depth,
                    cv2.COLORMAP_TURBO,
                )
                heatmap[~valid] = (0, 0, 0)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                depth_image = PILImage.fromarray(heatmap)
                depth_image.thumbnail((1100, 650), PILImage.NEAREST)
                depth_photo = ImageTk.PhotoImage(image=depth_image)
                depth_camera_label.config(image=depth_photo, text="")
                depth_camera_label.image = depth_photo
                rendered_depth = True
        except (TypeError, ValueError) as error:
            depth_camera_label.config(text=f"DEPTH DISPLAY ERROR\n{error}")

    if depth_frame_count == 0:
        depth_camera_label.config(text="DEPTH FEED WAITING")
    elif not rendered_depth:
        depth_camera_label.config(text="DEPTH DISPLAY ERROR")
    root.after(DEPTH_DISPLAY_PERIOD_MS, update_depth_heatmap)


def update_lidar_ui():
    if latest_lidar_scan is not None:
        ranges = np.asarray(
            latest_lidar_scan.ranges,
            dtype=np.float32
        )

        angles = (
            latest_lidar_scan.angle_min
            + np.arange(
                len(ranges),
                dtype=np.float32
            )
            * latest_lidar_scan.angle_increment
        )

        # -------------------------
        # Valid lidar rays
        # -------------------------

        valid = np.isfinite(ranges)

        valid &= (
            ranges >= latest_lidar_scan.range_min
        )

        valid &= (
            ranges
            <= min(
                latest_lidar_scan.range_max,
                LIDAR_MAX_RANGE
            )
        )

        # Front +/-45 degree sector
        wrapped_angles = (
            angles + np.pi
        ) % (2.0 * np.pi) - np.pi

        front = (
            valid
            & (
                np.abs(wrapped_angles)
                <= LIDAR_FRONT_ANGLE
            )
        )

        collision = (
            front
            & (
                ranges
                <= LIDAR_COLLISION_DISTANCE
            )
        )

        lidar_points.set_data(
            angles[valid],
            ranges[valid]
        )

        lidar_front_points.set_data(
            angles[front],
            ranges[front]
        )

        # -------------------------
        # Status
        # -------------------------

        valid_count = int(
            np.count_nonzero(valid)
        )

        if np.any(front):
            nearest_front = float(
                np.min(ranges[front])
            )

            status_text = (
                f"valid rays: {valid_count} | "
                f"nearest front: {nearest_front:.2f} m"
            )
        else:
            status_text = (
                f"valid rays: {valid_count} | "
                "nearest front: none"
            )

        if np.any(collision):
            status_text += " | COLLISION ZONE"
            lidar_status.set_color("red")
        else:
            lidar_status.set_color("black")

        # -------------------------
        # Convert lidar to world XY
        # -------------------------

        world_points = np.empty(
            (0, 2),
            dtype=np.float32
        )

        if (
            ugv_x is not None
            and ugv_y is not None
            and ugv_heading is not None
            and np.any(valid)
        ):
            local_x = (
                ranges[valid]
                * np.cos(angles[valid])
            )

            local_y = (
                ranges[valid]
                * np.sin(angles[valid])
            )

            cos_yaw = math.cos(ugv_heading)
            sin_yaw = math.sin(ugv_heading)

            world_x = (
                ugv_x
                + cos_yaw * local_x
                - sin_yaw * local_y
            )

            world_y = (
                ugv_y
                + sin_yaw * local_x
                + cos_yaw * local_y
            )

            world_points = np.column_stack(
                (world_x, world_y)
            ).astype(np.float32)

            lidar_map_points.set_data(
                world_x,
                world_y
            )

            # -------------------------
            # Husky position + heading
            # -------------------------

            lidar_robot_point.set_data(
                [ugv_x],
                [ugv_y]
            )

            lidar_robot_heading.set_data(
                [
                    ugv_x,
                    ugv_x + math.cos(ugv_heading)
                ],
                [
                    ugv_y,
                    ugv_y + math.sin(ugv_heading)
                ]
            )

        else:
            lidar_map_points.set_data([], [])
            lidar_robot_point.set_data([], [])
            lidar_robot_heading.set_data([], [])

        # -------------------------
        # Goal + planned path
        # -------------------------

        with camera_risk_lock:
            active_grass_goal = (
                None
                if camera_risk_active_goal is None
                else camera_risk_active_goal.copy()
            )
        if active_grass_goal is not None:
            lidar_grass_exclusion.center = (
                float(active_grass_goal[0]),
                float(active_grass_goal[1]),
            )
            lidar_grass_exclusion.set_radius(
                HUSKY_GRASS_EXCLUSION_RADIUS
            )
            lidar_grass_exclusion.set_visible(True)
        else:
            lidar_grass_exclusion.set_visible(False)

        if latest_husky_goal is not None:
            goal = np.asarray(
                latest_husky_goal,
                dtype=np.float32
            )

            lidar_goal_points.set_data(
                [goal[0]],
                [goal[1]]
            )

            if (
                ugv_x is not None
                and ugv_y is not None
            ):
                planning_points = world_points
                if (
                    active_grass_goal is not None
                    and math.hypot(
                        goal[0] - active_grass_goal[0],
                        goal[1] - active_grass_goal[1],
                    ) <= HUSKY_GRASS_EXCLUSION_RADIUS
                ):
                    planning_points = world_points[
                        np.linalg.norm(
                            world_points - np.asarray(active_grass_goal[:2]),
                            axis=1,
                        ) > HUSKY_GRASS_EXCLUSION_RADIUS
                    ]
                path = plan_path(
                    ugv_x,
                    ugv_y,
                    goal,
                    planning_points
                )

                if path.size:
                    lidar_path_points.set_data(
                        path[:, 0],
                        path[:, 1]
                    )

                    status_text += " | path planned"

                else:
                    lidar_path_points.set_data(
                        [],
                        []
                    )

                    status_text += (
                        " | no collision-free path"
                    )

        else:
            lidar_goal_points.set_data([], [])
            lidar_path_points.set_data([], [])

        lidar_status.set_text(status_text)

        lidar_map_axis.relim()
        lidar_map_axis.autoscale_view()

        lidar_canvas.draw_idle()

    root.after(MAIN_MAP_UPDATE_PERIOD_MS, update_lidar_ui)

def draw_grass_pinpoints():
    if drone_odom_sample is None:
        return
    frame = drone_odom_sample[0].header.frame_id
    with camera_risk_lock:
        pins = [dict(pin) for pin in grass_pinpoints.pins]
    live = {pin["id"] for pin in pins}
    for pin_id in [i for i in pinpoint_items if i not in live]:
        marker, label = pinpoint_items.pop(pin_id)
        canvas.delete(marker)
        canvas.delete(label)
    for pin in pins:
        if pin["frame_id"] != frame:
            continue
        status = pin.get("status", "pending")
        colour = POI_COLOURS.get(status, AMBER)
        text = f"GRASS {pin['id']}\n({pin['x']:.1f}, {pin['y']:.1f})"
        if status in POI_TAGS:
            text += f"\n{POI_TAGS[status]}"
        if pin["id"] not in pinpoint_items:
            marker = canvas.create_oval(0, 0, 0, 0, fill=colour, outline=TEXT, width=2)
            label = canvas.create_text(
                0, 0, fill=colour, font=("DejaVu Sans Mono", 9), text=text,
            )
            pinpoint_items[pin["id"]] = marker, label
        marker, label = pinpoint_items[pin["id"]]
        canvas.itemconfigure(marker, fill=colour)
        canvas.itemconfigure(label, fill=colour, text=text)
        x, y = world_to_map(pin["x"], pin["y"])
        canvas.coords(marker, x - 6, y - 6, x + 6, y + 6)
        canvas.coords(label, x, y + 24)


def yolo_detection_worker():
    global camera_risk_active_goal, camera_risk_goal_published
    global latest_husky_goal, camera_risk_completed_goals
    while not camera_risk_stop.is_set():
        camera_risk_wakeup.wait(0.2)
        camera_risk_wakeup.clear()
        if camera_risk_stop.is_set():
            break
        with camera_risk_lock:
            husky_position = camera_risk_husky_position
            if (
                camera_risk_active_goal is not None
                and husky_position is not None
                and math.hypot(
                    husky_position[0] - camera_risk_active_goal[0],
                    husky_position[1] - camera_risk_active_goal[1],
                ) <= camera_risk_goal_tolerance
            ):
                reached_goal = camera_risk_active_goal[:2]
                camera_risk_completed_goals.append(reached_goal)
                camera_risk_pending_goals[:] = [
                    goal for goal in camera_risk_pending_goals
                    if math.hypot(
                        goal[0] - reached_goal[0],
                        goal[1] - reached_goal[1],
                    ) > camera_risk_duplicate_radius
                ]
                camera_risk_logs.put(("GOAL", "YOLO goal reached"))
                if latest_husky_goal == tuple(camera_risk_active_goal[:2]):
                    latest_husky_goal = None
                camera_risk_active_goal = None
            if (
                husky_dispatch_enabled
                and camera_risk_active_goal is None
                and camera_risk_pending_goals
            ):
                camera_risk_active_goal = camera_risk_pending_goals.pop(0)
                publish_husky_goal(
                    float(camera_risk_active_goal[0]),
                    float(camera_risk_active_goal[1]),
                )
                camera_risk_goal_published += 1
                camera_risk_logs.put((
                    "GOAL",
                    f"YOLO Husky goal sent: "
                    f"({camera_risk_active_goal[0]:.2f}, "
                    f"{camera_risk_active_goal[1]:.2f})",
                ))


def mark_survey_coverage():
    """Tell the survey what each new camera frame saw, using the drone pose at that time."""
    while frame_stamps:
        stamp = frame_stamps.popleft()
        samples = list(odom_pin_samples)
        if not samples:
            continue
        message, sample_stamp = min(samples, key=lambda sample: abs(sample[1] - stamp))
        if abs(sample_stamp - stamp) > 0.25:
            continue
        pose = message.pose.pose
        q = pose.orientation
        mission.observe_frame(
            float(pose.position.x), float(pose.position.y), float(pose.position.z),
            math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            ),
        )


def update_camera_risk_ui():
    mark_survey_coverage()
    while not yolo_detection_queue.empty():
        point, frame_id, stamp = yolo_detection_queue.get_nowait()
        if not frame_id and drone_odom_sample is not None:
            frame_id = drone_odom_sample[0].header.frame_id
        with camera_risk_lock:
            # Every sighting counts towards confirming a pin.
            if auto_pinpoints.get():
                added = grass_pinpoints.observe([point], frame_id, stamp)
            else:
                added = []
            # Outside a survey a new detection is queued for START HUSKY; during a
            # survey the mission decides when the Husky moves.
            if not survey_mode_active:
                known = [candidate[:2] for candidate in camera_risk_pending_goals]
                if camera_risk_active_goal is not None:
                    known.append(camera_risk_active_goal[:2])
                known.extend(camera_risk_completed_goals)
                if not any(
                    np.linalg.norm(point[:2] - old_point) < camera_risk_duplicate_radius
                    for old_point in known
                ):
                    camera_risk_pending_goals.append(point)
        for pin in added:
            add_log(
                "GRASS",
                f"Pinned #{pin['id']} at "
                f"({pin['x']:.2f}, {pin['y']:.2f})",
            )
        camera_risk_wakeup.set()

    with camera_risk_lock:
        pin_count = len(grass_pinpoints.pins)
    while not camera_risk_logs.empty():
        source, message = camera_risk_logs.get_nowait()
        add_log(source, message)
    if latest_yolo_image is not None:
        risk_image = PILImage.fromarray(
            cv2.cvtColor(latest_yolo_image, cv2.COLOR_BGR2RGB)
        )
        risk_image.thumbnail((1100, 650), PILImage.LANCZOS)
        risk_photo = ImageTk.PhotoImage(image=risk_image)
        camera_risk_display.config(image=risk_photo, text="")
        camera_risk_display.image = risk_photo
    else:
        camera_risk_display.config(
            image="",
            text="YOLO FEED WAITING FOR ANNOTATED IMAGE",
        )
    if not auto_pinpoints.get():
        pinpoint_status.config(text=f"Pinning paused | Saved: {pin_count}")
    else:
        pinpoint_status.config(text=f"Auto pinning | Saved grass pins: {pin_count}")
    root.after(100, update_camera_risk_ui)


camera_risk_thread = threading.Thread(
    target=yolo_detection_worker,
    daemon=True,
    name="CAMERA_RISK_THREAD",
)
camera_risk_thread.start()


update_telemetry_ui()
update_lidar_ui()
update_main_lidar_overlay()
update_depth_heatmap()
update_camera_risk_ui()

root.mainloop()