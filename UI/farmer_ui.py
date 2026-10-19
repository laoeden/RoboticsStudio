import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from UI.grass_pinpoints import GrassPinpoints
from collections import deque
from HuskyScripts.grid_path_planner import plan_path

import tkinter as tk
from tkinter import ttk
from datetime import datetime
import math
import queue
import threading

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
camera_risk_duplicate_radius = 1.0
camera_risk_logs = queue.Queue()
yolo_detection_queue = queue.Queue()


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


husky_depth_sub = ros_node.create_subscription(
    RosImage,
    "/husky1/camera/depth/image",
    husky_depth_callback,
    10,
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

    pause_husky_dispatch()
    publish_command("STOP")
    spiral_active = False

    status_value.config(text="HALTED", fg=RED)
    mission_value.config(text="STOPPED")
    add_log("SYSTEM", "MISSION STOPPED")


def return_to_base():
    global spiral_active

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
info_row(left, "PROGRESS", "0%")
info_row(left, "HAZARDS", "0%")
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


# Grid
for x in range(0, 1000, 40):
    canvas.create_line(
        x, 0, x, 700,
        fill="#1b241a",
        tags="map_feature"
    )

for y in range(0, 700, 40):
    canvas.create_line(
        0, y, 1000, y,
        fill="#1b241a",
        tags="map_feature"
    )


# Mission boundary
canvas.create_polygon(
    120, 120,
    540, 90,
    760, 220,
    680, 500,
    220, 520,
    90, 340,
    outline=GREEN,
    fill="",
    width=2,
    dash=(8, 5),
    tags="map_feature"
)

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

# Convert ROS world coordinates to environment display coordinates
MAP_ORIGIN_X = 405
MAP_ORIGIN_Y = 315
MAP_SCALE = 25

def world_to_map(x, y):
    map_x = MAP_ORIGIN_X + (x * MAP_SCALE)
    map_y = MAP_ORIGIN_Y - (y * MAP_SCALE)

    return map_x, map_y

# Hazards
hazards = [
    (640, 190),
    (700, 320),
    (470, 420)
]

for hx, hy in hazards:
    canvas.create_polygon(
        hx, hy - 12,
        hx - 10, hy + 10,
        hx + 10, hy + 10,
        fill=RED,
        outline="",
        tags="map_feature"
    )

    canvas.create_text(
        hx,
        hy + 2,
        text="!",
        fill="white",
        font=("DejaVu Sans Mono", 9, "bold"),
        tags="map_feature"
    )


canvas.create_text(
    15,
    15,
    text='GRID REF: FIELD_A1',
    anchor="nw",
    fill=TEXT_DIM,
    font=("DejaVu Sans Mono", 8),
    tags="map_feature"
)


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

    tk.Label(
        frame,
        text=value,
        bg=PANEL,
        fg=TEXT,
        font=("DejaVu Sans Mono", 9, "bold")
    ).pack(side="right")


right_info("AREA", "FIELD_A1")
right_info("MODE", "SURVEY/MOW")
right_info("ETA", "01:20:00")
right_info("WAYPOINTS", "24")


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
    try:
        with camera_risk_lock:
            grass_pinpoints.clear()
            camera_risk_pending_goals.clear()
            camera_risk_active_goal = None
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
        add_log("UGV", "Coordinates collected — press START HUSKY when ready")
        return

    spiral_index += 1
    publish_drone_goal(*spiral_targets[spiral_index])

def get_synced_uav_position():
    return uav_x, uav_y, uav_z


def update_telemetry_ui():
    advance_spiral()

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

    draw_grass_pinpoints()

    rendered_depth = False
    if latest_husky_depth is not None:
        try:
            valid = np.isfinite(latest_husky_depth) & (latest_husky_depth > 0.0)
            if np.any(valid):
                near = 0.4
                far = 10.0
                normalized_depth = np.zeros_like(
                    latest_husky_depth,
                    dtype=np.uint8,
                )
                normalized_depth[valid] = np.clip(
                    (far - latest_husky_depth[valid]) * 255.0 / (far - near),
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
                depth_image.thumbnail((1100, 650), PILImage.LANCZOS)
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

    root.after(100, update_telemetry_ui)


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
                path = plan_path(
                    ugv_x,
                    ugv_y,
                    goal,
                    world_points
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

    root.after(
        100,
        update_lidar_ui
    )

def draw_grass_pinpoints():
    if drone_odom_sample is None:
        return
    frame = drone_odom_sample[0].header.frame_id
    with camera_risk_lock:
        pins = list(grass_pinpoints.pins)
    for pin in pins:
        if pin["frame_id"] != frame:
            continue
        if pin["id"] not in pinpoint_items:
            marker = canvas.create_oval(0, 0, 0, 0, fill=AMBER, outline=TEXT, width=2)
            label = canvas.create_text(
                0, 0, fill=AMBER, font=("DejaVu Sans Mono", 9),
                text=f"GRASS {pin['id']}\n({pin['x']:.1f}, {pin['y']:.1f})",
            )
            pinpoint_items[pin["id"]] = marker, label
        marker, label = pinpoint_items[pin["id"]]
        x, y = world_to_map(pin["x"], pin["y"])
        canvas.coords(marker, x - 6, y - 6, x + 6, y + 6)
        canvas.coords(label, x, y + 24)


def yolo_detection_worker():
    global camera_risk_active_goal, camera_risk_goal_published
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
                camera_risk_logs.put(("GOAL", "YOLO goal reached"))
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
def update_camera_risk_ui():
    with camera_risk_lock:
        pin_count = len(grass_pinpoints.pins)
    while not yolo_detection_queue.empty():
        point, frame_id, stamp = yolo_detection_queue.get_nowait()
        with camera_risk_lock:
            known = [candidate[:2] for candidate in camera_risk_pending_goals]
            if camera_risk_active_goal is not None:
                known.append(camera_risk_active_goal[:2])
            if any(
                np.linalg.norm(point[:2] - old_point)
                < camera_risk_duplicate_radius
                for old_point in known
            ):
                continue
            camera_risk_pending_goals.append(point)
            if auto_pinpoints.get():
                added = grass_pinpoints.observe(
                    [point], frame_id, stamp
                )
            else:
                added = []
        for pin in added:
            add_log(
                "GRASS",
                f"Pinned #{pin['id']} at "
                f"({pin['x']:.2f}, {pin['y']:.2f})",
            )
        camera_risk_wakeup.set()

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
update_camera_risk_ui()

root.mainloop()
