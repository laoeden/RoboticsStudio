import tkinter as tk
from datetime import datetime
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage, ImageTk

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
latest_rgb = None
rgb_frame_count = 0

ugv_x = None
ugv_y = None
ugv_heading = None

uav_x = None
uav_y = None

def odometry_callback(msg):
    global ugv_x, ugv_y

    ugv_x = msg.pose.pose.position.x
    ugv_y = msg.pose.pose.position.y
    
odom_sub = ros_node.create_subscription(
    Odometry,
    "/husky1/odometry",
    odometry_callback,
    10
)

def drone_odometry_callback(msg):
    global uav_x, uav_y

    uav_x = msg.pose.pose.position.x
    uav_y = msg.pose.pose.position.y


drone_odom_sub = ros_node.create_subscription(
    Odometry,
    "/parrot1/odometry",
    drone_odometry_callback,
    10
)


def rgb_callback(msg):
    global latest_rgb, rgb_frame_count

    try:
        latest_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb_frame_count += 1
    except Exception as error:
        ros_node.get_logger().error(f"RGB image conversion failed: {error}")


rgb_sub = ros_node.create_subscription(
    RosImage,
    "/parrot1/camera/image",
    rgb_callback,
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


def publish_command(command):
    msg = String()
    msg.data = command
    mission_command_pub.publish(msg)

    ros_node.get_logger().info(f"Published command: {command}")

def send_drone_goal():
    try:
        x = float(drone_x_entry.get())
        y = float(drone_y_entry.get())
        z = float(drone_z_entry.get())
    except ValueError:
        ros_node.get_logger().error("Drone coordinates must be numbers")
        return

    msg = PointStamped()
    msg.header.frame_id = "parrot1_odom"
    msg.point.x = x
    msg.point.y = y
    msg.point.z = z

    drone_goal_pub.publish(msg)

    ros_node.get_logger().info(
        f"Sent drone goal: ({x}, {y}, {z})"
    )

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def add_log(source, message):
    log.config(state="normal")
    log.insert(tk.END, f"[{timestamp()}] [{source}] {message}\n")
    log.see(tk.END)
    log.config(state="disabled")

def start_mission():
    publish_command("START")

    status_value.config(text="ACTIVE", fg=GREEN_BRIGHT)
    mission_value.config(text="EXECUTING")
    add_log("SYSTEM", "MISSION STARTED")


def stop_mission():
    publish_command("STOP")

    status_value.config(text="HALTED", fg=RED)
    mission_value.config(text="STOPPED")
    add_log("SYSTEM", "MISSION STOPPED")


def return_to_base():
    publish_command("RETURN_TO_BASE")

    status_value.config(text="RTB", fg=AMBER)
    mission_value.config(text="RETURN TO BASE")
    add_log("UGV", "RETURN TO BASE COMMAND SENT")


def emergency_stop():
    publish_command("EMERGENCY_STOP")

    status_value.config(text="E-STOP", fg=RED)
    mission_value.config(text="EMERGENCY STOP")
    add_log("SYSTEM", "EMERGENCY STOP ACTIVATED")

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
# MAIN AREA
# =========================================================

main = tk.Frame(root, bg=BG)
main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

main.grid_columnconfigure(1, weight=1)
main.grid_rowconfigure(0, weight=1)


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
left.grid_propagate(False)

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


info_row(left, "BATTERY", "92%")
info_row(left, "PROGRESS", "0%")
info_row(left, "HAZARDS", "0%")
info_row(left, "LOCALISATION", "FIXED")
info_row(left, "LINK", "GOOD")

ugv_x_label = info_row(left, "UGV X", "-- m")
ugv_y_label = info_row(left, "UGV Y", "-- m")


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


tactical_button("[ START MISSION ]", start_mission, GREEN_BRIGHT)
tactical_button("[ STOP ]", stop_mission, AMBER)
tactical_button("[ RETURN TO BASE ]", return_to_base)
tactical_button("[ EMERGENCY STOP ]", emergency_stop, RED)

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

tk.Button(
    drone_target_frame,
    text="SEND GOAL",
    command=send_drone_goal
).pack(side="left", padx=5)

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
    text="PARROT RGB CAMERA",
    bg=PANEL,
    fg=GREEN_BRIGHT,
    font=("DejaVu Sans Mono", 10, "bold")
).pack(anchor="w", padx=12, pady=8)

rgb_camera_label = tk.Label(
    camera_panel,
    text="RGB FEED WAITING",
    bg="#050805",
    fg=TEXT_DIM,
    font=("DejaVu Sans Mono", 10),
    width=52,
    height=24
)
rgb_camera_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))


# Grid
for x in range(0, 1000, 40):
    canvas.create_line(
        x, 0, x, 700,
        fill="#1b241a"
    )

for y in range(0, 700, 40):
    canvas.create_line(
        0, y, 1000, y,
        fill="#1b241a"
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
    dash=(8, 5)
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
        outline=""
    )

    canvas.create_text(
        hx,
        hy + 2,
        text="!",
        fill="white",
        font=("DejaVu Sans Mono", 9, "bold")
    )


canvas.create_text(
    15,
    15,
    text='GRID REF: FIELD_A1',
    anchor="nw",
    fill=TEXT_DIM,
    font=("DejaVu Sans Mono", 8)
)


# =========================================================
# LOG PANEL
# =========================================================

log_panel = tk.Frame(
    center,
    bg=PANEL,
    height=170,
    highlightbackground=BORDER,
    highlightthickness=1
)
log_panel.grid(row=1, column=0, sticky="ew", pady=(8, 0))
log_panel.grid_propagate(False)

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
right.grid_propagate(False)

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

def update_telemetry_ui():
    if ugv_x is not None:
        ugv_x_label.config(text=f"{ugv_x:.2f} m")

    if ugv_y is not None:
        ugv_y_label.config(text=f"{ugv_y:.2f} m")

    # Move UGV marker
    if ugv_x is not None and ugv_y is not None:
        map_x, map_y = world_to_map(ugv_x, ugv_y)

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
    if uav_x is not None and uav_y is not None:
        map_x, map_y = world_to_map(uav_x, uav_y)

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

    # Update RGB camera
    rendered_rgb = False

    if latest_rgb is not None:
        try:
            rgb_image = cv2.cvtColor(latest_rgb, cv2.COLOR_BGR2RGB)
            rgb_image = PILImage.fromarray(rgb_image)
            rgb_image.thumbnail((620, 500), PILImage.LANCZOS)
            rgb_photo = ImageTk.PhotoImage(image=rgb_image)

            rgb_camera_label.config(image=rgb_photo, text="")
            rgb_camera_label.image = rgb_photo
            rendered_rgb = True

        except Exception as error:
            rgb_camera_label.config(
                text=f"RGB DISPLAY ERROR\n{error}"
            )

    if rgb_frame_count == 0:
        rgb_camera_label.config(text="RGB FEED WAITING")
    elif not rendered_rgb:
        rgb_camera_label.config(text="RGB DISPLAY ERROR")

    root.after(100, update_telemetry_ui)


update_telemetry_ui()

root.mainloop()
