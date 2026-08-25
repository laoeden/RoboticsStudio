import tkinter as tk
from datetime import datetime

# =========================================================
# FARMER TACTICAL UI
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
root.title("FARMER Tactical Control Station")
root.geometry("1200x700")
root.configure(bg=BG)


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def add_log(source, message):
    log.config(state="normal")
    log.insert(tk.END, f"[{timestamp()}] [{source}] {message}\n")
    log.see(tk.END)
    log.config(state="disabled")


def start_mission():
    status_value.config(text="ACTIVE", fg=GREEN_BRIGHT)
    mission_value.config(text="EXECUTING")
    add_log("SYSTEM", "MISSION STARTED")


def stop_mission():
    status_value.config(text="HALTED", fg=RED)
    mission_value.config(text="STOPPED")
    add_log("SYSTEM", "MISSION STOPPED")


def return_to_base():
    status_value.config(text="RTB", fg=AMBER)
    mission_value.config(text="RETURN TO BASE")
    add_log("UGV", "RETURN TO BASE COMMAND SENT")


def emergency_stop():
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

    tk.Label(
        row,
        text=value,
        bg=PANEL,
        fg=TEXT,
        font=("DejaVu Sans Mono", 9, "bold")
    ).pack(side="right")


info_row(left, "BATTERY", "92%")
info_row(left, "PROGRESS", "0%")
info_row(left, "HAZARDS", "0%")
info_row(left, "LOCALISATION", "FIXED")
info_row(left, "LINK", "GOOD")


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


# =========================================================
# CENTER PANEL
# =========================================================

center = tk.Frame(main, bg=BG)
center.grid(row=0, column=1, sticky="nsew", padx=5)

center.grid_rowconfigure(0, weight=1)
center.grid_columnconfigure(0, weight=1)


map_panel = tk.Frame(
    center,
    bg=PANEL,
    highlightbackground=BORDER,
    highlightthickness=1
)
map_panel.grid(row=0, column=0, sticky="nsew")

tk.Label(
    map_panel,
    text="TACTICAL ENVIRONMENT DISPLAY",
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


# Route
canvas.create_line(
    150, 450,
    250, 380,
    350, 340,
    450, 300,
    550, 230,
    fill=GREEN_BRIGHT,
    width=2,
    dash=(6, 4)
)


# Base
canvas.create_rectangle(
    125, 425,
    155, 455,
    fill=GREEN,
    outline=GREEN_BRIGHT
)

canvas.create_text(
    140, 475,
    text="BASE",
    fill=TEXT,
    font=("DejaVu Sans Mono", 8)
)


# UGV
canvas.create_oval(
    390, 300,
    420, 330,
    fill=GREEN_BRIGHT,
    outline=TEXT
)

canvas.create_text(
    405, 350,
    text="UGV-01",
    fill=TEXT,
    font=("DejaVu Sans Mono", 9, "bold")
)


# UAV
canvas.create_polygon(
    530, 160,
    540, 170,
    530, 180,
    520, 170,
    fill=TEXT,
    outline=GREEN
)

canvas.create_text(
    530, 195,
    text="UAV-01",
    fill=TEXT,
    font=("DejaVu Sans Mono", 9)
)


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


root.mainloop()