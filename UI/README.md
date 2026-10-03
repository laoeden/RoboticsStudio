# FARMER User Interface

Start the simulation and drone movement controller, then run from the repository root:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 UI/farmer_ui.py
```

## Automatic dry-grass pinpoints

Tune the CAMERA RISK colour controls until the dry-grass patches have detection
boxes. With **Automatically pin dry grass** checked, fly the drone across the map.
The GUI combines patch depth with the camera mounting transform and drone
odometry to estimate the patch's position. Three repeated sightings confirm a
pin; detections within one metre of a saved pin are merged.

Amber `GRASS` markers and XY coordinates appear on the MAIN environment display.
They remain at their recorded coordinates as the drone moves. Pins are saved in
`UI/dry_grass_pinpoints.json` and loaded on the next GUI start. The file records
XYZ and the drone odometry frame for future use. Keep the same world and coordinate
origin when reusing a survey; move this file aside before starting a new survey.

The status below the colour sliders shows the saved count or missing sensor data.
Pinning requires RGB, depth, and drone odometry timestamps within 0.25 seconds.
Turning off automatic pinning pauses recording and keeps existing pins.

Pins represent colour-matched candidates, not confirmed grass classifications:
the path or kangaroo can also produce a pin if they pass the colour filter.
Positions are estimates using the current Parrot camera calibration and median
patch depth. This feature records locations only; it does not send Husky goals.
Before sending a recorded point later, transform it from its saved source frame
to the Husky controller's coordinate frame.

Use **CLEAR PINPOINTS** at the bottom right of MAIN to remove all grass markers,
pending detections, and the saved coordinate file. Recording pauses after clearing;
re-enable **Automatically pin dry grass** in CAMERA RISK to start a new survey.
