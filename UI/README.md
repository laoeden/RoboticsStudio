# FARMER User Interface

Start the complete simulation, controllers, YOLO detector, and GUI with:

```bash
./run_farmer.bash
```

## Topic dry-grass pinpoints

The GUI subscribes to `/parrot1/yolo/world_position` (`geometry_msgs/msg/PointStamped`).
Every two seconds it processes the coordinates received since the previous refresh
and adds amber `GRASS` markers to the MAIN map. The camera feed continues updating
separately. Points must have a source frame and finite XYZ coordinates; detections
within one metre of an existing pin in the same frame are merged.

Pins stay in memory for the current GUI session. No JSON file is read or written,
and restarting the GUI starts an empty map. Pins remain visible as the drone moves;
a lack of new topic messages does not erase already surveyed locations.

**Automatically pin dry grass** in CAMERA RISK pauses or resumes collection.
**CLEAR PINPOINTS** clears session pins and queued coordinates and pauses collection;
re-enable the checkbox to collect again.

New pins also populate the Husky target queue. After the drone survey finishes,
press **START HUSKY** to release queued targets. Repeated messages for a pin do not
requeue it after arrival. The map displays pins matching the drone odometry frame.
Husky goals use the existing shared odometry-origin assumption; deployments with
different coordinate origins require a transform before sending targets.
