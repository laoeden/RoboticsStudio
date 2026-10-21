"""Track grass sightings in memory in their source coordinate frame."""
import math


def match_pin_samples(rgb_samples, depth_samples, odom_samples, tolerance=0.25):
    """Find the newest RGB sample with depth and pose within the time window."""
    for rgb in reversed(tuple(rgb_samples)):
        if not depth_samples or not odom_samples:
            return None
        depth = min(tuple(depth_samples), key=lambda sample: abs(sample[1] - rgb[1]))
        odom = min(tuple(odom_samples), key=lambda sample: abs(sample[1] - rgb[1]))
        stamps = (rgb[1], depth[1], odom[1])
        if max(stamps) - min(stamps) <= tolerance:
            return rgb, depth, odom
    return None


class GrassPinpoints:
    def __init__(self, radius=1.0, confirmations=1):
        self.radius = radius
        self.confirmations = confirmations
        self.pins = []
        self.candidates = []

    def observe(self, points, frame_id, timestamp):
        if not frame_id:
            return []
        self.candidates = [c for c in self.candidates if 0 <= timestamp - c['last_seen'] <= 2.0]
        added = []
        touched = set()
        for point in points:
            x, y, z = map(float, point)
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            def nearby(record):
                return record['frame_id'] == frame_id and math.hypot(x - record['x'], y - record['y']) < self.radius
            if any(nearby(pin) for pin in self.pins):
                continue
            candidate = next((c for c in self.candidates if nearby(c)), None)
            if candidate is None:
                candidate = dict(x=x, y=y, z=z, frame_id=frame_id, count=0, last_seen=timestamp)
                self.candidates.append(candidate)
            if id(candidate) in touched:
                continue
            touched.add(id(candidate))
            count = candidate['count'] + 1
            for key, value in zip(('x', 'y', 'z'), (x, y, z)):
                candidate[key] += (value - candidate[key]) / count
            candidate.update(count=count, last_seen=timestamp)
            if count >= self.confirmations:
                pin = {key: candidate[key] for key in ('x', 'y', 'z', 'frame_id')}
                pin.update(id=len(self.pins) + 1, detected_at=timestamp)
                self.pins.append(pin)
                added.append(pin)
                self.candidates.remove(candidate)
        return added

    def clear(self):
        """Reset the current session's detections without accessing disk."""
        self.pins.clear()
        self.candidates.clear()
