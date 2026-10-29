"""Track grass sightings in memory in their source coordinate frame."""
import json
import math
from pathlib import Path


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


PENDING = 'pending'
ACTIVE = 'active'
OPEN_STATUSES = (PENDING, ACTIVE)
STATUSES = (PENDING, ACTIVE, 'cleared', 'false_positive', 'unreachable', 'visited')


class GrassPinpoints:
    def __init__(self, path=None, radius=1.0, confirmations=1):
        self.path = Path(path) if path is not None else None
        self.radius = radius
        self.confirmations = confirmations
        self.pins = []
        self.candidates = []
        self.load_error = None
        if self.path is not None and self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if data.get('version') != 1:
                    raise ValueError('Unsupported pinpoint file version')
                for pin in data['pins']:
                    if not pin['frame_id'] or not all(math.isfinite(float(pin[k])) for k in ('x', 'y', 'z')):
                        raise ValueError('Invalid pinpoint coordinates')
                for pin in data['pins']:
                    # Files written before the survey feature have no status.
                    if pin.get('status') not in STATUSES:
                        pin['status'] = PENDING
                self.pins = data['pins']
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.load_error = str(error)

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
            # A cleared pin is history (the hazard is gone), so it must not hide a new
            # sighting; this also keeps a restarted simulation from being ignored.
            if any(nearby(pin) for pin in self.pins if pin['status'] != 'cleared'):
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
                pin.update(id=self.next_id(), detected_at=timestamp, status=PENDING)
                self.pins.append(pin)
                added.append(pin)
                self.candidates.remove(candidate)
        if added and self.path is not None:
            self.save()
        return added

    def next_id(self):
        """Ids never repeat while pins exist, even after some were discarded."""
        return max((pin['id'] for pin in self.pins), default=0) + 1

    @staticmethod
    def _inside(pin, bounds):
        xmin, ymin, xmax, ymax = bounds
        return xmin <= pin['x'] <= xmax and ymin <= pin['y'] <= ymax

    def open_in(self, bounds):
        """Pins inside (xmin, ymin, xmax, ymax) that still need a visit."""
        return [pin for pin in self.pins
                if pin['status'] in OPEN_STATUSES and self._inside(pin, bounds)]

    def set_status(self, pin_id, status):
        """Record the outcome for one pin; returns False if the id is unknown."""
        if status not in STATUSES:
            raise ValueError(f'Unknown pin status {status!r}')
        for pin in self.pins:
            if pin['id'] == pin_id:
                if pin['status'] != status:
                    pin['status'] = status
                    self.save()
                return True
        return False

    def discard_for_resurvey(self, bounds):
        """Forget earlier results inside an area before surveying it again.

        Everything is dropped except false positives, which stay so the same
        spot is not pinned and visited again. Returns the discarded ids.
        """
        gone = [pin['id'] for pin in self.pins
                if self._inside(pin, bounds) and pin['status'] != 'false_positive']
        if gone:
            self.pins = [pin for pin in self.pins if pin['id'] not in gone]
        before = len(self.candidates)
        self.candidates = [c for c in self.candidates if not self._inside(c, bounds)]
        if gone:
            self.save()
        return gone

    def clear(self):
        """Reset the current session's detections without accessing disk."""
        if self.path is not None:
            self.path.unlink(missing_ok=True)
        self.pins.clear()
        self.candidates.clear()
        self.load_error = None

    def save(self):
        """Persist pins when this tracker was configured with a path."""
        if self.path is None:
            return
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'version': 1, 'pins': self.pins}, indent=2) + '\n')
        temporary.replace(self.path)
