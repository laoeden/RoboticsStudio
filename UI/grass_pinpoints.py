"""Confirm repeated grass sightings and persist positions in their source frame."""
import json
import math
from pathlib import Path


class GrassPinpoints:
    def __init__(self, path, radius=1.0, confirmations=3):
        self.path = Path(path)
        self.radius = radius
        self.confirmations = confirmations
        self.pins = []
        self.candidates = []
        self.load_error = None
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if data.get('version') != 1:
                    raise ValueError('Unsupported pinpoint file version')
                for pin in data['pins']:
                    if not pin['frame_id'] or not all(math.isfinite(float(pin[k])) for k in ('x', 'y', 'z')):
                        raise ValueError('Invalid pinpoint coordinates')
                self.pins = data['pins']
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.load_error = str(error)

    def observe(self, points, frame_id, timestamp):
        if not frame_id or self.load_error:
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
        if added:
            self.save()
        return added

    def clear(self):
        """Remove the persisted survey before resetting in-memory detections."""
        self.path.unlink(missing_ok=True)
        self.pins.clear()
        self.candidates.clear()
        self.load_error = None

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'version': 1, 'pins': self.pins}, indent=2) + '\n')
        temporary.replace(self.path)
