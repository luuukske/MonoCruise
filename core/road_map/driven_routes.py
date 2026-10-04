"""Local traces of driven positions, never road edges or braking evidence."""
from __future__ import annotations

from collections import defaultdict
import json
import logging
import math
from pathlib import Path
import queue
import threading

from core import settings

_log = logging.getLogger(__name__)
MAX_SEGMENTS = 200000
MAX_BYTES = 32 * 1024 * 1024
CELL = 128


class DrivenRoutes:
    def __init__(self, path: Path | None = None):
        self.path = path or settings.CONFIG_PATH.parent / 'maps' / 'driven-routes.jsonl'
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=128)
        self._worker = None
        self._grid = defaultdict(list)
        self._keys = set()
        self._last = None
        self._anchor = None
        self.status = 'Driven-route recording: waiting for game data'

    def start(self):
        with self._lock:
            if self._worker is not None:
                return
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def reset(self):
        self._last = self._anchor = None

    def observe(self, x, z, elevation, speed, stamp, *, valid=True):
        self.start()
        if not valid or not all(math.isfinite(v) for v in (x, z, elevation, speed, stamp)):
            self.reset()
            return
        if abs(x) > 200000 or abs(z) > 200000 or abs(elevation) > 5000 or abs(speed) > 80:
            self.reset()
            return
        point = (x, z, elevation)
        if self._last is not None:
            previous, previous_stamp = self._last
            dt = stamp-previous_stamp
            if dt == 0:
                return
            step = math.hypot(x-previous[0], z-previous[1])
            if not 0 < dt <= 1 or step > 90*dt+3 or abs(elevation-previous[2]) > 5:
                self.reset()
        self._last = point, stamp
        if abs(speed) < 0.1:
            self._anchor = None
            return
        if self._anchor is None:
            self._anchor = point
            return
        distance = math.hypot(x-self._anchor[0], z-self._anchor[1])
        if distance < 3:
            return
        if distance <= 30:
            try:
                self._queue.put_nowait((self._anchor, point))
            except queue.Full:
                self.status = 'Route recording queue full; this segment was skipped'
        self._anchor = point

    def _add(self, a, b):
        if len(a) != 3 or len(b) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (*a, *b)):
            return False
        if not 0 < math.hypot(b[0]-a[0], b[1]-a[1]) <= 30 or abs(b[2]-a[2]) > 5:
            return False
        if any(abs(p[0]) > 200000 or abs(p[1]) > 200000 or abs(p[2]) > 5000 for p in (a, b)):
            return False
        key = tuple(sorted(tuple(round(v/2) for v in p) for p in (a, b)))
        segment = tuple(a), tuple(b)
        with self._lock:
            if key in self._keys:
                return False
            if len(self._keys) >= MAX_SEGMENTS:
                self.status = 'Route recording capacity reached; existing routes are displayed'
                return False
            self._keys.add(key)
            for gx in range(math.floor(min(a[0], b[0])/CELL), math.floor(max(a[0], b[0])/CELL)+1):
                for gz in range(math.floor(min(a[1], b[1])/CELL), math.floor(max(a[1], b[1])/CELL)+1):
                    bucket = self._grid[gx, gz]
                    if len(bucket) < 512:
                        bucket.append(segment)
        return True

    def _run(self):
        try:
            if self.path.exists():
                with self.path.open(encoding='utf-8') as stream:
                    read_bytes = 0
                    for line in stream:
                        read_bytes += len(line.encode('utf-8'))
                        if read_bytes > MAX_BYTES:
                            break
                        try:
                            record = json.loads(line)
                            if isinstance(record, dict) and record.get('schema') == 1 and record.get('game') == 'ETS2':
                                self._add(record['a'], record['b'])
                        except (ValueError, TypeError, KeyError):
                            continue
            self.status = 'Driven routes loaded; recording stays on this device'
        except OSError:
            _log.warning('Driven routes could not be loaded')
            self.status = 'Previous route recordings could not be read'
        while True:
            segment = self._queue.get()
            try:
                if segment is None:
                    return
                a, b = segment
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size >= MAX_BYTES:
                    self.status = 'Route recording capacity reached; existing routes are displayed'
                    continue
                if not self._add(a, b):
                    continue
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'schema': 1, 'game': 'ETS2', 'a': a, 'b': b}, allow_nan=False)+'\n')
                self.status = 'Driven route recorded; it is not a road boundary'
            except OSError:
                _log.warning('Driven routes could not be saved')
                self.status = 'Route recording could not be written to file'
            finally:
                self._queue.task_done()

    def preview(self, x, z, elevation):
        if not all(math.isfinite(v) for v in (x, z, elevation)):
            return []
        gx, gz = math.floor(x/CELL), math.floor(z/CELL)
        with self._lock:
            segments = {s for ix in range(gx-1, gx+2) for iz in range(gz-1, gz+2)
                        for s in self._grid.get((ix, iz), [])}
        result = []
        for a, b in segments:
            dx, dz = b[0]-a[0], b[1]-a[1]
            length_sq = dx*dx+dz*dz
            t = max(0, min(1, ((x-a[0])*dx+(z-a[1])*dz)/length_sq))
            distance = math.hypot(x-a[0]-t*dx, z-a[1]-t*dz)
            if distance <= 120 and abs(elevation-a[2]-t*(b[2]-a[2])) <= 5:
                result.append((distance, (a, b)))
        return [s for _, s in sorted(result)[:1024]]

    def close(self):
        if self._worker is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                return
            self._worker.join(timeout=1)


routes = DrivenRoutes()
