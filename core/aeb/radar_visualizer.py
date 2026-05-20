"""
Live visualization server for radar speed/acceleration filtering.

Shows raw vs smoothed speed and smoothed acceleration of the ACC-tracked
lead vehicle, or the closest radar vehicle when no lead is tracked.
Single stream — no vehicle IDs, no selection UI.
"""

from flask import Flask, render_template_string
from flask_socketio import SocketIO
import threading
import time
from collections import deque


class RadarVisualizer:
    def __init__(self, port=5000):
        self.port = port
        self.app = Flask(__name__)
        # Force threaded async mode so we don't require eventlet/gevent.
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")

        self.raw_speed = deque(maxlen=100)
        self.filtered_speed = deque(maxlen=100)
        self.filtered_accel = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        self.last_update = 0.0
        self.stale_timeout = 1.0  # seconds before stream is considered idle
        self._is_tracked = False

        self.lock = threading.Lock()
        self.running = False
        self._server_thread = None

        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(self._get_html_template())

    def _get_html_template(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Lead Vehicle Radar</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: monospace; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #4CAF50; }
        .chart-container { background: #2a2a2a; padding: 20px; margin: 20px 0; border-radius: 8px; }
        canvas { max-height: 400px; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }
        .stat-box { background: #2a2a2a; padding: 15px; border-radius: 5px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 12px; color: #aaa; margin-top: 5px; }
        .status { text-align: center; color: #aaa; margin-bottom: 10px; }
        .status.active { color: #4CAF50; }
        .status.idle { color: #ff6b6b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MonoCruise Lead Vehicle Radar</h1>
        <div class="status" id="status">Waiting for vehicles...</div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value" id="raw-speed">0.00</div>
                <div class="stat-label">Raw Speed (m/s)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="filtered-speed">0.00</div>
                <div class="stat-label">Smoothed Speed (m/s)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="filtered-accel">0.00</div>
                <div class="stat-label">Smoothed Accel (m/s&sup2;)</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="speedChart"></canvas>
        </div>

        <div class="chart-container">
            <canvas id="accelChart"></canvas>
        </div>
    </div>

    <script>
        const socket = io();

        socket.on('connect', () => { console.log('socket connected'); });
        socket.on('disconnect', () => { console.log('socket disconnected'); });

        const speedCtx = document.getElementById('speedChart').getContext('2d');
        const accelCtx = document.getElementById('accelChart').getContext('2d');

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: true,
            animation: false,
            scales: {
                x: { display: false, grid: { color: '#444' } },
                y: { grid: { color: '#444' }, ticks: { color: '#fff' } }
            },
            plugins: { legend: { labels: { color: '#fff' } } }
        };

        const speedChart = new Chart(speedCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Raw Speed',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0
                    },
                    {
                        label: 'Smoothed Speed',
                        data: [],
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                ...commonOptions,
                plugins: {
                    ...commonOptions.plugins,
                    title: { display: true, text: 'Lead Speed (m/s)', color: '#fff' }
                }
            }
        });

        const accelChart = new Chart(accelCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Smoothed Acceleration',
                        data: [],
                        borderColor: '#ffd166',
                        backgroundColor: 'rgba(255, 209, 102, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                ...commonOptions,
                plugins: {
                    ...commonOptions.plugins,
                    title: { display: true, text: 'Lead Acceleration (m/s²)', color: '#fff' }
                }
            }
        });

        socket.on('update', (data) => {
            const statusEl = document.getElementById('status');
            if (data.active) {
                statusEl.textContent = data.is_tracked ? 'Tracking ACC lead' : 'Closest vehicle (no ACC lead)';
                statusEl.className = 'status active';
            } else {
                statusEl.textContent = 'No vehicles';
                statusEl.className = 'status idle';
            }

            speedChart.data.labels = data.labels;
            speedChart.data.datasets[0].data = data.raw_speed;
            speedChart.data.datasets[1].data = data.filtered_speed;
            speedChart.update();

            accelChart.data.labels = data.labels;
            accelChart.data.datasets[0].data = data.filtered_accel;
            accelChart.update();

            if (data.raw_speed.length > 0) {
                document.getElementById('raw-speed').textContent =
                    data.raw_speed[data.raw_speed.length - 1].toFixed(2);
                document.getElementById('filtered-speed').textContent =
                    data.filtered_speed[data.filtered_speed.length - 1].toFixed(2);
                document.getElementById('filtered-accel').textContent =
                    data.filtered_accel[data.filtered_accel.length - 1].toFixed(2);
            } else {
                document.getElementById('raw-speed').textContent = '0.00';
                document.getElementById('filtered-speed').textContent = '0.00';
                document.getElementById('filtered-accel').textContent = '0.00';
            }
        });
    </script>
</body>
</html>
"""

    def _emit_data(self):
        """Background thread to emit data to clients."""
        while self.running:
            with self.lock:
                active = (time.time() - self.last_update) < self.stale_timeout and len(self.timestamps) > 0
                payload = {
                    'active': active,
                    'is_tracked': self._is_tracked,
                    'labels': list(self.timestamps),
                    'raw_speed': list(self.raw_speed),
                    'filtered_speed': list(self.filtered_speed),
                    'filtered_accel': list(self.filtered_accel),
                }
            try:
                self.socketio.emit('update', payload)
            except Exception:
                pass
            time.sleep(0.1)

    def push_data(self, raw_speed, filtered_speed, filtered_accel, is_tracked: bool = True):
        """Push the vehicle's raw speed, smoothed speed, smoothed accel."""
        with self.lock:
            now = time.time()
            self.timestamps.append(now)
            self.raw_speed.append(float(raw_speed))
            self.filtered_speed.append(float(filtered_speed))
            self.filtered_accel.append(float(filtered_accel))
            self.last_update = now
            self._is_tracked = is_tracked

    def clear(self):
        """Drop buffered samples (call when no lead is present)."""
        with self.lock:
            self.timestamps.clear()
            self.raw_speed.clear()
            self.filtered_speed.clear()
            self.filtered_accel.clear()
            self.last_update = 0.0

    def _run_server(self) -> None:
        """Run the SocketIO server (blocks until stopped)."""
        self.socketio.run(
            self.app,
            host="127.0.0.1",
            port=self.port,
            debug=False,
            use_reloader=False,
        )

    def start(self):
        """Start the visualization server."""
        if self.running:
            return

        self.running = True
        threading.Thread(target=self._emit_data, daemon=True).start()
        # Run the web server in its own thread so we don't block callers.
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

    def stop(self):
        """Stop the visualization server."""
        self.running = False
        try:
            self.socketio.stop()
        except Exception:
            pass


if __name__ == "__main__":
    visualizer = RadarVisualizer(port=5000)
    visualizer.start()
