"""
Live visualization server for radar speed/acceleration filtering.
Run this alongside MonoCruise to visualize raw vs filtered data.
"""

from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO
import threading
import time
import math
from collections import deque, defaultdict

class RadarVisualizer:
    def __init__(self, port=5000):
        self.port = port
        self.app = Flask(__name__)
        # Force threaded async mode so we don't require eventlet/gevent.
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")
        
        self.vehicle_data = defaultdict(lambda: {
            "raw_speed": deque(maxlen=100),
            "filtered_speed": deque(maxlen=100),
            "filtered_accel": deque(maxlen=100),
            "timestamps": deque(maxlen=100),
            "last_update": 0
        })
        
        self.tracked_vehicle_id = None
        self.available_vehicles = set()
        self.lock = threading.Lock()
        self.running = False
        self._server_thread = None
        
        self._setup_routes()
        
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(self._get_html_template())
        
        @self.app.route('/track/<int:vehicle_id>', methods=['POST'])
        def track_vehicle(vehicle_id):
            with self.lock:
                self.tracked_vehicle_id = vehicle_id
            return jsonify({"status": "ok", "tracking": vehicle_id})
    
    def _get_html_template(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Radar Data Visualizer</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: monospace; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #4CAF50; }
        .controls { background: #2a2a2a; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .control-group { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        input, select, button { 
            padding: 10px; 
            background: #333; 
            color: #fff; 
            border: 1px solid #555; 
            border-radius: 4px;
            font-family: monospace;
        }
        button { cursor: pointer; background: #4CAF50; border: none; }
        button:hover { background: #45a049; }
        .chart-container { background: #2a2a2a; padding: 20px; margin: 20px 0; border-radius: 8px; }
        canvas { max-height: 400px; }
        .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 20px; }
        .stat-box { background: #333; padding: 15px; border-radius: 5px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 12px; color: #aaa; margin-top: 5px; }
        #vehicle-list { max-height: 200px; overflow-y: auto; background: #333; padding: 10px; border-radius: 4px; }
        .vehicle-item { padding: 5px; cursor: pointer; border-radius: 3px; }
        .vehicle-item:hover { background: #444; }
        .vehicle-item.active { background: #4CAF50; color: #000; }
        .status { color: #4CAF50; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MonoCruise Radar Filter Visualization</h1>
        
        <div class="controls">
            <div class="control-group">
                <label>Track Vehicle ID:</label>
                <input type="number" id="vehicle-id" placeholder="Enter vehicle ID" value="">
                <button onclick="trackVehicle()">Track</button>
                <span class="status" id="status">No vehicle tracked</span>
            </div>
            <div class="control-group">
                <label>Available Vehicles:</label>
            </div>
            <div id="vehicle-list"></div>
        </div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value" id="raw-speed">0.00</div>
                <div class="stat-label">Raw Speed (m/s)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="filtered-speed">0.00</div>
                <div class="stat-label">Filtered Speed (m/s)</div>
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
        let currentVehicleId = null;
        
        const speedCtx = document.getElementById('speedChart').getContext('2d');
        const accelCtx = document.getElementById('accelChart').getContext('2d');
        
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: true,
            animation: false,
            scales: {
                x: { 
                    display: false,
                    grid: { color: '#444' }
                },
                y: { 
                    grid: { color: '#444' },
                    ticks: { color: '#fff' }
                }
            },
            plugins: {
                legend: { 
                    labels: { color: '#fff' }
                }
            }
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
                        label: 'Filtered Speed',
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
                    title: {
                        display: true,
                        text: 'Speed Comparison (m/s)',
                        color: '#fff'
                    }
                }
            }
        });

        const accelChart = new Chart(accelCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Filtered Acceleration',
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
                    title: {
                        display: true,
                        text: 'Filtered Acceleration (m/s²)',
                        color: '#fff'
                    }
                }
            }
        });

        function trackVehicle() {
            const id = parseInt(document.getElementById('vehicle-id').value);
            if (isNaN(id)) return;
            
            fetch('/track/' + id, { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    currentVehicleId = id;
                    document.getElementById('status').textContent = 'Tracking vehicle ' + id;
                });
        }

        socket.on('update', (data) => {
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
            }
        });

        socket.on('vehicles', (vehicles) => {
            const list = document.getElementById('vehicle-list');
            list.innerHTML = vehicles.map(id => 
                `<div class="vehicle-item ${id === currentVehicleId ? 'active' : ''}" 
                      onclick="document.getElementById('vehicle-id').value=${id}; trackVehicle();">
                    Vehicle ID: ${id}
                </div>`
            ).join('');
        });
    </script>
</body>
</html>
"""
    
    def _emit_data(self):
        """Background thread to emit data to clients"""
        while self.running:
            with self.lock:
                if self.tracked_vehicle_id is not None and self.tracked_vehicle_id in self.vehicle_data:
                    data = self.vehicle_data[self.tracked_vehicle_id]
                    self.socketio.emit('update', {
                        'labels': list(data["timestamps"]),
                        'raw_speed': list(data["raw_speed"]),
                        'filtered_speed': list(data["filtered_speed"]),
                        'filtered_accel': list(data["filtered_accel"])
                    })
                
                self.socketio.emit('vehicles', list(self.available_vehicles))
            
            time.sleep(0.1)
    
    def _calculate_raw_values(self, position_history, previous_speed):
        """Direct calculation without filtering"""
        if len(position_history) < 2:
            return 0.0, 0.0
        
        mid = len(position_history) // 2
        old_half = list(position_history)[:mid]
        new_half = list(position_history)[mid:]
        
        old_time = sum(t for t, _ in old_half) / len(old_half)
        old_x = sum(p.x for _, p in old_half) / len(old_half)
        old_y = sum(p.y for _, p in old_half) / len(old_half)
        old_z = sum(p.z for _, p in old_half) / len(old_half)
        
        new_time = sum(t for t, _ in new_half) / len(new_half)
        new_x = sum(p.x for _, p in new_half) / len(new_half)
        new_y = sum(p.y for _, p in new_half) / len(new_half)
        new_z = sum(p.z for _, p in new_half) / len(new_half)
        
        time_diff = new_time - old_time
        if time_diff < 0.01:
            return 0.0, 0.0
        
        distance = math.sqrt(
            (new_x - old_x) ** 2 +
            (new_y - old_y) ** 2 +
            (new_z - old_z) ** 2
        )
        
        if distance > 50:
            return 0.0, 0.0
        
        speed = distance / time_diff
        accel = (speed - previous_speed) / time_diff
        return speed, accel
    
    def push_data(self, vehicle_id, speed, acceleration, raw_speed):
        """Push raw vs filtered speed and filtered acceleration for a vehicle."""
        with self.lock:
            self.available_vehicles.add(vehicle_id)
            
            data = self.vehicle_data[vehicle_id]
            current_time = time.time()
            
            data["timestamps"].append(current_time)
            data["filtered_speed"].append(speed)
            data["filtered_accel"].append(acceleration)
            data["raw_speed"].append(raw_speed)
            data["last_update"] = current_time
    
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
            # Requests SocketIO to stop the underlying server.
            self.socketio.stop()
        except Exception:
            pass

if __name__ == "__main__":
    visualizer = RadarVisualizer(port=5000)
    visualizer.start()