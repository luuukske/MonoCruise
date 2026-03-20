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
        self.vehicle_timeout = 5.0  # Remove vehicles not updated for 5 seconds
        
        self._setup_routes()
        
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(self._get_html_template())
        
        @self.app.route('/track/<int:vehicle_id>', methods=['POST'])
        def track_vehicle(vehicle_id):
            with self.lock:
                if vehicle_id not in self.available_vehicles:
                    return jsonify({"status": "error", "message": f"Vehicle {vehicle_id} not available"}), 400
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
                <button id="track-btn">Track</button>
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
        
        // Connection diagnostics
        socket.on('connect', () => {
            console.log('✓ Socket connected');
        });
        
        socket.on('disconnect', () => {
            console.log('✗ Socket disconnected');
        });
        
        // Attach persistent listener to vehicle list - do this ONCE, early
        const vehicleListContainer = document.getElementById('vehicle-list');
        vehicleListContainer.addEventListener('click', (e) => {
            const vehicleItem = e.target.closest('.vehicle-item');
            if (!vehicleItem) return;
            
            const vehicleId = parseInt(vehicleItem.dataset.vehicleId);
            console.log('✓ Vehicle clicked:', vehicleId);
            document.getElementById('vehicle-id').value = vehicleId;
            trackVehicle();
        });
        console.log('Vehicle list listener attached');
        
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
                scales: {
                    ...commonOptions.scales,
                    y: {
                        ...commonOptions.scales.y,
                        beginAtZero: true,
                        ticks: { color: '#fff' }
                    }
                },
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
            const inputField = document.getElementById('vehicle-id');
            const id = parseInt(inputField.value);
            
            console.log('trackVehicle called. Input value:', inputField.value, 'Parsed ID:', id);
            
            if (isNaN(id)) {
                document.getElementById('status').textContent = '❌ Invalid ID';
                console.log('ID is NaN, aborting');
                return;
            }
            
            console.log('Sending fetch for vehicle:', id);
            fetch('/track/' + id, { method: 'POST' })
                .then(r => {
                    console.log('Response status:', r.status);
                    return r.json().then(data => ({ ok: r.ok, status: r.status, ...data }));
                })
                .then(data => {
                    console.log('Response data:', data);
                    if (data.ok) {
                        currentVehicleId = id;
                        document.getElementById('status').textContent = '✓ Tracking vehicle ' + id;
                    } else {
                        document.getElementById('status').textContent = '❌ ' + (data.message || 'Unknown error');
                    }
                })
                .catch(err => {
                    console.error('Fetch error:', err);
                    document.getElementById('status').textContent = '❌ Request failed';
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
                `<div class="vehicle-item" data-vehicle-id="${id}" 
                      style="opacity: ${id === currentVehicleId ? '1' : '0.8'}; cursor: pointer;">
                    Vehicle ID: ${id}
                </div>`
            ).join('');
        });
        
        // Allow Enter key in input field
        document.getElementById('vehicle-id').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                console.log('Enter pressed');
                trackVehicle();
            }
        });
        
        // Track button click
        document.getElementById('track-btn').addEventListener('click', () => {
            console.log('Track button clicked');
            trackVehicle();
        });
    </script>
</body>
</html>
"""
    
    def _emit_data(self):
        """Background thread to emit data to clients"""
        last_vehicles = None
        while self.running:
            with self.lock:
                current_time = time.time()
                
                # Remove stale vehicles
                stale_vehicles = [
                    vid for vid in self.available_vehicles
                    if current_time - self.vehicle_data[vid]["last_update"] > self.vehicle_timeout
                ]
                for vid in stale_vehicles:
                    self.available_vehicles.discard(vid)
                    if self.tracked_vehicle_id == vid:
                        self.tracked_vehicle_id = None
                
                if self.tracked_vehicle_id is not None and self.tracked_vehicle_id in self.vehicle_data:
                    data = self.vehicle_data[self.tracked_vehicle_id]
                    self.socketio.emit('update', {
                        'labels': list(data["timestamps"]),
                        'raw_speed': list(data["raw_speed"]),
                        'filtered_speed': list(data["filtered_speed"]),
                        'filtered_accel': list(data["filtered_accel"])
                    })
                
                current_vehicles = sorted(list(self.available_vehicles))
                if current_vehicles != last_vehicles:
                    self.socketio.emit('vehicles', current_vehicles)
                    last_vehicles = current_vehicles
            
            time.sleep(0.1)
    
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