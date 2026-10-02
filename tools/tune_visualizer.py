"""Live tuning visualizer for accel_to_pedals_debug.csv. Run from the project root:
.venv/Scripts/python tools/tune_visualizer.py Then open http://localhost:8050 in your browser.
Shows the last N seconds of data (default 60s, set with --window)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

_DEFAULT_CSV = Path(__file__).resolve().parents[1] / "accel_to_pedals_debug.csv"
_UPDATE_MS = 1000
_GEARSHIFT_COLOR = "rgba(255,153,51,0.15)"


def _load(path: Path, window_s: float) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    for col in df.columns:
        if col not in ("utc", "pedal_state"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "t_s" not in df.columns or df.empty:
        return df
    latest_t = df["t_s"].dropna().iloc[-1] if df["t_s"].notna().any() else None
    if latest_t is None:
        return df
    return df[df["t_s"] >= latest_t - window_s].copy()


def _gearshift_shapes(df: pd.DataFrame) -> list[dict]:
    if "gearshift_active" not in df.columns:
        return []
    shapes = []
    in_shift = False
    start = None
    for _, row in df.iterrows():
        active = row["gearshift_active"] == 1
        if active and not in_shift:
            start = row["t_s"]
            in_shift = True
        elif not active and in_shift:
            shapes.append(dict(type="rect", x0=start, x1=row["t_s"],
                               xref="x", yref="paper", y0=0, y1=1,
                               fillcolor=_GEARSHIFT_COLOR, line_width=0, layer="below"))
            in_shift = False
    if in_shift and start is not None:
        shapes.append(dict(type="rect", x0=start, x1=df["t_s"].max(),
                           xref="x", yref="paper", y0=0, y1=1,
                           fillcolor=_GEARSHIFT_COLOR, line_width=0, layer="below"))
    return shapes


def _build_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty or "t_s" not in df.columns:
        return go.Figure()

    t = df["t_s"]
    shapes = _gearshift_shapes(df)

    fig = go.Figure()

    # Panel 1: Acceleration
    traces_p1 = []
    if "wanted_smooth" in df.columns:
        traces_p1.append(go.Scatter(x=t, y=df["wanted_smooth"], name="wanted_smooth",
                                    line=dict(color="steelblue", width=1.5), yaxis="y1"))
    if "raw_smooth" in df.columns:
        traces_p1.append(go.Scatter(x=t, y=df["raw_smooth"], name="raw_smooth",
                                    line=dict(color="darkorange", width=1), opacity=0.8, yaxis="y1"))
    if "error_ms2" in df.columns:
        traces_p1.append(go.Scatter(x=t, y=df["error_ms2"], name="error (raw−wanted)",
                                    line=dict(color="purple", width=0.8), opacity=0.7, yaxis="y1"))
    if "road_load_ms2" in df.columns:
        traces_p1.append(go.Scatter(x=t, y=df["road_load_ms2"], name="road_load",
                                    line=dict(color="gray", width=0.7, dash="dash"), opacity=0.6, yaxis="y1"))

    # Panel 2: Gas PID
    traces_p2 = []
    if "gas_p" in df.columns:
        traces_p2.append(go.Scatter(x=t, y=df["gas_p"], name="P",
                                    line=dict(color="steelblue", width=1), yaxis="y2"))
    if "gas_i" in df.columns:
        traces_p2.append(go.Scatter(x=t, y=df["gas_i"], name="I",
                                    line=dict(color="crimson", width=1.2), yaxis="y2"))
    if "gas_d" in df.columns:
        traces_p2.append(go.Scatter(x=t, y=df["gas_d"], name="D",
                                    line=dict(color="green", width=0.8), opacity=0.8, yaxis="y2"))
    if "gas_cmd" in df.columns:
        traces_p2.append(go.Scatter(x=t, y=df["gas_cmd"], name="gas_cmd",
                                    line=dict(color="black", width=1.4, dash="dash"), yaxis="y2"))

    # Panel 3: Brake / pedal commands
    traces_p3 = []
    if "brake_ff" in df.columns:
        traces_p3.append(go.Scatter(x=t, y=df["brake_ff"], name="brake FF",
                                    line=dict(color="salmon", width=1), yaxis="y3"))
    if "brake_trim_i" in df.columns:
        traces_p3.append(go.Scatter(x=t, y=df["brake_trim_i"], name="brake trim I",
                                    line=dict(color="firebrick", width=0.8), opacity=0.8, yaxis="y3"))
    if "brake_cmd" in df.columns:
        traces_p3.append(go.Scatter(x=t, y=-df["brake_cmd"], name="−brake_cmd",
                                    line=dict(color="red", width=1.2), yaxis="y3"))
    if "gas_cmd" in df.columns:
        traces_p3.append(go.Scatter(x=t, y=df["gas_cmd"], name="gas_cmd (p3)",
                                    line=dict(color="green", width=1.2), yaxis="y3"))
    if "game_throttle" in df.columns:
        traces_p3.append(go.Scatter(x=t, y=df["game_throttle"], name="game_throttle",
                                    line=dict(color="limegreen", width=0.8, dash="dash"), opacity=0.8, yaxis="y3"))
    if "game_clutch" in df.columns:
        traces_p3.append(go.Scatter(x=t, y=df["game_clutch"], name="game_clutch",
                                    fill="tozeroy", fillcolor="rgba(255,153,51,0.2)",
                                    line=dict(color="rgba(255,153,51,0.4)", width=0.5), yaxis="y3"))

    # Panel 4: Speed / gear / scales
    traces_p4 = []
    if "speed_ms" in df.columns:
        traces_p4.append(go.Scatter(x=t, y=df["speed_ms"], name="speed (m/s)",
                                    line=dict(color="navy", width=1.2), yaxis="y4"))
    if "gain_scale" in df.columns:
        traces_p4.append(go.Scatter(x=t, y=df["gain_scale"], name="gain_scale",
                                    line=dict(color="darkorange", width=0.8, dash="dash"), yaxis="y5"))
    if "brake_multiplier" in df.columns:
        traces_p4.append(go.Scatter(x=t, y=df["brake_multiplier"], name="brake_mult",
                                    line=dict(color="firebrick", width=0.8, dash="dash"), yaxis="y5"))
    if "gear" in df.columns:
        traces_p4.append(go.Scatter(x=t, y=df["gear"] * 0.1, name="gear×0.1",
                                    mode="markers", marker=dict(color="gray", size=3, opacity=0.4), yaxis="y5"))

    all_traces = traces_p1 + traces_p2 + traces_p3 + traces_p4

    # Assign each panel to a subplot row via domain
    layout = go.Layout(
        height=900,
        title="accel_to_pedals live tuning",
        shapes=shapes,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
        margin=dict(l=60, r=60, t=50, b=80),
        xaxis=dict(domain=[0, 1], anchor="y4", title="t (s)", showgrid=True),
        yaxis=dict(domain=[0.77, 1.0], title="Accel (m/s²)", showgrid=True),
        yaxis2=dict(domain=[0.52, 0.75], title="Gas PID", showgrid=True),
        yaxis3=dict(domain=[0.27, 0.50], title="Brake/pedal", showgrid=True, range=[-1.05, 1.05]),
        yaxis4=dict(domain=[0.0, 0.25], title="Speed (m/s)", showgrid=True),
        yaxis5=dict(domain=[0.0, 0.25], overlaying="y4", side="right", title="scale/gear"),
    )

    fig = go.Figure(data=all_traces, layout=layout)

    # Fix x-axis references so all panels share the same x axis
    for trace in fig.data:
        trace.xaxis = "x"

    return fig


def build_app(csv_path: Path, window_s: float, update_ms: int) -> Dash:
    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="main-graph", style={"height": "90vh"}),
        dcc.Interval(id="interval", interval=update_ms, n_intervals=0),
        dcc.Store(id="config", data={"csv": str(csv_path), "window": window_s}),
    ])

    @app.callback(
        Output("main-graph", "figure"),
        Input("interval", "n_intervals"),
        Input("config", "data"),
    )
    def refresh(_n: int, config: dict) -> go.Figure:
        df = _load(Path(config["csv"]), config["window"])
        return _build_figure(df)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Live visualizer for accel_to_pedals_debug.csv")
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV, help="Path to CSV file")
    parser.add_argument("--window", type=float, default=60.0, help="Time window in seconds (default 60)")
    parser.add_argument("--interval", type=int, default=_UPDATE_MS, help="Refresh interval ms (default 1000)")
    parser.add_argument("--port", type=int, default=8050, help="Port (default 8050)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        print("Start MonoCruise with cruise control active to generate the file.")
        sys.exit(1)

    app = build_app(args.csv, args.window, args.interval)
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()

