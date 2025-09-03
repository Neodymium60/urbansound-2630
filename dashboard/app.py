#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from flask import send_from_directory
from dash import Dash, html, dcc, dash_table, Input, Output, State
import plotly.graph_objects as go

DATA_DIR  = os.getenv("DATA_DIR", "/data")
CSV_PATH  = os.path.join(DATA_DIR, "labels.csv")
HOST      = os.getenv("HOST", "0.0.0.0")
PORT      = int(os.getenv("PORT", "5000"))
SITE_ID   = os.getenv("SITE_ID", None)  # optional default filter

app = Dash(__name__)
server = app.server  # for gunicorn etc.

# Serve WAVs so users can listen
@server.route("/audio/<path:fname>")
def audio(fname):
    return send_from_directory(DATA_DIR, fname, as_attachment=False)

def load_df():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        cols = ["timestamp", "filename", "label", "confidence", "site_id",
                "dbfs_rms", "dbfs_peak", "spl_est_dbA"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(CSV_PATH)
    # types
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ("confidence", "dbfs_rms", "dbfs_peak", "spl_est_dbA"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # sort newest first
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)
    return df

def site_options(df):
    vals = sorted([v for v in df["site_id"].dropna().unique().tolist() if v != ""])
    if not vals:
        return []
    return [{"label": v, "value": v} for v in vals]

def label_options(df):
    vals = sorted([v for v in df["label"].dropna().unique().tolist() if v != ""])
    return [{"label": v, "value": v} for v in vals]

def make_table(df):
    # Limit rows for the table
    cols = ["timestamp", "filename", "label", "confidence", "site_id",
            "dbfs_rms", "dbfs_peak", "spl_est_dbA"]
    present = [c for c in cols if c in df.columns]
    show = df[present].copy()
    # Add link to audio
    if "filename" in show.columns:
        show["audio"] = show["filename"].apply(lambda f: f"/audio/{f}")
        present.append("audio")
    return dash_table.DataTable(
        data=show.head(200).to_dict("records"),
        columns=[{"name": c, "id": c, "presentation": "markdown"} if c == "audio"
                 else {"name": c, "id": c} for c in present],
        style_table={"overflowX": "auto", "maxHeight": "60vh", "overflowY": "auto"},
        style_cell={"padding": "6px", "fontSize": "14px"},
        sort_action="native",
        filter_action="native",
        page_action="none",
    )

def timeseries_figure(df):
    # Prefer spl_est_dbA; fall back to dbfs_rms
    ycol = "spl_est_dbA" if "spl_est_dbA" in df.columns and df["spl_est_dbA"].notna().any() else "dbfs_rms"
    if "timestamp" not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), title="No data")
        return fig
    dfx = df.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfx["timestamp"], y=dfx[ycol], mode="lines+markers", name=ycol
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=10,r=10,t=30,b=10),
        title=f"{ycol} over time (latest first)",
        xaxis_title="time",
        yaxis_title=ycol,
    )
    return fig

def label_bar_figure(df):
    if "label" not in df.columns or df.empty:
        return go.Figure()
    counts = df["label"].value_counts().sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.values.tolist()))
    fig.update_layout(
        height=320,
        margin=dict(l=10,r=10,t=30,b=10),
        title="Label counts (last 200)",
        xaxis_title="label",
        yaxis_title="count",
    )
    return fig

app.layout = html.Div([
    html.H2("UrbanSound – Dashboard"),
    html.Div([
        html.Div([
            html.Label("Site"),
            dcc.Dropdown(id="site-filter", options=[], value=SITE_ID, placeholder="All"),
        ], style={"width": "25%", "display": "inline-block", "paddingRight": "10px"}),
        html.Div([
            html.Label("Label"),
            dcc.Dropdown(id="label-filter", options=[], value=None, placeholder="All"),
        ], style={"width": "25%", "display": "inline-block", "paddingRight": "10px"}),
        html.Div([
            html.Label("Auto-refresh (s)"),
            dcc.Input(id="refresh-sec", type="number", value=5, min=1, step=1, style={"width": "100%"}),
        ], style={"width": "15%", "display": "inline-block", "paddingRight": "10px"}),
        html.Div([
            html.Button("Refresh now", id="refresh-btn"),
        ], style={"width": "15%", "display": "inline-block"}),
    ], style={"marginBottom": "10px"}),
    dcc.Interval(id="tick", interval=5000, n_intervals=0),
    html.Div(id="cards", children=[
        html.Div([
            dcc.Graph(id="timeseries"),
        ], style={"width": "59%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            dcc.Graph(id="labelbar"),
        ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "1%"}),
    ]),
    html.Hr(),
    html.Div(id="table-wrap"),
    html.Div(id="footer", style={"color": "#777", "fontSize": "12px", "marginTop": "8px"}),
], style={"padding": "16px", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif"})

@app.callback(
    Output("site-filter", "options"),
    Output("label-filter", "options"),
    Output("timeseries", "figure"),
    Output("labelbar", "figure"),
    Output("table-wrap", "children"),
    Output("footer", "children"),
    Input("tick", "n_intervals"),
    Input("refresh-btn", "n_clicks"),
    State("site-filter", "value"),
    State("label-filter", "value"),
    State("refresh-sec", "value"),
)
def update(_n, _clicks, site_val, label_val, refresh_val):
    # Adjust interval
    if isinstance(refresh_val, (int, float)) and refresh_val > 0:
        app.callbacks._callbacks["tick.interval"]["state"]["tick.interval"]["value"] = int(refresh_val * 1000)

    df = load_df()
    total = len(df)
    # Filter by site/label
    if site_val:
        df = df[df["site_id"] == site_val]
    if label_val:
        df = df[df["label"] == label_val]

    # Build controls & visuals
    sopts = site_options(load_df())
    lopts = label_options(load_df())

    fig_ts = timeseries_figure(df.head(200))
    fig_lb = label_bar_figure(df.head(200))

    table = make_table(df)

    footer = f"{datetime.now().astimezone().isoformat(timespec='seconds')} — showing {len(df)} of {total} rows"
    return sopts, lopts, fig_ts, fig_lb, table, footer

if __name__ == "__main__":
    # Ensure data dir exists so /audio routes don't 404 on directory
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    app.run_server(host=HOST, port=PORT, debug=False)

