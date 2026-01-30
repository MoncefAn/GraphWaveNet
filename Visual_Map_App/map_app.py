"""
Interactive Traffic Prediction Dashboard using Streamlit
Run with: streamlit run traffic_dashboard.py

BEST FOR: Live demos, presentations, interactive exploration
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

from pathlib import Path



# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Graph WaveNet Traffic Predictions",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# DATA LOADING (CACHED)
# ============================================================================

@st.cache_data
def load_data():
    """Load and cache all data"""

    # Resolve repo root
    BASE_DIR = Path(__file__).resolve().parent.parent

    PREDICTIONS_FILE = (
        BASE_DIR / "Optimized" / "plots" / "final_predictions_20260125_185830.npz"
    )
    SENSOR_LOCATIONS = (
        BASE_DIR / "dcrnn_data-main" / "metr_la" / "graph_sensor_locations.csv"
    )
    SCALER_PATH = (
        BASE_DIR / "dcrnn_data-main" / "metr_la" / "processed_new" / "scaler.pkl"
    )

    # --- sanity check (optional but helpful) ---
    for p in [PREDICTIONS_FILE, SENSOR_LOCATIONS, SCALER_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    # Load predictions
    data = np.load(PREDICTIONS_FILE)
    predictions_norm = data['predictions']
    targets_norm = data['targets']
    
    # Inverse transform
    if predictions_norm.ndim == 4:
        S, H, N, _ = predictions_norm.shape
        pred_flat = predictions_norm[..., 0].reshape(-1, 1)
        target_flat = targets_norm[..., 0].reshape(-1, 1)
    else:
        S, H, N = predictions_norm.shape
        pred_flat = predictions_norm.reshape(-1, 1)
        target_flat = targets_norm.reshape(-1, 1)
    
    predictions = scaler.inverse_transform(pred_flat).reshape(S, H, N)
    targets = scaler.inverse_transform(target_flat).reshape(S, H, N)
    
    # Load sensor locations
    sensor_coords = pd.read_csv(SENSOR_LOCATIONS)
    if 'index' in sensor_coords.columns:
        sensor_coords = sensor_coords.drop(columns=['index'])
    sensor_coords.columns = ['sensor_id', 'lat', 'lon']
    
    return predictions, targets, sensor_coords


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

def create_sidebar(predictions):
    """Create sidebar with controls"""
    st.sidebar.title("🎛️ Controls")
    
    S, H, N = predictions.shape
    
    # Sample selection
    sample_idx = st.sidebar.slider(
        "Test Sample",
        0, S-1, 0,
        help="Select which test sample to visualize"
    )
    
    # Horizon selection
    horizon_minutes = st.sidebar.slider(
        "Prediction Horizon (minutes)",
        5, H*5, 30, step=5,
        help="How far ahead to predict"
    )
    horizon_idx = horizon_minutes // 5 - 1
    
    # Visualization mode
    viz_mode = st.sidebar.selectbox(
        "Visualization Mode",
        ["Interactive Map", "Comparison View", "Time Series", "Error Analysis"],
        help="Choose visualization type"
    )
    
    # Speed range
    st.sidebar.subheader("Display Settings")
    speed_range = st.sidebar.slider(
        "Speed Range (mph)",
        0, 100, (0, 70),
        help="Adjust color scale range"
    )
    
    # Show statistics
    show_stats = st.sidebar.checkbox("Show Statistics", value=True)
    
    return sample_idx, horizon_idx, viz_mode, speed_range, show_stats


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_interactive_map(predictions, targets, sensor_coords, sample_idx, horizon_idx, speed_range):
    """Create interactive map with predictions"""
    pred_speeds = predictions[sample_idx, horizon_idx, :]
    true_speeds = targets[sample_idx, horizon_idx, :]
    errors = np.abs(pred_speeds - true_speeds)
    
    df = sensor_coords.copy()
    df['pred_speed'] = pred_speeds
    df['true_speed'] = true_speeds
    df['error'] = errors
    
    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='pred_speed',
        size='error',
        hover_name='sensor_id',
        hover_data={
            'pred_speed': ':.1f',
            'true_speed': ':.1f',
            'error': ':.1f',
            'lat': False,
            'lon': False
        },
        color_continuous_scale='RdYlGn_r',
        range_color=speed_range,
        size_max=25,
        zoom=10,
        mapbox_style='open-street-map',
        title=f'Predictions: {(horizon_idx+1)*5} minutes ahead'
    )
    
    fig.update_layout(height=600)
    return fig


def plot_comparison_view(predictions, targets, sensor_coords, sample_idx, horizon_idx, speed_range):
    """Side-by-side comparison of predictions vs ground truth"""
    pred_speeds = predictions[sample_idx, horizon_idx, :]
    true_speeds = targets[sample_idx, horizon_idx, :]
    errors = np.abs(pred_speeds - true_speeds)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Predictions', 'Ground Truth', 'Absolute Error'],
        specs=[[{'type': 'scattermapbox'}]*3]
    )
    
    # Predictions
    fig.add_trace(
        go.Scattermapbox(
            lat=sensor_coords['lat'],
            lon=sensor_coords['lon'],
            mode='markers',
            marker=dict(
                size=12,
                color=pred_speeds,
                colorscale='RdYlGn_r',
                cmin=speed_range[0],
                cmax=speed_range[1],
                colorbar=dict(title="mph", x=0.28)
            ),
            text=[f"S{sid}: {speed:.1f} mph" for sid, speed in 
                  zip(sensor_coords['sensor_id'], pred_speeds)],
            hoverinfo='text'
        ),
        row=1, col=1
    )
    
    # Ground Truth
    fig.add_trace(
        go.Scattermapbox(
            lat=sensor_coords['lat'],
            lon=sensor_coords['lon'],
            mode='markers',
            marker=dict(
                size=12,
                color=true_speeds,
                colorscale='RdYlGn_r',
                cmin=speed_range[0],
                cmax=speed_range[1],
                showscale=False
            ),
            text=[f"S{sid}: {speed:.1f} mph" for sid, speed in 
                  zip(sensor_coords['sensor_id'], true_speeds)],
            hoverinfo='text'
        ),
        row=1, col=2
    )
    
    # Errors
    fig.add_trace(
        go.Scattermapbox(
            lat=sensor_coords['lat'],
            lon=sensor_coords['lon'],
            mode='markers',
            marker=dict(
                size=12,
                color=errors,
                colorscale='Reds',
                cmin=0,
                cmax=10,
                colorbar=dict(title="Error", x=0.95)
            ),
            text=[f"S{sid}: {err:.1f} mph error" for sid, err in 
                  zip(sensor_coords['sensor_id'], errors)],
            hoverinfo='text'
        ),
        row=1, col=3
    )
    
    # Update mapboxes
    center_lat = sensor_coords['lat'].mean()
    center_lon = sensor_coords['lon'].mean()
    
    for i in range(1, 4):
        fig.update_mapboxes(
            style='open-street-map',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10,
            row=1, col=i
        )
    
    fig.update_layout(height=500, showlegend=False)
    return fig


def plot_time_series(predictions, targets, sample_idx, selected_sensor):
    """Plot time series for a specific sensor"""
    H = predictions.shape[1]
    time_steps = np.arange(H) * 5
    
    pred = predictions[sample_idx, :, selected_sensor]
    true = targets[sample_idx, :, selected_sensor]
    
    fig = go.Figure()
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=pred,
        mode='lines+markers',
        name='Prediction',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Ground Truth
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=true,
        mode='lines+markers',
        name='Ground Truth',
        line=dict(color='blue', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Shaded error region
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_steps, time_steps[::-1]]),
        y=np.concatenate([pred, true[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Sensor {selected_sensor} - Prediction vs Ground Truth',
        xaxis_title='Time Ahead (minutes)',
        yaxis_title='Speed (mph)',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_error_analysis(predictions, targets, sample_idx):
    """Error distribution analysis"""
    errors = np.abs(predictions[sample_idx] - targets[sample_idx])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Error Distribution', 'Error by Horizon'],
        specs=[[{'type': 'histogram'}, {'type': 'box'}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=errors.flatten(),
            nbinsx=50,
            marker_color='indianred',
            name='Errors'
        ),
        row=1, col=1
    )
    
    # Box plot by horizon
    H = errors.shape[0]
    for h in range(H):
        fig.add_trace(
            go.Box(
                y=errors[h, :],
                name=f'{(h+1)*5}min',
                marker_color=px.colors.sequential.Reds[min(h, 8)]
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Error (mph)", row=1, col=1)
    fig.update_xaxes(title_text="Prediction Horizon", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Error (mph)", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=False)
    return fig





# ============================================================================
# METRICS DISPLAY
# ============================================================================

def display_metrics(predictions, targets, sample_idx, horizon_idx):
    """Display key metrics"""
    pred = predictions[sample_idx, horizon_idx, :]
    true = targets[sample_idx, horizon_idx, :]
    
    mae = np.abs(pred - true).mean()
    rmse = np.sqrt(((pred - true)**2).mean())
    mape = np.mean(np.abs((pred - true) / (true + 1e-5))) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{mae:.2f} mph", help="Mean Absolute Error")
    
    with col2:
        st.metric("RMSE", f"{rmse:.2f} mph", help="Root Mean Squared Error")
    
    with col3:
        st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")
    
    with col4:
        st.metric("Sensors", len(pred), help="Number of sensors")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🚗 Graph WaveNet Traffic Predictions")
    st.markdown("**Los Angeles Traffic Network - Interactive Dashboard**")
    
    # Load data
    with st.spinner("Loading data..."):
        predictions, targets, sensor_coords = load_data()
    
    # Sidebar
    sample_idx, horizon_idx, viz_mode, speed_range, show_stats = create_sidebar(predictions)
    
    # Display metrics
    if show_stats:
        st.subheader("📊 Performance Metrics")
        display_metrics(predictions, targets, sample_idx, horizon_idx)
        st.divider()
    
    # Main visualization
    st.subheader(f"📍 {viz_mode}")
    
    if viz_mode == "Interactive Map":
        fig = plot_interactive_map(predictions, targets, sensor_coords, 
                                   sample_idx, horizon_idx, speed_range)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_mode == "Comparison View":
        fig = plot_comparison_view(predictions, targets, sensor_coords,
                                   sample_idx, horizon_idx, speed_range)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_mode == "Time Series":
        # Sensor selector
        selected_sensor = st.selectbox(
            "Select Sensor",
            range(predictions.shape[2]),
            format_func=lambda x: f"Sensor {sensor_coords.iloc[x]['sensor_id']}"
        )
        
        fig = plot_time_series(predictions, targets, sample_idx, selected_sensor)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_mode == "Error Analysis":
        fig = plot_error_analysis(predictions, targets, sample_idx)
        st.plotly_chart(fig, use_container_width=True)
    
    
    # Footer
    st.divider()
    st.caption("Graph WaveNet Implementation | Created with Streamlit & Plotly")


if __name__ == "__main__":
    main()
