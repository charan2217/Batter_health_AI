import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from models.battery import BatteryPack
import pandas as pd
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Advanced Battery Health Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 2px solid #4CAF50;
        background-color: #1E1E1E;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .prediction-message {
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        margin: 5px 0;
        background: rgba(76, 175, 80, 0.1);
        border-radius: 0 8px 8px 0;
    }
    
    .prediction-message.warning {
        border-left-color: #FFC107;
        background: rgba(255, 193, 7, 0.1);
    }
    
    .prediction-message.danger {
        border-left-color: #F44336;
        background: rgba(244, 67, 54, 0.1);
    }
    
    .cooling-animation {
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .cooling-active {
        color: #4CAF50;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'battery' not in st.session_state:
    st.session_state.battery = BatteryPack(rows=5, cols=5, use_pinn=True, ambient_temp=25.0)
    st.session_state.history = {
        'time': [],
        'max_temp': [],
        'avg_temp': [],
        'min_voltage': [],
        'max_current': [],
        'cooling_status': []
    }
    st.session_state.time_elapsed = 0.0
    st.session_state.running = False
    st.session_state.last_update = time.time()
    st.session_state.fan_speed = 0
    st.session_state.pump_level = 0

# Helper functions
def get_cooling_animation(speed, component='fan', state=0):
    """Generate cooling component animation based on speed and state"""
    if component == 'fan':
        icon = "🔄"
        if state == 1:  # Starting
            return f"<span class='cooling-animation'>{icon} Starting...</span>"
        elif state == 2:  # On
            return f"<span class='cooling-animation cooling-active'>{icon} Fan: {speed}% (Active)</span>"
        elif state == 3:  # Stopping
            return f"<span class='cooling-animation'>{icon} Cooling down...</span>"
        else:  # Off
            return f"<span class='cooling-animation'>{icon} Fan: {speed}% (Idle)</span>"
    else:
        icon = "💧"
        if state == 1:  # Starting
            return f"<span class='cooling-animation'>{icon} Starting...</span>"
        elif state == 2:  # On
            return f"<span class='cooling-animation cooling-active'>{icon} Pump: {speed}% (Active)</span>"
        elif state == 3:  # Stopping
            return f"<span class='cooling-animation'>{icon} Draining...</span>"
        else:  # Off
            return f"<span class='cooling-animation'>{icon} Pump: {speed}% (Idle)</span>"

def update_simulation(dt=1.0):
    """Update the battery simulation state"""
    if not st.session_state.running:
        return None
        
    try:
        battery = st.session_state.battery
        
        # Update simulation time
        current_time = time.time()
        elapsed = min(current_time - st.session_state.last_update, 0.1)  # Cap at 100ms
        st.session_state.last_update = current_time
        st.session_state.time_elapsed += elapsed
        
        # Update battery state
        battery.update(dt)
        
        # Get current state
        state = battery.get_state()
        
        # Update history
        st.session_state.history['time'].append(st.session_state.time_elapsed)
        st.session_state.history['max_temp'].append(state['max_temperature'])
        st.session_state.history['avg_temp'].append(state['avg_temperature'])
        st.session_state.history['min_voltage'].append(min(cell['voltage'] for cell in state['cells']))
        st.session_state.history['max_current'].append(max(abs(cell['current']) for cell in state['cells']))
        
        # Update cooling status and animation state
        fan_speed = state.get('fan_speed', 0)
        pump_level = state.get('pump_level', 0)
        cooling_state = state.get('cooling_animation_state', 0)
        
        st.session_state.fan_speed = fan_speed
        st.session_state.pump_level = pump_level
        st.session_state.cooling_state = cooling_state
        
        # Get the latest state from the battery
        state = st.session_state.battery.get_state()
        
        # Ensure prediction_messages is always a list and properly populated
        if hasattr(st.session_state.battery, 'prediction_messages'):
            # Get the latest messages from the battery instance
            state['prediction_messages'] = st.session_state.battery.prediction_messages
        else:
            # Initialize empty list if not exists
            st.session_state.battery.prediction_messages = []
            state['prediction_messages'] = []
        
        # Store the full state in session for access in other parts of the app
        st.session_state.battery_state = state
        
        # Store last prediction if available
        if state['prediction_messages']:
            st.session_state.last_prediction = state['prediction_messages'][-1]
        elif 'last_prediction' in state and state['last_prediction']:
            st.session_state.last_prediction = state['last_prediction']
        
        return state
        
    except Exception as e:
        st.error(f"Error updating simulation: {str(e)}")
        return None

def create_heatmap_figure(temp_grid, max_temp):
    """Create an interactive heatmap figure"""
    # Ensure temp_grid is a numpy array
    if not isinstance(temp_grid, np.ndarray):
        temp_grid = np.array(temp_grid)
        
    fig = go.Figure(data=go.Heatmap(
        z=temp_grid,
        colorscale='jet',
        zmin=20,  # Min temperature
        zmax=80,  # Max temperature
        colorbar=dict(title='°C'),
        hoverongaps=False,
        showscale=True
    ))
    
    fig.update_layout(
        title="Battery Pack Temperature Distribution",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        width=600,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        font=dict(color='white')
    )
    
    return fig

# Main layout
st.title("")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("" if not st.session_state.running else ""):
        st.session_state.running = not st.session_state.running
        st.session_state.last_update = time.time()
    
    st.markdown("### Simulation Speed")
    sim_speed = st.slider("Update interval (ms)", 100, 1000, 200, 100, key="sim_speed")
    
    st.markdown("### Cooling Controls")
    # Convert to float for consistent typing
    st.session_state.fan_speed = float(st.slider("Fan Speed (%)", 0.0, 100.0, float(st.session_state.fan_speed), 5.0))
    st.session_state.pump_level = float(st.slider("Pump Level (%)", 0.0, 100.0, float(st.session_state.pump_level), 5.0))
    
    # Update battery with cooling settings
    if 'battery' in st.session_state and hasattr(st.session_state.battery, 'rl_agent'):
        st.session_state.battery.rl_agent.set_cooling(
            st.session_state.fan_speed / 100,
            st.session_state.pump_level / 100
        )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Heatmap placeholder
    heatmap_placeholder = st.empty()
    
    # Metrics row
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown("<div class='metric-card'>"
                     "<h3></h3>"
                     "<h2>0.0°C</h2>"
                     "</div>", unsafe_allow_html=True)
    with metric_cols[1]:
        st.markdown("<div class='metric-card'>"
                     "<h3></h3>"
                     "<h2>0.0A</h2>"
                     "</div>", unsafe_allow_html=True)
    with metric_cols[2]:
        st.markdown("<div class='metric-card'>"
                     "<h3></h3>"
                     "<h2>0.0V</h2>"
                     "</div>", unsafe_allow_html=True)
    with metric_cols[3]:
        st.markdown("<div class='metric-card'>"
                     "<h3></h3>"
                     "<h2>0.0s</h2>"
                     "</div>", unsafe_allow_html=True)
    
    # Cooling status with animation
    st.markdown("### Cooling System Status")
    cooling_cols = st.columns(2)
    with cooling_cols[0]:
        fan_state = st.session_state.get('cooling_state', 0)
        st.markdown(f"<div class='metric-card'>{get_cooling_animation(st.session_state.fan_speed, 'fan', fan_state)}</div>", 
                   unsafe_allow_html=True)
    with cooling_cols[1]:
        pump_state = st.session_state.get('cooling_state', 0)
        st.markdown(f"<div class='metric-card'>{get_cooling_animation(st.session_state.pump_level, 'pump', pump_state)}</div>", 
                   unsafe_allow_html=True)
                   
    # AI Predictions and Alerts
    st.markdown("### 🔮 AI Predictions & Alerts")
    with st.expander("Live Battery Insights", expanded=True):
        # Initialize state if not exists
        state = getattr(st.session_state, 'battery_state', {})
        # Get messages from battery state if available
        if 'prediction_messages' in state and state['prediction_messages']:
            messages = state['prediction_messages']
            # Display up to 5 most recent messages
            for msg in reversed(messages[-5:]):
                # Get message and timestamp
                message = msg.get('message', '')
                timestamp = msg.get('timestamp', time.time())
                
                # Determine message class based on severity
                msg_class = 'info'
                if 'severity' in msg:
                    if msg['severity'] == 'high':
                        msg_class = 'warning'
                    elif msg['severity'] == 'critical':
                        msg_class = 'danger'
                
                # Format timestamp
                if isinstance(timestamp, (int, float)):
                    time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
                else:
                    time_str = '--:--:--'
                
                # Display message with timestamp and styling
                st.markdown(
                    f"<div class='prediction-message {msg_class}'>"
                    f"<small style='opacity:0.7;'>[{time_str}]</small><br>"
                    f"{message}"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No prediction messages available yet. The system is initializing...")
    # Temperature history chart
    st.markdown("### 📈 Temperature History")
    temp_chart = st.empty()
    
    # Risk assessment with dynamic colors
    st.markdown("### ⚠️ Risk Assessment")
    risk_cols = st.columns(2)
    
    # Get risk level from state if available
    risk_level = state.get('risk_level', 'LOW') if 'state' in locals() and state else 'LOW'
    health_status = state.get('health_status', 'GOOD') if 'state' in locals() and state else 'GOOD'
    
    # Set colors based on risk and health
    risk_color = '#4CAF50'  # Green
    if risk_level == 'MEDIUM':
        risk_color = '#FFC107'  # Yellow
    elif risk_level == 'HIGH':
        risk_color = '#FF9800'  # Orange
    elif risk_level == 'CRITICAL':
        risk_color = '#F44336'  # Red
        
    health_color = '#4CAF50'  # Green
    if health_status == 'FAIR':
        health_color = '#FFC107'  # Yellow
    elif health_status == 'POOR':
        health_color = '#F44336'  # Red
    
    with risk_cols[0]:
        st.markdown(
            f"<div class='metric-card' style='border-left: 4px solid {risk_color}'>"
            f"<h3>Thermal Risk</h3>"
            f"<h2 style='color: {risk_color}'>{risk_level}</h2>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with risk_cols[1]:
        st.markdown(
            f"<div class='metric-card' style='border-left: 4px solid {health_color}'>"
            f"<h3>Battery Health</h3>"
            f"<h2 style='color: {health_color}'>{health_status}</h2>"
            f"</div>", 
            unsafe_allow_html=True
        )
    
    # Thermal Analysis
    st.markdown("### 🔍 Thermal Analysis")
    if state and 'thermal_analysis' in state:
        analysis = state['thermal_analysis']
        temp = state.get('max_temperature', 0)
        
        # Show current temperature status
        if temp > 60:
            st.error(f"🔥 CRITICAL: {temp:.1f}°C - Thermal runaway detected!")
        elif temp > 50:
            st.warning(f"⚠️ WARNING: {temp:.1f}°C - Approaching thermal limits")
        elif temp > 40:
            st.warning(f"ℹ️ ELEVATED: {temp:.1f}°C - Monitor closely")
        else:
            st.success(f"✅ NORMAL: {temp:.1f}°C - Operating within safe range")
        
        # Show trend analysis if available
        if 'trend' in analysis:
            trend = analysis['trend']
            if trend > 0.2:
                st.error(f"📈 Rapid temperature increase: +{trend:.2f}°C/s")
            elif trend > 0.1:
                st.warning(f"📈 Temperature rising: +{trend:.2f}°C/s")
            elif trend < -0.1:
                st.success(f"📉 Temperature decreasing: {trend:.2f}°C/s")
    else:
        st.info("Thermal analysis data will appear here")

# Main simulation loop
while True:
    if st.session_state.running:
        # Update simulation
        state = update_simulation(0.1)  # Small time step for smooth animation
        
        if state:
            # Update heatmap with unique key
            temp_grid = np.array(state['temperature_grid'])  # Ensure it's a numpy array
            fig = create_heatmap_figure(temp_grid, state['max_temperature'])
            heatmap_placeholder.plotly_chart(fig, use_container_width=True, key=f"heatmap_{st.session_state.time_elapsed:.1f}")
            
            # Update metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.markdown(f"<div class='metric-card'>"
                             f"<h3>🌡️ Max Temp</h3>"
                             f"<h2>{state['max_temperature']:.1f}°C</h2>"
                             f"</div>", unsafe_allow_html=True)
            with metric_cols[1]:
                max_current = max(abs(cell['current']) for cell in state['cells'])
                st.markdown(f"<div class='metric-card'>"
                             f"<h3>⚡ Max Current</h3>"
                             f"<h2>{max_current:.2f}A</h2>"
                             f"</div>", unsafe_allow_html=True)
            with metric_cols[2]:
                min_voltage = min(cell['voltage'] for cell in state['cells'])
                st.markdown(f"<div class='metric-card'>"
                             f"<h3>🔋 Min Voltage</h3>"
                             f"<h2>{min_voltage:.2f}V</h2>"
                             f"</div>", unsafe_allow_html=True)
            with metric_cols[3]:
                st.markdown(f"<div class='metric-card'>"
                             f"<h3>⏱️ Time</h3>"
                             f"<h2>{st.session_state.time_elapsed:.1f}s</h2>"
                             f"</div>", unsafe_allow_html=True)
            
            # Update temperature history chart
            if len(st.session_state.history['time']) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.history['time'],
                    y=st.session_state.history['max_temp'],
                    name='Max Temp',
                    line=dict(color='red', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state.history['time'],
                    y=st.session_state.history['avg_temp'],
                    name='Avg Temp',
                    line=dict(color='orange', width=2)
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.2)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=dict(color='white')
                )
                temp_chart.plotly_chart(fig, use_container_width=True, key=f"temp_chart_{st.session_state.time_elapsed:.1f}")
            
            # Update prediction messages
            if 'prediction_messages' in state and state['prediction_messages']:
                with st.sidebar.expander("🔮 AI Predictions", expanded=True):
                    # Show only the 3 most recent messages
                    for msg in state['prediction_messages'][-3:]:
                        # Determine message class based on severity
                        msg_class = 'info'
                        if 'severity' in msg:
                            if msg['severity'] == 'high':
                                msg_class = 'warning'
                            elif msg['severity'] == 'critical':
                                msg_class = 'danger'
                        
                        # Format timestamp
                        time_str = time.strftime('%H:%M:%S', time.localtime(msg.get('timestamp', time.time())))
                        
                        # Display message with timestamp and styling
                        st.markdown(
                            f"<div class='prediction-message {msg_class}'>"
                            f"<small style='opacity:0.7;'>[{time_str}]</small><br>"
                            f"{msg['message']}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
    
    # Add a small delay to prevent high CPU usage
    time.sleep(st.session_state.sim_speed / 1000)
    
    # Break the loop if the page is being refreshed
    if not st.session_state.running and st.session_state.time_elapsed > 0:
        break
