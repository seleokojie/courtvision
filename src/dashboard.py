import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import time
import warnings

# Suppress pandas SQLAlchemy warning (we're using psycopg2 directly which works fine)
warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')

# Page config
st.set_page_config(
    page_title="CourtVision Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Database connection
DB_HOST = os.environ.get('DB_HOST', 'localhost')

@st.cache_resource
def get_db_connection():
    """Create database connection with retry logic."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                dbname="courtvision",
                user="admin",
                password="password",
                host=DB_HOST
            )
            return conn
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.error(f"Failed to connect to database: {e}")
                return None

def fetch_shot_data(limit=1000):
    """Fetch shot telemetry data from the database."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    query = """
        SELECT 
            id,
            game_id,
            player_name,
            shot_distance,
            expected_points,
            shot_grade,
            created_at
        FROM shot_telemetry 
        ORDER BY created_at DESC
        LIMIT %s
    """
    try:
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def fetch_aggregate_stats():
    """Fetch aggregate statistics from the database."""
    conn = get_db_connection()
    if conn is None:
        return {}
    
    try:
        cur = conn.cursor()
        
        # Total shots
        cur.execute("SELECT COUNT(*) FROM shot_telemetry")
        total_shots = cur.fetchone()[0]
        
        # Average expected points
        cur.execute("SELECT AVG(expected_points) FROM shot_telemetry")
        avg_xp = cur.fetchone()[0] or 0
        
        # Grade distribution
        cur.execute("""
            SELECT shot_grade, COUNT(*) 
            FROM shot_telemetry 
            GROUP BY shot_grade 
            ORDER BY shot_grade
        """)
        grade_dist = dict(cur.fetchall())
        
        # Shots per player (top 10)
        cur.execute("""
            SELECT player_name, COUNT(*) as shots, AVG(expected_points) as avg_xp
            FROM shot_telemetry 
            GROUP BY player_name 
            ORDER BY shots DESC 
            LIMIT 10
        """)
        player_stats = cur.fetchall()
        
        # Distance distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN shot_distance <= 3 THEN 'Rim (0-3ft)'
                    WHEN shot_distance <= 10 THEN 'Paint (4-10ft)'
                    WHEN shot_distance <= 16 THEN 'Mid-Range (11-16ft)'
                    WHEN shot_distance <= 23 THEN 'Long 2 (17-23ft)'
                    ELSE '3-Pointer (24+ft)'
                END as zone,
                COUNT(*) as count,
                AVG(expected_points) as avg_xp
            FROM shot_telemetry
            GROUP BY zone
            ORDER BY MIN(shot_distance)
        """)
        zone_stats = cur.fetchall()
        
        # Recent activity (last 5 minutes)
        cur.execute("""
            SELECT COUNT(*) 
            FROM shot_telemetry 
            WHERE created_at >= NOW() - INTERVAL '5 minutes'
        """)
        recent_shots = cur.fetchone()[0]
        
        # Shots over time (last hour, grouped by minute)
        cur.execute("""
            SELECT 
                DATE_TRUNC('minute', created_at) as minute,
                COUNT(*) as shots
            FROM shot_telemetry
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            GROUP BY minute
            ORDER BY minute
        """)
        time_series = cur.fetchall()
        
        return {
            'total_shots': total_shots,
            'avg_xp': avg_xp,
            'grade_dist': grade_dist,
            'player_stats': player_stats,
            'zone_stats': zone_stats,
            'recent_shots': recent_shots,
            'time_series': time_series
        }
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return {}

def create_shot_zone_chart(zone_stats):
    """Create a horizontal bar chart for shot zones."""
    if not zone_stats:
        return None
    
    zones = [z[0] for z in zone_stats]
    counts = [z[1] for z in zone_stats]
    avg_xps = [float(z[2]) for z in zone_stats]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Shots by Zone", "Avg Expected Points"),
        horizontal_spacing=0.2
    )
    
    # Shot counts
    fig.add_trace(
        go.Bar(
            y=zones,
            x=counts,
            orientation='h',
            name='Shot Count',
            marker_color='#667eea',
            text=counts,
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Average XP
    fig.add_trace(
        go.Bar(
            y=zones,
            x=avg_xps,
            orientation='h',
            name='Avg XP',
            marker_color='#764ba2',
            text=[f'{xp:.2f}' for xp in avg_xps],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=150, r=80, t=50, b=30)
    )
    
    # Update x-axes for grid lines
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_grade_gauge(grade_dist, total_shots):
    """Create a donut chart for shot grade distribution."""
    grades = {'A': 0, 'B': 0, 'C': 0}
    grades.update({k.strip(): v for k, v in grade_dist.items()})
    
    labels = ['Grade A (Premium)', 'Grade B (Good)', 'Grade C (Poor)']
    values = [grades.get('A', 0), grades.get('B', 0), grades.get('C', 0)]
    colors = ['#00d26a', '#ffc107', '#ff6b6b']
    
    # Calculate percentages for display
    percentages = [(v / total_shots * 100) if total_shots > 0 else 0 for v in values]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='percent',
        textfont=dict(size=16, color='white'),
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=14)
        ),
        annotations=[dict(
            text=f'{total_shots}<br>shots',
            x=0.5, y=0.5,
            font=dict(size=20, color='white'),
            showarrow=False
        )]
    )
    
    return fig

def create_player_chart(player_stats):
    """Create a horizontal bar chart for top players."""
    if not player_stats:
        return None
    
    # Reverse for horizontal bar chart (top player at top)
    players = [p[0] for p in player_stats][::-1]
    shots = [p[1] for p in player_stats][::-1]
    avg_xps = [float(p[2]) for p in player_stats][::-1]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Shot Volume', 'Avg Expected Points'),
        horizontal_spacing=0.15
    )
    
    # Shot counts
    fig.add_trace(
        go.Bar(
            y=players,
            x=shots,
            orientation='h',
            marker_color='#667eea',
            text=shots,
            textposition='outside',
            name='Shots'
        ),
        row=1, col=1
    )
    
    # Avg XP
    fig.add_trace(
        go.Bar(
            y=players,
            x=avg_xps,
            orientation='h',
            marker_color='#764ba2',
            text=[f'{xp:.2f}' for xp in avg_xps],
            textposition='outside',
            name='Avg XP'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=450,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=150, r=50, t=50, b=30)
    )
    
    # Update x-axes
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_time_series_chart(time_series):
    """Create a time series chart of shot activity."""
    if not time_series:
        return None
    
    times = [t[0] for t in time_series]
    counts = [t[1] for t in time_series]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=counts,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#667eea', width=2),
        marker=dict(size=6),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        title='Shot Activity (Last Hour)',
        xaxis_title='Time',
        yaxis_title='Shots per Minute',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_distance_histogram(df):
    """Create a histogram of shot distances."""
    if df.empty:
        return None
    
    fig = px.histogram(
        df,
        x='shot_distance',
        nbins=25,
        title='Shot Distance Distribution',
        labels={'shot_distance': 'Distance (feet)', 'count': 'Number of Shots'},
        color_discrete_sequence=['#667eea']
    )
    
    # Add 3-point line marker
    fig.add_vline(x=23, line_dash="dash", line_color="#ff6b6b", 
                  annotation_text="3PT Line", annotation_position="top")
    
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title='Count'),
        margin=dict(l=50, r=30, t=50, b=50)
    )
    
    return fig

def create_xp_by_distance_scatter(df):
    """Create scatter plot of expected points by distance."""
    if df.empty:
        return None
    
    fig = px.scatter(
        df,
        x='shot_distance',
        y='expected_points',
        color='shot_grade',
        title='Expected Points by Shot Distance',
        labels={
            'shot_distance': 'Distance (feet)',
            'expected_points': 'Expected Points',
            'shot_grade': 'Grade'
        },
        color_discrete_map={'A': '#00d26a', 'B': '#ffc107', 'C': '#ff6b6b', ' A': '#00d26a', ' B': '#ffc107', ' C': '#ff6b6b'}
    )
    
    # Add 3-point line marker
    fig.add_vline(x=23, line_dash="dash", line_color="#ff6b6b", 
                  annotation_text="3PT Line", annotation_position="top")
    
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=30, t=70, b=50)
    )
    
    return fig
    
    return fig

def create_basketball_court():
    """Create a simple basketball court visualization."""
    fig = go.Figure()
    
    # Court outline (half court)
    court_shapes = [
        # Three point line (arc)
        dict(
            type="path",
            path="M -220 -47.5 A 237.5 237.5 0 0 1 220 -47.5",
            line_color="white",
            fillcolor="rgba(0,0,0,0)"
        ),
        # Three point corners
        dict(type="line", x0=-220, y0=-47.5, x1=-220, y1=-250, line_color="white"),
        dict(type="line", x0=220, y0=-47.5, x1=220, y1=-250, line_color="white"),
        # Paint
        dict(type="rect", x0=-80, y0=-250, x1=80, y1=-190+60, line_color="white", fillcolor="rgba(102,126,234,0.1)"),
        # Free throw circle
        dict(type="circle", x0=-60, y0=-190+60-60, x1=60, y1=-190+60+60, line_color="white"),
        # Restricted area
        dict(type="circle", x0=-40, y0=-250, x1=40, y1=-170, line_color="white"),
        # Rim
        dict(type="circle", x0=-7.5, y0=-247.5, x1=7.5, y1=-232.5, line_color="orange", line_width=3),
        # Backboard
        dict(type="line", x0=-30, y0=-250, x1=30, y1=-250, line_color="white", line_width=3),
    ]
    
    for shape in court_shapes:
        fig.add_shape(**shape)
    
    fig.update_layout(
        xaxis=dict(range=[-250, 250], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-260, 100], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#2d4a3e',  # Court green
        title='Shot Chart (Simulated Distribution)'
    )
    
    return fig

def simulate_shot_positions(df, n_points=200):
    """Simulate shot positions based on distance (since we don't have coordinates)."""
    if df.empty:
        return None
    
    sample = df.sample(min(n_points, len(df)))
    
    # Generate random angles and use actual distances
    angles = np.random.uniform(0, np.pi, len(sample))
    distances = sample['shot_distance'].values * 10  # Scale for visualization
    
    # Convert polar to cartesian
    x = distances * np.cos(angles)
    y = -250 + distances * np.sin(angles)
    
    fig = create_basketball_court()
    
    # Color by grade
    colors = sample['shot_grade'].map({'A': '#00d26a', 'B': '#ffc107', 'C': '#ff6b6b'}).fillna('#666')
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=[f"{row['player_name']}<br>{row['shot_distance']}ft<br>XP: {row['expected_points']:.2f}" 
              for _, row in sample.iterrows()],
        hoverinfo='text'
    ))
    
    return fig

# Main Dashboard
def main():
    # Header
    st.markdown("# 🏀 CourtVision Live Dashboard")
    st.markdown("Real-time NBA shot analytics powered by Kafka streaming")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=True)
    data_limit = st.sidebar.slider("Data points to load", 100, 5000, 1000, 100)
    
    if auto_refresh:
        st.sidebar.info("Dashboard refreshes every 10 seconds")
    
    # Fetch data
    with st.spinner("Loading data..."):
        stats = fetch_aggregate_stats()
        df = fetch_shot_data(limit=data_limit)
    
    if not stats or stats.get('total_shots', 0) == 0:
        st.warning("⚠️ No shot data available yet. Make sure the producer and consumer are running.")
        st.info("""
        **Quick Start:**
        1. Start the services: `docker compose up -d`
        2. Run the producer: `python src/producer.py`
        3. Wait for data to flow through the pipeline
        """)
        return
    
    # Top metrics row
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Shots",
            value=f"{stats['total_shots']:,}",
            delta=f"+{stats['recent_shots']} (5min)"
        )
    
    with col2:
        st.metric(
            label="Avg Expected Points",
            value=f"{stats['avg_xp']:.3f}",
            delta="per shot"
        )
    
    with col3:
        grade_a_pct = stats['grade_dist'].get('A', 0) / stats['total_shots'] * 100 if stats['total_shots'] > 0 else 0
        st.metric(
            label="Premium Shots (A)",
            value=f"{grade_a_pct:.1f}%",
            delta="of total"
        )
    
    with col4:
        three_pt_shots = sum(1 for _, row in df.iterrows() if row['shot_distance'] > 23) if not df.empty else 0
        three_pt_pct = three_pt_shots / len(df) * 100 if len(df) > 0 else 0
        st.metric(
            label="3-Point Attempts",
            value=f"{three_pt_pct:.1f}%",
            delta="of sample"
        )
    
    # Shot Grade Distribution
    st.markdown("### 🎯 Model Performance: Shot Grade Distribution")
    grade_fig = create_grade_gauge(stats['grade_dist'], stats['total_shots'])
    if grade_fig:
        st.plotly_chart(grade_fig, width="stretch")
    
    st.markdown("""
    <small>
    <b>Grade Legend:</b> 
    <span style='color:#00d26a'>●</span> A = Premium (XP > 1.1) | 
    <span style='color:#ffc107'>●</span> B = Good (XP 0.9-1.1) | 
    <span style='color:#ff6b6b'>●</span> C = Poor (XP < 0.9)
    </small>
    """, unsafe_allow_html=True)
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Shot Zone Analysis
        st.markdown("### 🗺️ Shot Zone Analysis")
        zone_fig = create_shot_zone_chart(stats['zone_stats'])
        if zone_fig:
            st.plotly_chart(zone_fig, width="stretch")
    
    with col2:
        # Shot Court Visualization
        st.markdown("### 🏟️ Shot Distribution (Simulated)")
        court_fig = simulate_shot_positions(df)
        if court_fig:
            st.plotly_chart(court_fig, width="stretch")
    
    # Distance Analysis
    st.markdown("### 📏 Distance Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        hist_fig = create_distance_histogram(df)
        if hist_fig:
            st.plotly_chart(hist_fig, width="stretch")
    
    with col2:
        scatter_fig = create_xp_by_distance_scatter(df)
        if scatter_fig:
            st.plotly_chart(scatter_fig, width="stretch")
    
    # Player Stats
    st.markdown("### 👤 Top Players")
    player_fig = create_player_chart(stats['player_stats'])
    if player_fig:
        st.plotly_chart(player_fig, width="stretch")
    
    # Activity Timeline
    st.markdown("### 📈 Live Activity")
    time_fig = create_time_series_chart(stats['time_series'])
    if time_fig:
        st.plotly_chart(time_fig, width="stretch")
    else:
        st.info("No activity in the last hour. Start the producer to see live data flow.")
    
    # Recent shots table
    st.markdown("### 📋 Recent Shots")
    if not df.empty:
        display_df = df.head(20)[['player_name', 'shot_distance', 'expected_points', 'shot_grade', 'created_at']].copy()
        display_df.columns = ['Player', 'Distance (ft)', 'Expected Points', 'Grade', 'Time']
        display_df['Expected Points'] = display_df['Expected Points'].round(3)
        st.dataframe(display_df, width="stretch", hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <small>
    <b>CourtVision</b> | Real-time NBA Shot Analytics<br>
    Data flows: Producer → Kafka → Consumer → PostgreSQL → Dashboard
    </small>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
