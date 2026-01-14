import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="CFD-Assisted Corrosion and RBI Tool", layout="wide")
st.title("CFD-Assisted Corrosion and RBI Tool – Level 1")

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("Pipe & Flow Inputs (SI Units)")

pipe_length = st.sidebar.number_input("Pipe Length (m)", 1.0, 50.0, 10.0)
pipe_ID = st.sidebar.number_input("Pipe Inner Diameter (m)", 0.05, 1.5, 0.3)
velocity = st.sidebar.number_input("Flow Velocity (m/s)", 0.1, 20.0, 2.5)
pressure = st.sidebar.number_input("Operating Pressure (MPa)", 0.1, 25.0, 5.0)

fitting = st.sidebar.selectbox(
    "Pipe Geometry",
    ["Straight", "Elbow", "Tee", "Reducer"]
)

fluid = st.sidebar.selectbox(
    "Flowing Fluid",
    ["Water", "Glycol", "Steam", "Acetate", "Gas Condensate"]
)

nominal_t = st.sidebar.number_input("Nominal Thickness (mm)", 1.0, 20.0, 6.02)
min_allow_t = st.sidebar.number_input("Minimum Allowable Thickness (mm)", 1.0, 10.0, 3.0)

st.sidebar.header("UT Thickness History")
years = st.sidebar.text_input("Years (comma separated)", "2019,2021,2023,2025")
thicknesses = st.sidebar.text_input("Thickness mm (comma separated)", "7.0,6.52,5.73,4.02")

# ================================
# PROCESS INPUT DATA
# ================================
years = np.array([int(y) for y in years.split(",")])
thicknesses = np.array([float(t) for t in thicknesses.split(",")])

# ================================
# CORROSION RATE & API 570
# ================================
time_span = years[-1] - years[0]
corrosion_rate = (thicknesses[0] - thicknesses[-1]) / time_span  # mm/year
remaining_life = (thicknesses[-1] - min_allow_t) / corrosion_rate
next_inspection = min(remaining_life * 0.5, 5)  # Level-1 simplified

# ================================
# CFD – REDUCED ORDER MODEL
# ================================
nx, ny = 120, 60
x = np.linspace(0, pipe_length, nx)
y = np.linspace(-pipe_ID / 2, pipe_ID / 2, ny)
X, Y = np.meshgrid(x, y)

# Velocity profile (parabolic)
velocity_field = velocity * (1 - (Y / (pipe_ID / 2)) ** 2)

# Geometry factor
geo_factor = {
    "Straight": 1.0,
    "Elbow": 1.8,
    "Tee": 2.3,
    "Reducer": 1.6
}[fitting]

rho = 1000
tau_field = rho * velocity**2 * geo_factor * np.exp(-((X - pipe_length / 2) ** 2))

# ================================
# HOTSPOT RISK SCORE
# ================================
tau_norm = tau_field / np.max(tau_field)
cr_norm = corrosion_rate / max(corrosion_rate, 1e-3)
risk_score = tau_norm * cr_norm
hotspot_threshold = 0.75
hotspots = risk_score > hotspot_threshold

# ================================
# MONTE CARLO RBI
# ================================
n_sim = 5000
cr_dist = np.random.normal(corrosion_rate, 0.15 * corrosion_rate, n_sim)
life_dist = (thicknesses[-1] - min_allow_t) / cr_dist
failure_prob = np.mean(life_dist < 5)

# ================================
# LAYOUT PLOTS
# ================================
col1, col2 = st.columns(2)

# Thickness Degradation Curve
with col1:
    st.subheader("Thickness Degradation Curve")
    future_years = np.arange(years[-1], years[-1] + 15)
    future_thk = thicknesses[-1] - corrosion_rate * (future_years - years[-1])

    fig, ax = plt.subplots()
    ax.plot(years, thicknesses, "o-", label="Measured")
    ax.plot(future_years, future_thk, "--", label="Predicted")
    ax.axhline(min_allow_t, color="r", linestyle="--", label="Min Allowable")
    ax.set_xlabel("Year")
    ax.set_ylabel("Thickness (mm)")
    ax.legend()
    st.pyplot(fig)

# Velocity Contour
with col2:
    st.subheader("Velocity Contour (m/s)")
    fig, ax = plt.subplots()
    c = ax.contourf(X, Y, velocity_field, 50)
    plt.colorbar(c)
    st.pyplot(fig)

# Wall Shear Stress Contour and Centerline
col3, col4 = st.columns(2)

with col3:
    st.subheader("Wall Shear Stress τ (Pa)")
    fig, ax = plt.subplots()
    c = ax.contourf(X, Y, tau_field, 50)
    plt.colorbar(c)
    st.pyplot(fig)

with col4:
    st.subheader("τ Along Pipe Centerline")
    fig, ax = plt.subplots()
    ax.plot(x, tau_field[ny//2, :])
    ax.set_xlabel("Pipe Length (m)")
    ax.set_ylabel("τ (Pa)")
    st.pyplot(fig)

# ================================
# 3D Hotspot Visualization
# ================================
st.subheader("3D Pipe Hotspot Visualization")
theta = np.linspace(0, 2*np.pi, ny)
X3d, Theta = np.meshgrid(x, theta)
Y3d = (pipe_ID/2) * np.cos(Theta)
Z3d = (pipe_ID/2) * np.sin(Theta)

risk_3d = np.tile(risk_score.mean(axis=0), (ny,1))  # average along radius for display

fig3d = go.Figure(data=[go.Surface(
    x=X3d, y=Y3d, z=Z3d,
    surfacecolor=risk_3d,
    colorscale='YlOrRd',
    cmin=0,
    cmax=1,
    colorbar=dict(title="Risk Score")
)])
fig3d.update_layout(
    scene=dict(
        xaxis_title='Pipe Length (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)'
    ),
    width=900,
    height=500
)
st.plotly_chart(fig3d, use_container_width=True)
st.write("Red zones indicate high-risk hotspots along the pipe.")

# Monte Carlo RBI Histogram
st.subheader("Monte Carlo RBI – Remaining Life Distribution")
fig, ax = plt.subplots()
ax.hist(life_dist, bins=50, density=True)
ax.axvline(5, color="r", linestyle="--", label="5 Year Threshold")
ax.set_xlabel("Remaining Life (years)")
ax.legend()
st.pyplot(fig)

# ================================
# RESULTS SUMMARY
# ================================
st.success("Analysis Completed")
st.write("### Key Results")
st.write(f"**Corrosion Rate:** {corrosion_rate:.3f} mm/year")
st.write(f"**Remaining Useful Life:** {remaining_life:.1f} years")
st.write(f"**Recommended Next Inspection (API 570):** {next_inspection:.1f} years")
st.write(f"**Probability of Failure within 5 years:** {failure_prob*100:.1f}%")
