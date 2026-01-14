import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="CFD Assisted Corrosion and RBI Tool",
    layout="wide"
)
st.title("CFD Assisted Corrosion and RBI Tool – Level 1 (API 570)")

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("Pipe & Flow Inputs (SI Units)")

pipe_length = st.sidebar.number_input("Pipe Length (m)", 1.0, 100.0, 10.0)
pipe_ID = st.sidebar.number_input("Pipe Inner Diameter (m)", 0.05, 2.0, 0.30)
velocity = st.sidebar.number_input("Flow Velocity (m/s)", 0.1, 25.0, 2.5)
pressure = st.sidebar.number_input("Operating Pressure (MPa)", 0.1, 30.0, 5.0)

fitting = st.sidebar.selectbox(
    "Pipe Geometry",
    ["Straight", "Elbow", "Tee", "Reducer"]
)

fluid = st.sidebar.selectbox(
    "Flowing Fluid",
    ["Water", "Glycol", "Steam", "Acetate", "Gas Condensate"]
)

nominal_t = st.sidebar.number_input("Nominal Thickness (mm)", 1.0, 25.0, 6.02)
min_allow_t = st.sidebar.number_input("Minimum Allowable Thickness (mm)", 1.0, 10.0, 3.0)

st.sidebar.header("UT Thickness History")
years_input = st.sidebar.text_input("Inspection Years (comma separated)", "2019,2021,2023,2025")
thk_input = st.sidebar.text_input("Measured Thickness (mm)", "7.0,6.52,5.73,4.02")

# EXECUTE BUTTON
run_analysis = st.sidebar.button(" Execute Analysis")

# ================================
# 3D PIPE BUILDERS
# ================================
def build_straight_pipe(x, r):
    theta = np.linspace(0, 2*np.pi, 60)
    Xc, Theta = np.meshgrid(x, theta)
    Yc = r * np.cos(Theta)
    Zc = r * np.sin(Theta)
    return Xc, Yc, Zc

def build_elbow_pipe(r, angle_deg=90, n=120):
    theta = np.linspace(0, 2*np.pi, 60)
    bend_angle = np.linspace(0, np.deg2rad(angle_deg), n)
    Theta, Phi = np.meshgrid(theta, bend_angle)
    Rc = 3 * r
    Xc = Rc * np.cos(Phi)
    Yc = Rc * np.sin(Phi) + r * np.cos(Theta)
    Zc = r * np.sin(Theta)
    return Xc, Yc, Zc

def build_reducer_pipe(x, r1, r2):
    theta = np.linspace(0, 2*np.pi, 60)
    radius = np.linspace(r1, r2, len(x))
    Xc, Theta = np.meshgrid(x, theta)
    Rc = np.tile(radius, (len(theta), 1))
    Yc = Rc * np.cos(Theta)
    Zc = Rc * np.sin(Theta)
    return Xc, Yc, Zc

def build_tee_pipe(x, r):
    theta = np.linspace(0, 2*np.pi, 60)
    Xc, Theta = np.meshgrid(x, theta)
    Yc = r * np.cos(Theta)
    Zc = r * np.sin(Theta)
    return Xc, Yc, Zc

# ================================
# ANALYSIS BLOCK
# ================================
if run_analysis:

    # ------------------------
    # PROCESS INPUT DATA
    # ------------------------
    try:
        years = np.array([int(y) for y in years_input.split(",")])
        thicknesses = np.array([float(t) for t in thk_input.split(",")])
    except:
        st.error("Invalid UT input format.")
        st.stop()

    if len(years) != len(thicknesses):
        st.error("Years and thickness arrays must have same length.")
        st.stop()

    if thicknesses[-1] <= min_allow_t:
        st.error("Pipe already below minimum allowable thickness.")
        st.stop()

    # ------------------------
    # CORROSION RATE (API 570 Level-1)
    # ------------------------
    time_span = years[-1] - years[0]
    corrosion_rate = (thicknesses[0] - thicknesses[-1]) / time_span  # mm/year

    if corrosion_rate <= 0:
        st.error("Non-physical corrosion rate detected.")
        st.stop()

    remaining_life = (thicknesses[-1] - min_allow_t) / corrosion_rate
    next_inspection = min(remaining_life * 0.5, 5.0)  # conservative

    # ------------------------
    # REDUCED-ORDER CFD MODEL
    # ------------------------
    nx, ny = 140, 70
    x = np.linspace(0, pipe_length, nx)
    y = np.linspace(-pipe_ID / 2, pipe_ID / 2, ny)
    X, Y = np.meshgrid(x, y)

    # Axial velocity profile (parabolic)
    velocity_field = velocity * (1 - (Y / (pipe_ID / 2)) ** 2)

    # Geometry factor for wall shear stress
    geo_factor = {
        "Straight": 1.0,
        "Elbow": 1.8,
        "Tee": 2.3,
        "Reducer": 1.6
    }[fitting]

    rho = 1000.0  # kg/m3
    tau_field = rho * velocity**2 * geo_factor * np.exp(
        -((X - pipe_length / 2) ** 2) / (0.1 * pipe_length)**2
    )

    # ------------------------
    # HOTSPOT SCORE
    # ------------------------
    tau_norm = tau_field / np.max(tau_field)
    cr_norm = corrosion_rate / corrosion_rate
    risk_score = tau_norm * cr_norm

    # ------------------------
    # MONTE CARLO RBI
    # ------------------------
    n_sim = 5000
    cr_dist = np.random.normal(corrosion_rate, 0.15 * corrosion_rate, n_sim)
    cr_dist = cr_dist[cr_dist > 0]
    life_dist = (thicknesses[-1] - min_allow_t) / cr_dist
    failure_prob_5yr = np.mean(life_dist < 5)

    # ------------------------
    # PLOTS
    # ------------------------
    col1, col2 = st.columns(2)

    # Thickness Degradation
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

    # Wall Shear Stress Contour
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
        ax.plot(x, tau_field[ny // 2, :])
        ax.set_xlabel("Pipe Length (m)")
        ax.set_ylabel("τ (Pa)")
        st.pyplot(fig)

    # ------------------------
    # 3D HOTSPOT VISUALIZATION
    # ------------------------
    st.subheader("3D Pipe Corrosion Hotspots")
    pipe_radius = pipe_ID / 2

    if fitting == "Straight":
        X3d, Y3d, Z3d = build_straight_pipe(x, pipe_radius)
        risk_3d = np.tile(risk_score.mean(axis=0), (60, 1))
    elif fitting == "Elbow":
        X3d, Y3d, Z3d = build_elbow_pipe(pipe_radius)
        risk_3d = np.tile(np.linspace(0, 1, X3d.shape[1]), (60, 1))
    elif fitting == "Reducer":
        X3d, Y3d, Z3d = build_reducer_pipe(x, pipe_radius, pipe_radius * 0.6)
        risk_3d = np.tile(risk_score.mean(axis=0), (60, 1))
    elif fitting == "Tee":
        X3d, Y3d, Z3d = build_tee_pipe(x, pipe_radius)
        risk_3d = np.tile(risk_score.mean(axis=0), (60, 1))

    fig3d = go.Figure(data=[
        go.Surface(
            x=X3d, y=Y3d, z=Z3d,
            surfacecolor=risk_3d,
            colorscale="YlOrRd",
            cmin=0, cmax=1,
            colorbar=dict(title="Risk Index")
        )
    ])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data"
        ),
        height=550
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # Monte Carlo Histogram
    st.subheader("Monte Carlo RBI – Remaining Life Distribution")
    fig, ax = plt.subplots()
    ax.hist(life_dist, bins=50, density=True)
    ax.axvline(5, color="r", linestyle="--", label="5-year threshold")
    ax.set_xlabel("Remaining Life (years)")
    ax.legend()
    st.pyplot(fig)

    # ================================
    # RESULTS SUMMARY
    # ================================
    st.success("Analysis Completed Successfully")
    st.write("### Key Results")
    st.write(f"**Corrosion Rate:** {corrosion_rate:.3f} mm/year")
    st.write(f"**Remaining Useful Life:** {remaining_life:.2f} years")
    st.write(f"**Recommended Next Inspection (API 570):** {next_inspection:.2f} years")
    st.write(f"**Failure Probability within 5 years:** {failure_prob_5yr*100:.1f}%")

else:
    st.info("Enter inputs and click **Execute Analysis** to run the assessment.")

