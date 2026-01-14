import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, interpolate
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CFD-Assisted Corrosion and RBI Tool", layout="wide")
st.title("CFD-Assisted Corrosion and RBI Tool – Level 1 (3D Mesh Geometry)")

# ------------------- PHYSICAL CONSTANTS -------------------
FLUID_PROPERTIES = {
    "Water": {"density": 998.2, "viscosity": 0.001002, "corr_factor": 1.0},
    "Glycol": {"density": 1116.0, "viscosity": 0.0161, "corr_factor": 1.2},
    "Steam": {"density": 0.6, "viscosity": 1.34e-5, "corr_factor": 2.5},
    "Acetate": {"density": 1048.0, "viscosity": 0.00078, "corr_factor": 3.0},
    "Gas Condensate": {"density": 780.0, "viscosity": 0.0004, "corr_factor": 1.8}
}

MATERIALS = {
    "Carbon Steel": {"yield_strength": 240, "tensile_strength": 415},
    "Stainless 316": {"yield_strength": 290, "tensile_strength": 580},
    "Duplex Steel": {"yield_strength": 550, "tensile_strength": 750}
}

# ------------------- SIDEBAR INPUTS -------------------
st.sidebar.header("🔧 Geometry & Flow Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    pipe_length = st.number_input("Pipe Length (m)", 1.0, 100.0, 10.0, step=0.5)
    pipe_ID = st.number_input("Pipe Inner Diameter (m)", 0.05, 2.0, 0.3, step=0.01)
with col2:
    velocity = st.number_input("Flow Velocity (m/s)", 0.1, 50.0, 2.5, step=0.1)
    pressure = st.number_input("Operating Pressure (MPa)", 0.1, 30.0, 5.0, step=0.1)

fitting = st.sidebar.selectbox("Pipe Geometry", ["Straight", "Elbow", "Tee", "Reducer"])
material = st.sidebar.selectbox("Pipe Material", list(MATERIALS.keys()))
fluid = st.sidebar.selectbox("Flowing Fluid", list(FLUID_PROPERTIES.keys()))

if fitting == "Reducer":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        reducer_d1 = st.number_input("Large End Diameter (m)", 0.05, 2.0, 0.3, step=0.01)
    with col2:
        reducer_d2 = st.number_input("Small End Diameter (m)", 0.05, 2.0, 0.15, step=0.01)
    reducer_length = st.sidebar.slider("Reducer Length (m)", 0.5, 5.0, 1.0, step=0.1)
else:
    reducer_d1 = reducer_d2 = pipe_ID
    reducer_length = 1.0

st.sidebar.header("📏 Wall Thickness Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    nominal_t = st.number_input("Nominal Thickness (mm)", 1.0, 50.0, 8.0, step=0.5)
with col2:
    min_allow_t = st.number_input("Min Allowable Thickness (mm)", 1.0, 10.0, 3.5, step=0.1)

st.sidebar.header(" UT Thickness History")
years_input = st.sidebar.text_input("Inspection Years (comma separated)", "2019,2021,2023,2025")
thk_input = st.sidebar.text_input("Measured Thickness (mm)", "8.0,7.2,6.3,5.1")

# ------------------- IMPROVED GEOMETRY FUNCTIONS -------------------
def create_cylinder_mesh(length, radius, resolution=20):
    """Create proper cylindrical mesh with end caps"""
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, length, resolution)
    
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = z_grid
    
    # Add end caps
    r_end = np.linspace(0, radius, resolution//2)
    theta_end = np.linspace(0, 2*np.pi, resolution)
    r_grid, theta_grid = np.meshgrid(r_end, theta_end)
    
    x_start = r_grid * np.cos(theta_grid)
    y_start = r_grid * np.sin(theta_grid)
    z_start = np.zeros_like(x_start)
    
    x_end = r_grid * np.cos(theta_grid)
    y_end = r_grid * np.sin(theta_grid)
    z_end = np.ones_like(x_end) * length
    
    # Combine
    x_all = np.concatenate([x.flatten(), x_start.flatten(), x_end.flatten()])
    y_all = np.concatenate([y.flatten(), y_start.flatten(), y_end.flatten()])
    z_all = np.concatenate([z.flatten(), z_start.flatten(), z_end.flatten()])
    
    return x_all, y_all, z_all

def create_elbow_mesh(radius, bend_radius=3, angle_deg=90, resolution=20):
    """Create 90-degree elbow with proper curvature"""
    theta = np.linspace(0, 2*np.pi, resolution)  # circumferential
    phi = np.linspace(0, np.pi/2, resolution)    # bend angle
    
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Torus coordinates for elbow
    R = bend_radius * radius  # Bend radius
    x = (R + radius * np.cos(theta_grid)) * np.cos(phi_grid)
    y = (R + radius * np.cos(theta_grid)) * np.sin(phi_grid)
    z = radius * np.sin(theta_grid)
    
    return x.flatten(), y.flatten(), z.flatten()

def create_reducer_mesh(length, r_start, r_end, resolution=20):
    """Create conical reducer mesh"""
    z = np.linspace(0, length, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    
    z_grid, theta_grid = np.meshgrid(z, theta)
    
    # Linear taper
    radius = r_start + (r_end - r_start) * (z_grid / length)
    
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = z_grid
    
    # Add end caps
    z_flat = z.flatten()
    radius_flat = radius.flatten()
    
    # Create hotspot intensity (higher at smaller end)
    intensity = 1.0 + 0.5 * (1 - z_flat/length)  # Higher at smaller end
    
    return x.flatten(), y.flatten(), z.flatten(), intensity

def create_tee_mesh(main_length, main_radius, branch_length=None, resolution=20):
    """Create tee junction mesh"""
    if branch_length is None:
        branch_length = main_radius * 3
    
    # Main pipe
    x_main, y_main, z_main = create_cylinder_mesh(main_length, main_radius, resolution)
    
    # Branch pipe (rotated)
    theta = np.linspace(0, 2*np.pi, resolution)
    z_br = np.linspace(-branch_length/2, branch_length/2, resolution)
    
    z_grid, theta_grid = np.meshgrid(z_br, theta)
    
    x_br = main_radius * np.cos(theta_grid)
    y_br = z_grid
    z_br = np.ones_like(x_br) * (main_length/2)
    
    # Combine
    x_all = np.concatenate([x_main.flatten(), x_br.flatten()])
    y_all = np.concatenate([y_main.flatten(), y_br.flatten()])
    z_all = np.concatenate([z_main.flatten(), z_br.flatten()])
    
    return x_all, y_all, z_all

# ------------------- ENGINEERING CALCULATIONS -------------------
def calculate_flow_regime(velocity, diameter, density, viscosity):
    """Calculate Reynolds number and flow regime"""
    Re = (density * velocity * diameter) / viscosity
    if Re < 2300:
        regime = "Laminar"
    elif Re < 4000:
        regime = "Transition"
    else:
        regime = "Turbulent"
    return Re, regime

def calculate_velocity_profile(x, y, diameter, velocity, Re):
    """Calculate velocity field based on flow regime"""
    if Re < 2300:
        # Parabolic profile for laminar flow
        r = np.sqrt(x**2 + y**2)
        r_max = diameter / 2
        velocity_field = 2 * velocity * (1 - (r/r_max)**2)
    else:
        # 1/7 power law for turbulent flow
        r = np.sqrt(x**2 + y**2)
        r_max = diameter / 2
        velocity_field = velocity * (1 - r/r_max)**(1/7)
    return velocity_field

def calculate_shear_stress(velocity_field, density, viscosity, diameter, geometry_factor=1.0):
    """Calculate wall shear stress with geometry effects"""
    # Fanning friction factor approximation
    Re = np.mean(velocity_field) * density * diameter / viscosity
    if Re < 2300:
        f = 16 / Re
    else:
        f = 0.079 * Re**(-0.25)
    
    tau_w = 0.5 * density * f * velocity_field**2
    return tau_w * geometry_factor

def calculate_geometry_factor(fitting):
    """Get geometry factor for shear stress enhancement"""
    factors = {
        "Straight": 1.0,
        "Elbow": 2.5,  # Higher for elbow due to secondary flows
        "Tee": 3.0,    # Highest for tee due to flow impingement
        "Reducer": 1.8 # Moderate for reducer due to acceleration
    }
    return factors.get(fitting, 1.0)

def fit_corrosion_rate(years, thicknesses):
    """Fit corrosion rate using linear regression"""
    if len(years) < 2:
        return 0.0, 0.0, None
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, thicknesses)
    corrosion_rate = -slope  # Negative slope means decreasing thickness
    
    # Predict future thickness
    future_years = np.arange(years[-1], years[-1] + 20)
    predicted_thk = intercept + slope * future_years
    
    return corrosion_rate, r_value**2, (future_years, predicted_thk)

# ------------------- MAIN EXECUTION -------------------
run_analysis = st.sidebar.button(" Execute Analysis", type="primary")

if run_analysis:
    # ------------------- INPUT VALIDATION -------------------
    try:
        years = np.array([int(y.strip()) for y in years_input.split(",") if y.strip()])
        thicknesses = np.array([float(t.strip()) for t in thk_input.split(",") if t.strip()])
    except:
        st.error("Invalid input format. Please use comma-separated numbers.")
        st.stop()
    
    if len(years) != len(thicknesses):
        st.error(f"Years ({len(years)}) and thickness ({len(thicknesses)}) count mismatch")
        st.stop()
    
    if len(years) < 2:
        st.error("At least 2 inspection points required")
        st.stop()
    
    # Sort by year
    sort_idx = np.argsort(years)
    years = years[sort_idx]
    thicknesses = thicknesses[sort_idx]
    
    if thicknesses[-1] <= min_allow_t:
        st.error(" Current thickness is below minimum allowable!")
    
    # ------------------- FLUID PROPERTIES -------------------
    fluid_props = FLUID_PROPERTIES[fluid]
    material_props = MATERIALS[material]
    
    # Calculate Reynolds number
    Re, flow_regime = calculate_flow_regime(
        velocity, pipe_ID, fluid_props["density"], fluid_props["viscosity"]
    )
    
    # ------------------- CORROSION ANALYSIS -------------------
    corrosion_rate, r_squared, future_pred = fit_corrosion_rate(years, thicknesses)
    
    if corrosion_rate <= 0:
        st.warning(" Calculated corrosion rate is non-positive. Check your thickness data.")
        corrosion_rate = 0.01  # Default minimum
    
    remaining_life = max(0, (thicknesses[-1] - min_allow_t) / corrosion_rate)
    
    # API 570 inspection interval calculation
    next_inspection = min(remaining_life/2, 10.0, max(0.5, remaining_life*0.25))
    
    # ------------------- CFD FIELD CALCULATION -------------------
    # Create grid
    nx, ny = 100, 50
    x = np.linspace(-pipe_ID/2, pipe_ID/2, nx)
    y = np.linspace(-pipe_ID/2, pipe_ID/2, ny)
    X, Y = np.meshgrid(x, y)
    
    # Calculate velocity field
    velocity_field = calculate_velocity_profile(X, Y, pipe_ID, velocity, Re)
    
    # Calculate shear stress
    geo_factor = calculate_geometry_factor(fitting)
    tau_field = calculate_shear_stress(
        velocity_field, fluid_props["density"], 
        fluid_props["viscosity"], pipe_ID, geo_factor
    )
    
    # Risk score based on shear stress and geometry
    risk_score = tau_field / np.max(tau_field) if np.max(tau_field) > 0 else np.zeros_like(tau_field)
    
    # ------------------- MONTE CARLO SIMULATION -------------------
    n_sim = 10000
    # Use lognormal distribution for corrosion rate (more realistic)
    mean_cr = corrosion_rate
    std_cr = 0.3 * corrosion_rate  # 30% COV
    # Convert to lognormal parameters
    sigma = np.sqrt(np.log(1 + (std_cr/mean_cr)**2))
    mu = np.log(mean_cr) - 0.5*sigma**2
    cr_dist = np.random.lognormal(mu, sigma, n_sim)
    cr_dist = cr_dist[cr_dist > 0]
    
    life_dist = (thicknesses[-1] - min_allow_t) / cr_dist
    failure_prob_5yr = np.mean(life_dist < 5)
    failure_prob_10yr = np.mean(life_dist < 10)
    
    # ------------------- 3D GEOMETRY VISUALIZATION -------------------
    st.header("📐 3D Pipe Geometry with Risk Hotspots")
    
    # Create appropriate mesh based on fitting
    if fitting == "Straight":
        Xc, Yc, Zc = create_cylinder_mesh(pipe_length, pipe_ID/2, resolution=30)
        # Create intensity based on position (higher in middle)
        intensity = 0.5 + 0.5 * np.sin(2*np.pi*Zc/pipe_length)
        
    elif fitting == "Elbow":
        Xc, Yc, Zc = create_elbow_mesh(pipe_ID/2, bend_radius=3, resolution=25)
        # Higher intensity at outer bend
        intensity = 0.7 + 0.3 * (Xc - np.min(Xc)) / (np.max(Xc) - np.min(Xc))
        
    elif fitting == "Reducer":
        Xc, Yc, Zc, intensity = create_reducer_mesh(
            reducer_length, reducer_d1/2, reducer_d2/2, resolution=25
        )
        
    elif fitting == "Tee":
        Xc, Yc, Zc = create_tee_mesh(pipe_length, pipe_ID/2, branch_length=pipe_length/3, resolution=25)
        # Higher intensity at branch junction
        center_z = pipe_length/2
        intensity = 0.5 + 0.5 * np.exp(-((Zc - center_z)**2)/(0.1*pipe_length)**2)
    
    # Create 3D plot
    fig3d = go.Figure(data=[
        go.Scatter3d(
            x=Xc, y=Yc, z=Zc,
            mode='markers',
            marker=dict(
                size=3,
                color=intensity,
                colorscale='RdYlBu_r',  # Red = high risk
                cmin=0,
                cmax=1,
                opacity=0.8,
                colorbar=dict(title="Risk Index", x=1.02)
            ),
            name="Risk Distribution"
        )
    ])
    
    fig3d.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        title=f"3D Geometry: {fitting} - Risk Hotspots"
    )
    
    st.plotly_chart(fig3d, use_container_width=True)
    
    # ------------------- RESULTS DASHBOARD -------------------
    st.header(" Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Corrosion Rate", f"{corrosion_rate:.3f} mm/yr", 
                 delta=f"R²={r_squared:.3f}" if r_squared else None)
    with col2:
        st.metric("Remaining Life", f"{remaining_life:.1f} years", 
                 delta_color="inverse" if remaining_life < 10 else "normal")
    with col3:
        st.metric("Next Inspection", f"{next_inspection:.1f} years")
    with col4:
        st.metric("Flow Regime", flow_regime, f"Re={Re:.0f}")
    
    # ------------------- PLOTS -------------------
    st.header("📈 Detailed Analysis Plots")
    
    # Plot 1: Thickness History
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(years, thicknesses, 'bo-', linewidth=2, markersize=8, label="Measured")
    if future_pred:
        future_years, future_thk = future_pred
        ax1.plot(future_years, future_thk, 'r--', linewidth=2, label="Predicted")
    ax1.axhline(y=min_allow_t, color='r', linestyle=':', linewidth=2, label="Min Allowable")
    ax1.fill_between([years[-1], years[-1] + remaining_life], 
                     min_allow_t, thicknesses[-1], alpha=0.2, color='green', label="Remaining Life")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Thickness (mm)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Thickness History & Prediction")
    
    # Plot 2: Velocity Profile
    im1 = ax2.contourf(X, Y, velocity_field, levels=50, cmap='viridis')
    plt.colorbar(im1, ax=ax2, label='Velocity (m/s)')
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(f"Velocity Profile - {flow_regime} Flow")
    ax2.set_aspect('equal')
    
    st.pyplot(fig1)
    plt.close(fig1)
    
    # Plot 3: Shear Stress and Monte Carlo
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
    
    im2 = ax3.contourf(X, Y, tau_field, levels=50, cmap='plasma')
    plt.colorbar(im2, ax=ax3, label='Shear Stress (Pa)')
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title(f"Wall Shear Stress (Max: {np.max(tau_field):.1f} Pa)")
    ax3.set_aspect('equal')
    
    # Monte Carlo histogram
    ax4.hist(life_dist, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=5, color='red', linestyle='--', linewidth=2, label="5-year limit")
    ax4.axvline(x=10, color='orange', linestyle='--', linewidth=2, label="10-year limit")
    ax4.set_xlabel("Remaining Life (years)")
    ax4.set_ylabel("Probability Density")
    ax4.set_title(f"Monte Carlo RBI\n5-yr failure prob: {failure_prob_5yr*100:.1f}%")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    st.pyplot(fig2)
    plt.close(fig2)
    
    # ------------------- DETAILED RESULTS -------------------
    st.header("🔍 Detailed Engineering Analysis")
    
    with st.expander("Fluid & Flow Properties"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Density:** {fluid_props['density']:.1f} kg/m³")
            st.write(f"**Viscosity:** {fluid_props['viscosity']:.6f} Pa·s")
        with col2:
            st.write(f"**Reynolds Number:** {Re:.0f}")
            st.write(f"**Flow Regime:** {flow_regime}")
        with col3:
            st.write(f"**Geometry Factor:** {geo_factor:.2f}")
            st.write(f"**Max Shear Stress:** {np.max(tau_field):.2f} Pa")
    
    with st.expander("Corrosion Analysis Details"):
        st.write(f"**Corrosion Rate:** {corrosion_rate:.4f} mm/year")
        st.write(f"**Regression R²:** {r_squared:.4f}")
        st.write(f"**Remaining Corrosion Allowance:** {thicknesses[-1] - min_allow_t:.2f} mm")
        
        # Create corrosion rate table
        if len(years) > 1:
            yearly_rates = []
            for i in range(len(years)-1):
                rate = (thicknesses[i] - thicknesses[i+1]) / (years[i+1] - years[i])
                yearly_rates.append(rate)
            
            rate_df = pd.DataFrame({
                'Period': [f"{years[i]}-{years[i+1]}" for i in range(len(years)-1)],
                'Corrosion Rate (mm/yr)': yearly_rates
            })
            st.table(rate_df)
    
    with st.expander("Risk Assessment"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("5-year Failure Probability", f"{failure_prob_5yr*100:.1f}%")
            if failure_prob_5yr > 0.1:
                st.error("High risk - Immediate action recommended")
            elif failure_prob_5yr > 0.05:
                st.warning("Moderate risk - Monitor closely")
            else:
                st.success("Low risk - Normal operation")
        
        with col2:
            st.metric("10-year Failure Probability", f"{failure_prob_10yr*100:.1f}%")
            st.write(f"**95% Confidence Life:** {np.percentile(life_dist, 5):.1f} years")
    
    # ------------------- RECOMMENDATIONS -------------------
    st.header(" Recommendations")
    
    recommendations = []
    
    if remaining_life < 5:
        recommendations.append(" **CRITICAL**: Plan for replacement within 2 years")
    elif remaining_life < 10:
        recommendations.append(" **WARNING**: Consider replacement in next turnaround")
    
    if failure_prob_5yr > 0.1:
        recommendations.append(" **HIGH RISK**: Increase inspection frequency to annually")
    elif failure_prob_5yr > 0.05:
        recommendations.append(" **MODERATE RISK**: Maintain current inspection schedule")
    else:
        recommendations.append(" **LOW RISK**: Standard inspection schedule acceptable")
    
    if flow_regime == "Turbulent" and np.max(tau_field) > 100:
        recommendations.append(" **FLOW ISSUE**: High shear stress detected. Consider flow modification")
    
    for rec in recommendations:
        st.write(rec)
    
    # ------------------- EXPORT OPTIONS -------------------
    st.download_button(
        label=" Download Results Summary",
        data=f"""
        CFD-Assisted Corrosion Analysis Report
        =====================================
        
        Geometry: {fitting}
        Material: {material}
        Fluid: {fluid}
        
        Input Parameters:
        - Pipe Length: {pipe_length} m
        - Inner Diameter: {pipe_ID} m
        - Flow Velocity: {velocity} m/s
        - Pressure: {pressure} MPa
        
        Results:
        - Corrosion Rate: {corrosion_rate:.3f} mm/year
        - Remaining Life: {remaining_life:.1f} years
        - Next Inspection: {next_inspection:.1f} years
        - 5-year Failure Probability: {failure_prob_5yr*100:.1f}%
        - Flow Regime: {flow_regime} (Re={Re:.0f})
        
        Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        """,
        file_name=f"corrosion_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )
    
    st.success(" Analysis completed successfully!")
    
else:
    # Initial state
    st.markdown("""
    ## Welcome to the CFD-Assisted Corrosion and RBI Tool
    
    ### How to use:
    1. **Configure Parameters** in the sidebar
    2. **Enter UT thickness history** (comma-separated years and thicknesses)
    3. Click **"Execute Analysis"** to run the simulation
    
    ### Features:
    -  3D geometry visualization with risk hotspots
    -  CFD-based flow and shear stress analysis
    -  Corrosion rate prediction and remaining life calculation
    -  Monte Carlo RBI with failure probabilities
    -  API 570 compliant inspection planning
    
    ### Supported Geometries:
    - Straight pipes
    - 90° elbows
    - Tee junctions
    - Reducers (concentric)
    """)
    
    # Display placeholder image/instructions
    col1, col2 = st.columns(2)
    with col1:
        st.info(" **Tip**: For accurate results, provide at least 3 inspection points over time")
    with col2:
        st.info(" **Note**: Results are engineering estimates. Always consult with qualified engineers for critical decisions.")
