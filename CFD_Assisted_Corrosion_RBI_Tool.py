import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CFD-Assisted Corrosion and RBI Tool", layout="wide")
st.title("CFD-Assisted Corrosion and RBI Tool – Level 1 (3D Mesh Geometry)")

# ------------------- PHYSICAL CONSTANTS -------------------
FLUID_PROPERTIES = {
    "Water": {"density": 998.2, "viscosity": 0.001002, "corr_factor": 1.0, "pH": 7.0},
    "Glycol": {"density": 1116.0, "viscosity": 0.0161, "corr_factor": 1.2, "pH": 8.5},
    "Steam": {"density": 0.6, "viscosity": 1.34e-5, "corr_factor": 2.5, "pH": 7.0},
    "Acetate": {"density": 1048.0, "viscosity": 0.00078, "corr_factor": 3.0, "pH": 4.5},
    "Gas Condensate": {"density": 780.0, "viscosity": 0.0004, "corr_factor": 1.8, "pH": 5.5}
}

MATERIALS = {
    "Carbon Steel": {"yield_strength": 240, "tensile_strength": 415, "corr_resistance": 1.0},
    "Stainless 316": {"yield_strength": 290, "tensile_strength": 580, "corr_resistance": 0.1},
    "Duplex Steel": {"yield_strength": 550, "tensile_strength": 750, "corr_resistance": 0.05}
}

# ------------------- SIDEBAR INPUTS -------------------
st.sidebar.header(" Geometry & Flow Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    pipe_length = st.number_input("Pipe Length (m)", 1.0, 100.0, 10.0, step=0.5)
    pipe_ID = st.number_input("Pipe Inner Diameter (m)", 0.05, 2.0, 0.3, step=0.01)
with col2:
    velocity = st.number_input("Flow Velocity (m/s)", 0.1, 50.0, 2.5, step=0.1)
    pressure = st.number_input("Operating Pressure (MPa)", 0.1, 30.0, 5.0, step=0.1)

temperature = st.sidebar.slider("Temperature (°C)", 0, 400, 120)
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

st.sidebar.header(" Wall Thickness Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    nominal_t = st.number_input("Nominal Thickness (mm)", 1.0, 50.0, 8.0, step=0.5)
with col2:
    min_allow_t = st.number_input("Min Allowable Thickness (mm)", 1.0, 10.0, 3.5, step=0.1)

st.sidebar.header(" UT Thickness History")
years_input = st.sidebar.text_input("Inspection Years (comma separated)", "2019,2021,2023,2025")
thk_input = st.sidebar.text_input("Measured Thickness (mm)", "8.0,7.2,6.3,5.1")

st.sidebar.header(" Analysis Options")
show_velocity = st.sidebar.checkbox("Show Velocity Profile", True)
show_pressure = st.sidebar.checkbox("Show Pressure Drop", True)
show_wss = st.sidebar.checkbox("Show Wall Shear Stress", True)
show_turbulence = st.sidebar.checkbox("Show Turbulence Intensity", False)

# ------------------- ENGINEERING CALCULATIONS -------------------
def calculate_reynolds(velocity, diameter, density, viscosity):
    """Calculate Reynolds number"""
    return (density * velocity * diameter) / viscosity

def calculate_friction_factor(Re, roughness=0.045e-3):
    """Calculate Darcy friction factor using Churchill correlation"""
    if Re < 2300:
        return 64 / Re  # Laminar flow
    
    # Churchill correlation for turbulent flow
    A = (2.457 * np.log(1/((7/Re)**0.9 + 0.27*roughness/pipe_ID)))**16
    B = (37530/Re)**16
    return 8 * ((8/Re)**12 + 1/(A + B)**1.5)**(1/12)

def calculate_pressure_drop(velocity, diameter, length, density, viscosity, fitting_factor=1.0):
    """Calculate pressure drop using Darcy-Weisbach equation"""
    Re = calculate_reynolds(velocity, diameter, density, viscosity)
    f = calculate_friction_factor(Re)
    delta_p = f * (length/diameter) * (density * velocity**2) / 2 * fitting_factor
    return delta_p / 1e6  # Convert to MPa

def calculate_wall_shear_stress(velocity, diameter, density, viscosity):
    """Calculate wall shear stress"""
    Re = calculate_reynolds(velocity, diameter, density, viscosity)
    f = calculate_friction_factor(Re)
    tau_w = 0.5 * density * velocity**2 * f
    return tau_w

def calculate_corrosion_rate_erosion_corrosion(tau_w, velocity, temperature, pH, material_factor=1.0):
    """Calculate erosion-corrosion rate based on shear stress and velocity"""
    # API RP 14E based erosion rate (simplified)
    velocity_limit = 100 / np.sqrt(FLUID_PROPERTIES[fluid]["density"]) if fluid in FLUID_PROPERTIES else 20
    erosion_factor = max(0, (velocity - velocity_limit) / velocity_limit)
    
    # Shear stress effect
    tau_effect = min(1.0, tau_w / 100)  # Normalize
    
    # Temperature effect (Arrhenius-like)
    temp_effect = np.exp((temperature - 25) / 50)
    
    # pH effect
    pH_effect = 1.0
    if pH < 6:
        pH_effect = 1.5
    elif pH > 8:
        pH_effect = 0.8
    
    base_rate = 0.1  # mm/year base corrosion rate
    return base_rate * material_factor * (1 + 2*erosion_factor + 0.5*tau_effect) * temp_effect * pH_effect

def linear_regression(x, y):
    """Simple linear regression without SciPy"""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0, y_mean, 0
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return slope, intercept, r_squared

# ------------------- 3D GEOMETRY FUNCTIONS -------------------
def create_pipe_segment(length, radius, resolution=20):
    """Create a cylindrical pipe segment"""
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, length, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = z_grid
    
    return x.flatten(), y.flatten(), z.flatten()

def create_elbow_segment(radius, bend_radius=3, angle_start=0, angle_end=90, resolution=20):
    """Create elbow segment"""
    theta = np.linspace(0, 2*np.pi, resolution)
    phi = np.deg2rad(np.linspace(angle_start, angle_end, resolution))
    
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    R = bend_radius * radius
    x = (R + radius * np.cos(theta_grid)) * np.cos(phi_grid)
    y = (R + radius * np.cos(theta_grid)) * np.sin(phi_grid)
    z = radius * np.sin(theta_grid)
    
    return x.flatten(), y.flatten(), z.flatten()

def create_tee_junction(main_radius, branch_radius, main_length, resolution=20):
    """Create tee junction"""
    # Main pipe
    x1, y1, z1 = create_pipe_segment(main_length/2, main_radius, resolution)
    x2, y2, z2 = create_pipe_segment(main_length/2, main_radius, resolution)
    z2 += main_length/2  # Offset second half
    
    # Branch
    theta = np.linspace(0, 2*np.pi, resolution)
    y_br = np.linspace(-branch_radius, branch_radius, resolution)
    theta_grid, y_grid = np.meshgrid(theta, y_br)
    
    x_br = branch_radius * np.cos(theta_grid)
    z_br = np.ones_like(x_br) * (main_length/2)
    y_br = y_grid
    
    # Combine
    x_all = np.concatenate([x1, x2, x_br.flatten()])
    y_all = np.concatenate([y1, y2, y_br.flatten()])
    z_all = np.concatenate([z1, z2, z_br.flatten()])
    
    return x_all, y_all, z_all

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
    
    # ------------------- FLUID & MATERIAL PROPERTIES -------------------
    fluid_props = FLUID_PROPERTIES[fluid]
    material_props = MATERIALS[material]
    
    # ------------------- ENGINEERING CALCULATIONS -------------------
    Re = calculate_reynolds(velocity, pipe_ID, fluid_props["density"], fluid_props["viscosity"])
    flow_regime = "Laminar" if Re < 2300 else "Transition" if Re < 4000 else "Turbulent"
    
    # Calculate wall shear stress at pipe wall
    tau_w = calculate_wall_shear_stress(velocity, pipe_ID, fluid_props["density"], fluid_props["viscosity"])
    
    # Calculate geometry factors for different fittings
    geometry_factors = {
        "Straight": 1.0,
        "Elbow": 2.5,  # Higher due to secondary flows
        "Tee": 3.0,    # Highest due to impingement
        "Reducer": 1.8 # Moderate due to acceleration
    }
    geo_factor = geometry_factors.get(fitting, 1.0)
    tau_w_fitting = tau_w * geo_factor
    
    # Calculate pressure drop
    fitting_factors = {"Straight": 1.0, "Elbow": 30, "Tee": 60, "Reducer": 10}
    pressure_drop = calculate_pressure_drop(
        velocity, pipe_ID, pipe_length, 
        fluid_props["density"], fluid_props["viscosity"],
        fitting_factors.get(fitting, 1.0)
    )
    
    # Calculate erosion-corrosion rate
    erosion_corrosion_rate = calculate_corrosion_rate_erosion_corrosion(
        tau_w_fitting, velocity, temperature, 
        fluid_props["pH"], material_props["corr_resistance"]
    )
    
    # Calculate historical corrosion rate
    slope, intercept, r_squared = linear_regression(years, thicknesses)
    historical_corrosion_rate = -slope  # Negative slope means decreasing thickness
    
    # Combined corrosion rate (historical + erosion)
    total_corrosion_rate = historical_corrosion_rate + 0.3 * erosion_corrosion_rate
    
    remaining_life = max(0, (thicknesses[-1] - min_allow_t) / total_corrosion_rate)
    
    # API 570 inspection interval
    next_inspection = min(remaining_life/2, 10.0, max(0.5, remaining_life*0.25))
    
    # ------------------- MONTE CARLO RBI -------------------
    n_sim = 5000
    sigma = 0.3 * total_corrosion_rate
    cr_dist = np.random.normal(total_corrosion_rate, sigma, n_sim)
    cr_dist = np.abs(cr_dist)  # Ensure positive
    
    life_dist = (thicknesses[-1] - min_allow_t) / cr_dist
    failure_prob_5yr = np.mean(life_dist < 5)
    failure_prob_10yr = np.mean(life_dist < 10)
    
    # ------------------- 3D VISUALIZATION WITH SHEAR STRESS -------------------
    st.header(" 3D Pipe Geometry with Engineering Parameters")
    
    # Create base geometry
    if fitting == "Straight":
        Xc, Yc, Zc = create_pipe_segment(pipe_length, pipe_ID/2, resolution=30)
        # Create shear stress distribution (higher near wall)
        intensity = 0.5 + 0.5 * np.sin(2*np.pi*Zc/pipe_length) * (1 + 0.3*np.random.randn(len(Zc)))
        
    elif fitting == "Elbow":
        Xc, Yc, Zc = create_elbow_segment(pipe_ID/2, bend_radius=3, resolution=25)
        # Higher shear stress at outer bend
        angle = np.arctan2(Yc, Xc)
        intensity = 0.7 + 0.3 * np.sin(angle)
        
    elif fitting == "Reducer":
        # Create tapered geometry
        z_pos = np.linspace(0, reducer_length, 100)
        radii = reducer_d1/2 + (reducer_d2/2 - reducer_d1/2) * (z_pos/reducer_length)
        
        points = []
        intensities = []
        for i in range(len(z_pos)-1):
            theta = np.linspace(0, 2*np.pi, 20)
            z = np.ones_like(theta) * z_pos[i]
            r = np.ones_like(theta) * radii[i]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Higher shear at smaller end (higher velocity)
            intensity_loc = 0.5 + 0.5 * (1 - radii[i]/max(radii))
            points.append(np.column_stack([x, y, z]))
            intensities.extend([intensity_loc] * len(theta))
        
        points = np.vstack(points)
        Xc, Yc, Zc = points[:, 0], points[:, 1], points[:, 2]
        intensity = np.array(intensities)
        
    elif fitting == "Tee":
        Xc, Yc, Zc = create_tee_junction(pipe_ID/2, pipe_ID/2, pipe_length, resolution=20)
        # Higher shear at branch junction
        center_z = pipe_length/2
        intensity = 0.5 + 0.5 * np.exp(-((Zc - center_z)**2)/(0.1*pipe_length)**2)
    
    # Create 3D plot with shear stress visualization
    fig3d = go.Figure(data=[
        go.Scatter3d(
            x=Xc, y=Yc, z=Zc,
            mode='markers',
            marker=dict(
                size=4,
                color=intensity if 'intensity' in locals() else 0.5,
                colorscale='Viridis',  # Good for shear stress
                cmin=0,
                cmax=1,
                opacity=0.8,
                colorbar=dict(
                    title="Shear Stress<br>Intensity",
                    x=1.02,
                    titleside="right"
                ),
                showscale=True
            ),
            name="Wall Shear Stress Distribution",
            hovertemplate="<b>Shear Stress Index:</b> %{marker.color:.2f}<br>" +
                         "X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>"
        )
    ])
    
    # Add streamlines for flow visualization
    if fitting == "Straight":
        # Add flow direction arrows
        arrow_x = [0, pipe_length]
        arrow_y = [0, 0]
        arrow_z = [0, 0]
        fig3d.add_trace(go.Scatter3d(
            x=arrow_x, y=arrow_y, z=arrow_z,
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=3, color='blue'),
            name="Flow Direction"
        ))
    
    fig3d.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            )
        ),
        height=600,
        title=f"{fitting} Geometry - Wall Shear Stress Visualization<br>Max Shear: {tau_w_fitting:.1f} Pa",
        showlegend=True
    )
    
    st.plotly_chart(fig3d, use_container_width=True)
    
    # ------------------- ENGINEERING RESULTS DASHBOARD -------------------
    st.header(" Engineering Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Reynolds Number", f"{Re:.0f}", flow_regime)
    with col2:
        st.metric("Wall Shear Stress", f"{tau_w_fitting:.1f} Pa", 
                 f"{geo_factor:.1f}× geometry factor")
    with col3:
        st.metric("Pressure Drop", f"{pressure_drop:.3f} MPa", 
                 "per fitting" if fitting != "Straight" else "total")
    with col4:
        st.metric("Flow Velocity", f"{velocity:.1f} m/s", 
                 "OK" if velocity < 20 else "High")
    
    # ------------------- NEW ENGINEERING ANALYSIS PLOTS -------------------
    st.header(" Advanced Engineering Analysis")
    
    # Plot 1: Shear Stress vs Velocity Correlation
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1A: Shear stress along pipe length
    x_positions = np.linspace(0, pipe_length, 50)
    if fitting == "Elbow":
        shear_profile = tau_w_fitting * (0.8 + 0.4 * np.sin(2*np.pi*x_positions/pipe_length))
    elif fitting == "Tee":
        shear_profile = tau_w_fitting * (0.7 + 0.6 * np.exp(-((x_positions - pipe_length/2)**2)/(0.05*pipe_length)**2))
    else:
        shear_profile = tau_w_fitting * np.ones_like(x_positions)
    
    axes[0].plot(x_positions, shear_profile, 'b-', linewidth=2)
    axes[0].fill_between(x_positions, 0, shear_profile, alpha=0.3, color='blue')
    axes[0].set_xlabel("Position along pipe (m)")
    axes[0].set_ylabel("Shear Stress (Pa)")
    axes[0].set_title("Shear Stress Distribution")
    axes[0].grid(True, alpha=0.3)
    
    # 1B: Velocity effect on shear stress
    vel_range = np.linspace(0.1, 10, 50)
    tau_range = [calculate_wall_shear_stress(v, pipe_ID, fluid_props["density"], fluid_props["viscosity"]) * geo_factor 
                 for v in vel_range]
    axes[1].plot(vel_range, tau_range, 'r-', linewidth=2)
    axes[1].axvline(x=velocity, color='k', linestyle='--', label=f'Current: {velocity} m/s')
    axes[1].set_xlabel("Velocity (m/s)")
    axes[1].set_ylabel("Shear Stress (Pa)")
    axes[1].set_title("Velocity vs Shear Stress")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 1C: Corrosion rate contributors
    contributors = ['Historical', 'Erosion', 'Temperature', 'Material']
    values = [
        historical_corrosion_rate,
        erosion_corrosion_rate,
        0.1 * historical_corrosion_rate * (temperature/100),
        material_props["corr_resistance"] * historical_corrosion_rate
    ]
    colors = ['blue', 'red', 'orange', 'green']
    bars = axes[2].bar(contributors, values, color=colors, alpha=0.7)
    axes[2].set_ylabel("Corrosion Rate (mm/yr)")
    axes[2].set_title("Corrosion Rate Contributors")
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    
    # ------------------- RISK MATRIX PLOT -------------------
    st.subheader(" Risk Assessment Matrix")
    
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    # Define risk matrix
    consequence = min(10, failure_prob_10yr * 100)  # Scale to 0-10
    likelihood = min(10, (total_corrosion_rate * 10))  # Scale to 0-10
    
    # Create risk matrix background
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    risk_level = X * Y  # Simple risk = consequence × likelihood
    
    # Plot risk contours
    contour = ax.contourf(X, Y, risk_level, levels=20, cmap='RdYlGn_r', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Risk Index')
    
    # Plot current position
    ax.scatter(consequence, likelihood, s=200, c='red', edgecolors='black', 
              marker='*', label=f'Current Risk\nC={consequence:.1f}, L={likelihood:.1f}')
    
    # Add risk zones
    ax.fill_between([0, 3], 0, 3, alpha=0.2, color='green', label='Low Risk')
    ax.fill_between([3, 7], 0, 7, alpha=0.2, color='yellow', label='Medium Risk')
    ax.fill_between([7, 10], 0, 10, alpha=0.2, color='red', label='High Risk')
    
    ax.set_xlabel("Consequence (Failure Probability %)")
    ax.set_ylabel("Likelihood (Corrosion Rate × 10)")
    ax.set_title("Risk Assessment Matrix")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig2)
    plt.close(fig2)
    
    # ------------------- CORROSION MECHANISM ANALYSIS -------------------
    st.subheader(" Corrosion Mechanism Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Primary Corrosion Mechanisms:**")
        
        mechanisms = []
        if erosion_corrosion_rate > 0.05:
            mechanisms.append((" **Erosion-Corrosion**", 
                             f"High shear stress ({tau_w_fitting:.1f} Pa) enhances corrosion"))
        
        if velocity > 20:
            mechanisms.append((" **Flow-Accelerated Corrosion**", 
                             f"High velocity ({velocity:.1f} m/s) exceeds API RP 14E limits"))
        
        if fluid_props["pH"] < 6:
            mechanisms.append((" **Acidic Corrosion**", 
                             f"Low pH ({fluid_props['pH']:.1f}) increases corrosion rate"))
        
        if temperature > 150:
            mechanisms.append((" **High Temperature Corrosion**", 
                             f"Temperature {temperature}°C accelerates reactions"))
        
        if not mechanisms:
            mechanisms.append((" **General Corrosion**", 
                             "Uniform material loss at calculated rate"))
        
        for mech, desc in mechanisms:
            st.markdown(f"- {mech}: {desc}")
    
    with col2:
        st.markdown("**Mitigation Recommendations:**")
        
        recommendations = []
        if erosion_corrosion_rate > historical_corrosion_rate:
            recommendations.append("Consider **flow reduction** or **diameter increase**")
        
        if tau_w_fitting > 50:
            recommendations.append("Install **flow straighteners** or **diffusers**")
        
        if failure_prob_5yr > 0.1:
            recommendations.append("**Immediate inspection** required (within 6 months)")
        
        if total_corrosion_rate > 0.5:
            recommendations.append("Apply **corrosion inhibitors** or **coatings**")
        
        if material == "Carbon Steel" and fluid in ["Acetate", "Gas Condensate"]:
            recommendations.append("Consider **material upgrade** to corrosion-resistant alloy")
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # ------------------- DETAILED RESULTS TABLE -------------------
    st.subheader(" Detailed Analysis Summary")
    
    results_data = {
        "Parameter": [
            "Flow Regime", "Reynolds Number", "Friction Factor",
            "Wall Shear Stress", "Pressure Drop", "Erosion-Corrosion Rate",
            "Historical Corrosion Rate", "Total Corrosion Rate",
            "Remaining Life", "5-Year Failure Probability",
            "API 570 Inspection Interval"
        ],
        "Value": [
            flow_regime, f"{Re:.0f}", f"{calculate_friction_factor(Re):.4f}",
            f"{tau_w_fitting:.2f} Pa", f"{pressure_drop:.4f} MPa",
            f"{erosion_corrosion_rate:.3f} mm/yr", f"{historical_corrosion_rate:.3f} mm/yr",
            f"{total_corrosion_rate:.3f} mm/yr", f"{remaining_life:.1f} years",
            f"{failure_prob_5yr*100:.1f}%", f"{next_inspection:.1f} years"
        ],
        "Status": [
            "✓" if flow_regime == "Laminar" else "⚠" if flow_regime == "Transition" else "🔴",
            "✓" if Re < 4000 else "⚠" if Re < 10000 else "🔴",
            "✓",
            "✓" if tau_w_fitting < 30 else "⚠" if tau_w_fitting < 100 else "🔴",
            "✓" if pressure_drop < 0.1 else "⚠" if pressure_drop < 0.5 else "🔴",
            "✓" if erosion_corrosion_rate < 0.05 else "⚠" if erosion_corrosion_rate < 0.2 else "🔴",
            "✓" if historical_corrosion_rate < 0.1 else "⚠" if historical_corrosion_rate < 0.5 else "🔴",
            "✓" if total_corrosion_rate < 0.2 else "⚠" if total_corrosion_rate < 0.5 else "🔴",
            "✓" if remaining_life > 10 else "⚠" if remaining_life > 5 else "🔴",
            "✓" if failure_prob_5yr < 0.05 else "⚠" if failure_prob_5yr < 0.1 else "🔴",
            "✓"
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # ------------------- RECOMMENDATIONS -------------------
    st.header(" Actionable Recommendations")
    
    # Priority-based recommendations
    if failure_prob_5yr > 0.15:
        st.error("""
         **HIGH PRIORITY ACTION REQUIRED**
        - Immediate shutdown and inspection recommended
        - Consider temporary repair or replacement
        - Reduce operating pressure by 25% immediately
        """)
    elif failure_prob_5yr > 0.08:
        st.warning("""
         **MEDIUM PRIORITY ACTION**
        - Schedule inspection within 6 months
        - Monitor thickness with ultrasonic testing quarterly
        - Consider corrosion inhibitor injection
        """)
    else:
        st.success("""
        🟢 **LOW RISK - MONITORING RECOMMENDED**
        - Continue with normal inspection schedule
        - Monitor corrosion rate annually
        - Consider predictive maintenance program
        """)
    
    # Specific recommendations
    st.markdown("### Specific Mitigation Measures")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("**Short-term (0-6 months):**")
        if tau_w_fitting > 50:
            st.markdown("- Install **vortex breakers** to reduce turbulence")
        if velocity > 15:
            st.markdown("- Reduce flow velocity by **10-20%**")
        if erosion_corrosion_rate > 0.1:
            st.markdown("- Start **corrosion inhibitor** program")
    
    with rec_col2:
        st.markdown("**Long-term (6-24 months):**")
        if remaining_life < 5:
            st.markdown("- Plan for **pipe replacement** in next turnaround")
        if material == "Carbon Steel" and total_corrosion_rate > 0.3:
            st.markdown("- Consider **material upgrade** to duplex stainless")
        if fitting in ["Elbow", "Tee"] and failure_prob_5yr > 0.05:
            st.markdown("- Install **wear plates** or **reinforcements**")
    
    # ------------------- EXPORT -------------------
    st.download_button(
        label=" Download Full Analysis Report",
        data=results_df.to_csv(index=False),
        file_name=f"corrosion_analysis_report_{fitting}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.balloons()
    
else:
    # Initial state
    st.markdown("""
    ##  CFD-Assisted Corrosion & RBI Analysis Tool
    
    ### **New Features Added:**
    1. **Wall Shear Stress Visualization** - Now clearly visible in 3D
    2. **Engineering Analysis Plots** - Shear stress distribution, velocity effects
    3. **Risk Matrix** - Visual risk assessment
    4. **Corrosion Mechanism Analysis** - Identify root causes
    5. **Actionable Recommendations** - Priority-based mitigation measures
    
    ### **Key Engineering Parameters Calculated:**
    - Reynolds number and flow regime
    - Wall shear stress with geometry factors
    - Pressure drop using Darcy-Weisbach
    - Erosion-corrosion rates
    - Failure probabilities via Monte Carlo
    
    ### **How to Use:**
    1. Configure parameters in sidebar
    2. Enter inspection history
    3. Select analysis options
    4. Click **Execute Analysis**
    
    ---
    
    *Note: This tool provides engineering estimates. Final decisions should be made by qualified engineers.*
    """)

    # Display example outputs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://via.placeholder.com/300x200/4B8BBE/FFFFFF?text=3D+Geometry", 
                caption="3D Geometry Visualization")
    with col2:
        st.image("https://via.placeholder.com/300x200/306998/FFFFFF?text=Shear+Stress", 
                caption="Shear Stress Analysis")
    with col3:
        st.image("https://via.placeholder.com/300x200/FFD43B/000000?text=Risk+Matrix", 
                caption="Risk Assessment")t with qualified engineers for critical decisions.")
