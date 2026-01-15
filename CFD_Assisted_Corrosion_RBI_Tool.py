import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="CFD-Assisted Corrosion and RBI Tool", layout="wide")
st.title("CFD-Assisted Corrosion and RBI Tool - Engineering Analysis")

# ------------------- PHYSICAL CONSTANTS -------------------
FLUID_PROPERTIES = {
    "Water": {"density": 998.2, "viscosity": 0.001002, "corr_factor": 1.0},
    "Glycol": {"density": 1116.0, "viscosity": 0.0161, "corr_factor": 1.2},
    "Steam": {"density": 0.6, "viscosity": 1.34e-5, "corr_factor": 2.5},
    "Acetate": {"density": 1048.0, "viscosity": 0.00078, "corr_factor": 3.0},
    "Gas Condensate": {"density": 780.0, "viscosity": 0.0004, "corr_factor": 1.8}
}

MATERIALS = {
    "Carbon Steel": {"yield_strength": 240, "tensile_strength": 415, "corr_resistance": 1.0},
    "Stainless 316": {"yield_strength": 290, "tensile_strength": 580, "corr_resistance": 0.1},
    "Duplex Steel": {"yield_strength": 550, "tensile_strength": 750, "corr_resistance": 0.05}
}

# ------------------- SIDEBAR INPUTS -------------------
st.sidebar.header("Geometry & Flow Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    pipe_length = st.number_input("Pipe Length (m)", 1.0, 100.0, 10.0, step=0.5)
    pipe_ID = st.number_input("Pipe Inner Diameter (m)", 0.05, 2.0, 0.3, step=0.01)
with col2:
    velocity = st.number_input("Flow Velocity (m/s)", 0.1, 50.0, 2.5, step=0.1)
    pressure = st.number_input("Operating Pressure (MPa)", 0.1, 30.0, 5.0, step=0.1)

temperature = st.sidebar.slider("Temperature (C)", 0, 400, 120)
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

st.sidebar.header("Wall Thickness Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    nominal_t = st.number_input("Nominal Thickness (mm)", 1.0, 50.0, 8.0, step=0.5)
with col2:
    min_allow_t = st.number_input("Min Allowable Thickness (mm)", 1.0, 10.0, 3.5, step=0.1)

st.sidebar.header("UT Thickness History")
years_input = st.sidebar.text_input("Inspection Years (comma separated)", "2019,2021,2023,2025")
thk_input = st.sidebar.text_input("Measured Thickness (mm)", "8.0,7.2,6.3,5.1")

st.sidebar.header("Analysis Options")
show_wss = st.sidebar.checkbox("Show Wall Shear Stress Analysis", True)
show_risk = st.sidebar.checkbox("Show Risk Matrix", True)
show_corrosion = st.sidebar.checkbox("Show Corrosion Mechanisms", True)

# ------------------- ENGINEERING CALCULATIONS -------------------
def calculate_reynolds(velocity, diameter, density, viscosity):
    """Calculate Reynolds number"""
    return (density * velocity * diameter) / viscosity

def calculate_friction_factor(Re, roughness=0.045e-3):
    """Calculate Darcy friction factor"""
    if Re < 2300:
        return 64 / Re  # Laminar flow
    else:
        # Colebrook-White approximation
        f = 0.25 / (np.log10(roughness/(3.7*pipe_ID) + 5.74/Re**0.9))**2
        return f

def calculate_wall_shear_stress(velocity, diameter, density, viscosity):
    """Calculate wall shear stress"""
    Re = calculate_reynolds(velocity, diameter, density, viscosity)
    f = calculate_friction_factor(Re)
    tau_w = 0.5 * density * velocity**2 * f
    return tau_w

def calculate_pressure_drop(velocity, diameter, length, density, viscosity, fitting_factor=1.0):
    """Calculate pressure drop using Darcy-Weisbach"""
    Re = calculate_reynolds(velocity, diameter, density, viscosity)
    f = calculate_friction_factor(Re)
    delta_p = f * (length/diameter) * (density * velocity**2) / 2 * fitting_factor
    return delta_p / 1e6  # Convert to MPa

def linear_regression(x, y):
    """Simple linear regression"""
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
def create_pipe_mesh(length, radius, n_circ=20, n_length=30):
    """Create cylindrical pipe mesh"""
    theta = np.linspace(0, 2*np.pi, n_circ)
    z = np.linspace(0, length, n_length)
    
    Theta, Z = np.meshgrid(theta, z)
    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)
    
    return X.flatten(), Y.flatten(), Z.flatten()

def create_elbow_mesh(radius, bend_radius=3, n_circ=20, n_bend=20):
    """Create elbow mesh"""
    theta = np.linspace(0, 2*np.pi, n_circ)
    phi = np.linspace(0, np.pi/2, n_bend)
    
    Theta, Phi = np.meshgrid(theta, phi)
    
    R = bend_radius * radius
    X = (R + radius * np.cos(Theta)) * np.cos(Phi)
    Y = (R + radius * np.cos(Theta)) * np.sin(Phi)
    Z = radius * np.sin(Theta)
    
    return X.flatten(), Y.flatten(), Z.flatten()

def create_reducer_mesh(length, r_start, r_end, n_circ=20, n_length=20):
    """Create reducer mesh"""
    z = np.linspace(0, length, n_length)
    theta = np.linspace(0, 2*np.pi, n_circ)
    
    Z_grid, Theta_grid = np.meshgrid(z, theta)
    
    # Linear taper
    radii = r_start + (r_end - r_start) * (Z_grid / length)
    
    X = radii * np.cos(Theta_grid)
    Y = radii * np.sin(Theta_grid)
    
    return X.flatten(), Y.flatten(), Z_grid.flatten()

def create_tee_mesh(main_length, main_radius, branch_length=None, n_circ=20):
    """Create tee junction mesh"""
    if branch_length is None:
        branch_length = main_length / 3
    
    # Main pipe
    x_main, y_main, z_main = create_pipe_mesh(main_length, main_radius, n_circ, 30)
    
    # Branch
    theta = np.linspace(0, 2*np.pi, n_circ)
    y_br = np.linspace(-branch_length/2, branch_length/2, 20)
    Theta_grid, Y_grid = np.meshgrid(theta, y_br)
    
    x_br = main_radius * np.cos(Theta_grid)
    z_br = np.ones_like(x_br) * (main_length/2)
    
    # Combine
    X = np.concatenate([x_main, x_br.flatten()])
    Y = np.concatenate([y_main, Y_grid.flatten()])
    Z = np.concatenate([z_main, z_br.flatten()])
    
    return X, Y, Z

# ------------------- MAIN EXECUTION -------------------
run_analysis = st.sidebar.button("Execute Analysis", type="primary")

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
        st.error("Current thickness is below minimum allowable!")
    
    # ------------------- FLUID PROPERTIES -------------------
    fluid_props = FLUID_PROPERTIES[fluid]
    material_props = MATERIALS[material]
    
    # ------------------- ENGINEERING CALCULATIONS -------------------
    Re = calculate_reynolds(velocity, pipe_ID, fluid_props["density"], fluid_props["viscosity"])
    flow_regime = "Laminar" if Re < 2300 else "Transition" if Re < 4000 else "Turbulent"
    
    # Wall shear stress calculation
    tau_w = calculate_wall_shear_stress(velocity, pipe_ID, fluid_props["density"], fluid_props["viscosity"])
    
    # Geometry factors for shear stress
    geometry_factors = {
        "Straight": 1.0,
        "Elbow": 2.5,
        "Tee": 3.0,
        "Reducer": 1.8
    }
    geo_factor = geometry_factors.get(fitting, 1.0)
    tau_w_fitting = tau_w * geo_factor
    
    # Pressure drop
    fitting_factors = {"Straight": 1.0, "Elbow": 30, "Tee": 60, "Reducer": 10}
    pressure_drop = calculate_pressure_drop(
        velocity, pipe_ID, pipe_length, 
        fluid_props["density"], fluid_props["viscosity"],
        fitting_factors.get(fitting, 1.0)
    )
    
    # Corrosion rate from historical data
    slope, intercept, r_squared = linear_regression(years, thicknesses)
    historical_corrosion_rate = -slope  # Negative slope means decreasing thickness
    
    # Erosion-corrosion rate estimation
    velocity_limit = 100 / np.sqrt(fluid_props["density"])
    erosion_factor = max(0, (velocity - velocity_limit) / velocity_limit)
    erosion_corrosion_rate = 0.05 * erosion_factor * geo_factor
    
    # Total corrosion rate
    total_corrosion_rate = historical_corrosion_rate + erosion_corrosion_rate
    
    # Remaining life
    remaining_life = max(0, (thicknesses[-1] - min_allow_t) / total_corrosion_rate)
    
    # Inspection interval (API 570)
    next_inspection = min(remaining_life/2, 10.0, max(0.5, remaining_life*0.25))
    
    # Monte Carlo RBI
    n_sim = 5000
    sigma = 0.3 * total_corrosion_rate
    cr_dist = np.random.normal(total_corrosion_rate, sigma, n_sim)
    cr_dist = np.abs(cr_dist)
    
    life_dist = (thicknesses[-1] - min_allow_t) / cr_dist
    failure_prob_5yr = np.mean(life_dist < 5)
    failure_prob_10yr = np.mean(life_dist < 10)
    
    # ------------------- 3D VISUALIZATION -------------------
    st.header("3D Pipe Geometry with Shear Stress Visualization")
    
    # Create geometry
    if fitting == "Straight":
        Xc, Yc, Zc = create_pipe_mesh(pipe_length, pipe_ID/2)
        # Create shear stress distribution (higher in middle)
        r = np.sqrt(Xc**2 + Yc**2)
        intensity = 0.5 + 0.5 * np.sin(2*np.pi*Zc/pipe_length) * (1 - r/(pipe_ID/2))
        
    elif fitting == "Elbow":
        Xc, Yc, Zc = create_elbow_mesh(pipe_ID/2)
        # Higher shear at outer bend
        angle = np.arctan2(Yc, Xc)
        intensity = 0.7 + 0.3 * np.sin(angle)
        
    elif fitting == "Reducer":
        Xc, Yc, Zc = create_reducer_mesh(reducer_length, reducer_d1/2, reducer_d2/2)
        # Higher shear at smaller end
        z_norm = Zc / reducer_length
        intensity = 0.5 + 0.5 * (1 - z_norm)
        
    elif fitting == "Tee":
        Xc, Yc, Zc = create_tee_mesh(pipe_length, pipe_ID/2)
        # Higher shear at branch junction
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
                colorscale='Viridis',
                cmin=0,
                cmax=1,
                opacity=0.8,
                colorbar=dict(title="Shear Stress Index", x=1.02),
                showscale=True
            ),
            name="Wall Shear Stress",
            hovertemplate="Shear Stress Index: %{marker.color:.2f}<br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>"
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
        height=500,
        title=f"{fitting} Geometry - Wall Shear Stress Distribution",
        showlegend=True
    )
    
    st.plotly_chart(fig3d, use_container_width=True)
    
    # ------------------- ENGINEERING RESULTS -------------------
    st.header("Engineering Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Reynolds Number", f"{Re:.0f}", flow_regime)
    with col2:
        st.metric("Wall Shear Stress", f"{tau_w_fitting:.1f} Pa", 
                 f"Geometry factor: {geo_factor:.1f}")
    with col3:
        st.metric("Pressure Drop", f"{pressure_drop:.3f} MPa")
    with col4:
        st.metric("Flow Velocity", f"{velocity:.1f} m/s")
    
    # ------------------- WALL SHEAR STRESS ANALYSIS -------------------
    if show_wss:
        st.subheader("Wall Shear Stress Analysis")
        
        fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Shear stress distribution
        x_pos = np.linspace(0, pipe_length, 100)
        if fitting == "Elbow":
            shear_profile = tau_w_fitting * (0.8 + 0.4 * np.sin(2*np.pi*x_pos/pipe_length))
        elif fitting == "Tee":
            shear_profile = tau_w_fitting * (0.7 + 0.6 * np.exp(-((x_pos - pipe_length/2)**2)/(0.05*pipe_length)**2))
        else:
            shear_profile = tau_w_fitting * np.ones_like(x_pos)
        
        axes[0].plot(x_pos, shear_profile, 'b-', linewidth=2)
        axes[0].fill_between(x_pos, 0, shear_profile, alpha=0.3, color='blue')
        axes[0].set_xlabel("Position along pipe (m)")
        axes[0].set_ylabel("Shear Stress (Pa)")
        axes[0].set_title("Shear Stress Distribution")
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Velocity vs shear stress
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
        
        # Plot 3: Shear stress components
        components = ['Base Flow', 'Geometry Effect', 'Fitting Type', 'Flow Regime']
        values = [
            tau_w,
            tau_w_fitting - tau_w,
            0.1 * tau_w_fitting,
            0.05 * tau_w_fitting if flow_regime == "Turbulent" else 0
        ]
        colors = ['blue', 'green', 'orange', 'red']
        bars = axes[2].bar(components, values, color=colors, alpha=0.7)
        axes[2].set_ylabel("Shear Stress (Pa)")
        axes[2].set_title("Shear Stress Components")
        axes[2].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)
    
    # ------------------- RISK MATRIX -------------------
    if show_risk:
        st.subheader("Risk Assessment Matrix")
        
        fig2, ax = plt.subplots(figsize=(8, 6))
        
        # Define risk matrix
        consequence = min(10, failure_prob_10yr * 100)
        likelihood = min(10, (total_corrosion_rate * 10))
        
        # Create risk matrix
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        risk_level = X * Y
        
        # Plot risk contours
        contour = ax.contourf(X, Y, risk_level, levels=20, cmap='RdYlGn_r', alpha=0.6)
        plt.colorbar(contour, ax=ax, label='Risk Index')
        
        # Plot current position
        ax.scatter(consequence, likelihood, s=200, c='red', edgecolors='black', 
                  marker='*', label=f'Current Risk (C={consequence:.1f}, L={likelihood:.1f})')
        
        # Add risk zones
        ax.fill_between([0, 3], 0, 3, alpha=0.2, color='green', label='Low Risk')
        ax.fill_between([3, 7], 0, 7, alpha=0.2, color='yellow', label='Medium Risk')
        ax.fill_between([7, 10], 0, 10, alpha=0.2, color='red', label='High Risk')
        
        ax.set_xlabel("Consequence (Failure Probability %)")
        ax.set_ylabel("Likelihood (Corrosion Rate x 10)")
        ax.set_title("Risk Assessment Matrix")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        plt.close(fig2)
    
    # ------------------- CORROSION ANALYSIS -------------------
    st.subheader("Corrosion Analysis")
    
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Thickness history
    axes[0].plot(years, thicknesses, 'bo-', linewidth=2, markersize=8, label="Measured")
    
    # Predict future
    future_years = np.arange(years[-1], years[-1] + 15)
    future_thk = thicknesses[-1] - total_corrosion_rate * (future_years - years[-1])
    axes[0].plot(future_years, future_thk, 'r--', linewidth=2, label="Predicted")
    
    axes[0].axhline(y=min_allow_t, color='r', linestyle=':', linewidth=2, label="Min Allowable")
    axes[0].fill_between([years[-1], years[-1] + remaining_life], 
                        min_allow_t, thicknesses[-1], alpha=0.2, color='green', label="Remaining Life")
    
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Thickness (mm)")
    axes[0].set_title("Thickness History & Prediction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Monte Carlo results
    axes[1].hist(life_dist, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].axvline(x=5, color='red', linestyle='--', linewidth=2, label="5-year limit")
    axes[1].axvline(x=10, color='orange', linestyle='--', linewidth=2, label="10-year limit")
    axes[1].set_xlabel("Remaining Life (years)")
    axes[1].set_ylabel("Probability Density")
    axes[1].set_title(f"Monte Carlo RBI - 5-yr failure prob: {failure_prob_5yr*100:.1f}%")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    
    # ------------------- CORROSION MECHANISMS -------------------
    if show_corrosion:
        st.subheader("Corrosion Mechanism Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Identified Mechanisms:**")
            
            mechanisms = []
            if erosion_corrosion_rate > 0.05:
                mechanisms.append(("Erosion-Corrosion", 
                                 f"High shear stress ({tau_w_fitting:.1f} Pa) and velocity"))
            
            if velocity > 20:
                mechanisms.append(("Flow-Accelerated Corrosion", 
                                 f"Velocity ({velocity:.1f} m/s) exceeds typical limits"))
            
            if temperature > 150:
                mechanisms.append(("High Temperature Corrosion", 
                                 f"Temperature {temperature}C accelerates reactions"))
            
            if not mechanisms:
                mechanisms.append(("General Corrosion", 
                                 "Uniform material loss at calculated rate"))
            
            for mech, desc in mechanisms:
                st.markdown(f"- **{mech}**: {desc}")
        
        with col2:
            st.markdown("**Mitigation Recommendations:**")
            
            recommendations = []
            if erosion_corrosion_rate > historical_corrosion_rate:
                recommendations.append("Consider flow reduction or diameter increase")
            
            if tau_w_fitting > 50:
                recommendations.append("Install flow straighteners or diffusers")
            
            if failure_prob_5yr > 0.1:
                recommendations.append("Immediate inspection required (within 6 months)")
            
            if total_corrosion_rate > 0.5:
                recommendations.append("Apply corrosion inhibitors or coatings")
            
            if material == "Carbon Steel" and fluid in ["Acetate", "Gas Condensate"]:
                recommendations.append("Consider material upgrade to corrosion-resistant alloy")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    # ------------------- DETAILED RESULTS -------------------
    st.subheader("Detailed Analysis Summary")
    
    results_data = {
        "Parameter": [
            "Flow Regime",
            "Reynolds Number",
            "Wall Shear Stress",
            "Pressure Drop",
            "Historical Corrosion Rate",
            "Erosion-Corrosion Rate",
            "Total Corrosion Rate",
            "Remaining Life",
            "5-Year Failure Probability",
            "Inspection Interval"
        ],
        "Value": [
            flow_regime,
            f"{Re:.0f}",
            f"{tau_w_fitting:.2f} Pa",
            f"{pressure_drop:.4f} MPa",
            f"{historical_corrosion_rate:.3f} mm/yr",
            f"{erosion_corrosion_rate:.3f} mm/yr",
            f"{total_corrosion_rate:.3f} mm/yr",
            f"{remaining_life:.1f} years",
            f"{failure_prob_5yr*100:.1f}%",
            f"{next_inspection:.1f} years"
        ],
        "Status": [
            "OK" if flow_regime == "Laminar" else "Warning" if flow_regime == "Transition" else "High",
            "OK" if Re < 4000 else "Warning" if Re < 10000 else "High",
            "OK" if tau_w_fitting < 30 else "Warning" if tau_w_fitting < 100 else "High",
            "OK" if pressure_drop < 0.1 else "Warning" if pressure_drop < 0.5 else "High",
            "OK" if historical_corrosion_rate < 0.1 else "Warning" if historical_corrosion_rate < 0.5 else "High",
            "OK" if erosion_corrosion_rate < 0.05 else "Warning" if erosion_corrosion_rate < 0.2 else "High",
            "OK" if total_corrosion_rate < 0.2 else "Warning" if total_corrosion_rate < 0.5 else "High",
            "OK" if remaining_life > 10 else "Warning" if remaining_life > 5 else "Critical",
            "OK" if failure_prob_5yr < 0.05 else "Warning" if failure_prob_5yr < 0.1 else "Critical",
            "Recommended"
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # ------------------- RECOMMENDATIONS -------------------
    st.subheader("Actionable Recommendations")
    
    if failure_prob_5yr > 0.15:
        st.error("HIGH PRIORITY ACTION REQUIRED - Immediate shutdown and inspection recommended")
    elif failure_prob_5yr > 0.08:
        st.warning("MEDIUM PRIORITY ACTION - Schedule inspection within 6 months")
    else:
        st.success("LOW RISK - Continue with normal inspection schedule")
    
    # Export button
    st.download_button(
        label="Download Analysis Report",
        data=results_df.to_csv(index=False),
        file_name=f"corrosion_analysis_{fitting}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
else:
    # Initial state
    st.markdown("""
    ## CFD-Assisted Corrosion & RBI Analysis Tool
    
    ### Features:
    1. **3D Geometry Visualization** - Straight pipes, elbows, tees, reducers
    2. **Wall Shear Stress Analysis** - Calculated using engineering correlations
    3. **Flow Analysis** - Reynolds number, flow regime, pressure drop
    4. **Corrosion Rate Prediction** - Historical + erosion-corrosion rates
    5. **Risk-Based Inspection** - Monte Carlo simulation for failure probabilities
    6. **Risk Matrix** - Visual risk assessment
    
    ### How to Use:
    1. Configure parameters in sidebar
    2. Enter inspection history data
    3. Select analysis options
    4. Click Execute Analysis
    
    Note: This tool provides engineering estimates. Final decisions should be made by qualified engineers.
    """)
