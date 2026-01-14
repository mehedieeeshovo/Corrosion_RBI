import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="CFD-Assisted Corrosion and RBI Tool", layout="wide")
st.title("CFD-Assisted Corrosion and RBI Tool – Level 1 (3D Mesh Geometry)")

# ------------------- SIDEBAR INPUTS -------------------
pipe_length = st.sidebar.number_input("Pipe Length (m)", 1.0, 100.0, 10.0)
pipe_ID = st.sidebar.number_input("Pipe Inner Diameter (m)", 0.05, 2.0, 0.3)
velocity = st.sidebar.number_input("Flow Velocity (m/s)", 0.1, 25.0, 2.5)
pressure = st.sidebar.number_input("Operating Pressure (MPa)", 0.1, 30.0, 5.0)

fitting = st.sidebar.selectbox("Pipe Geometry", ["Straight", "Elbow", "Tee", "Reducer"])
fluid = st.sidebar.selectbox("Flowing Fluid", ["Water", "Glycol", "Steam", "Acetate", "Gas Condensate"])

if fitting=="Reducer":
    reducer_option = st.sidebar.selectbox("Reducer Size (inch)", ["4-2","2-1","6-4","3-2"])
    r_start_inch, r_end_inch = map(int, reducer_option.split("-"))
    r_start = r_start_inch*0.0254
    r_end = r_end_inch*0.0254
else:
    r_start = r_end = pipe_ID/2

nominal_t = st.sidebar.number_input("Nominal Thickness (mm)", 1.0, 25.0, 6.02)
min_allow_t = st.sidebar.number_input("Minimum Allowable Thickness (mm)", 1.0, 10.0, 3.0)

st.sidebar.header("UT Thickness History")
years_input = st.sidebar.text_input("Inspection Years (comma separated)", "2019,2021,2023,2025")
thk_input = st.sidebar.text_input("Measured Thickness (mm)", "7.0,6.52,5.73,4.02")

run_analysis = st.sidebar.button(" Execute Analysis")

# ------------------- FUNCTIONS TO BUILD PIPE MESH -------------------
def cylinder_mesh(x_start,x_end,radius,n_points=20):
    """Build straight cylinder mesh along X-axis"""
    theta = np.linspace(0,2*np.pi,n_points)
    x = np.linspace(x_start,x_end,n_points)
    X, Theta = np.meshgrid(x,theta)
    Y = radius*np.cos(Theta)
    Z = radius*np.sin(Theta)
    return X.flatten(), Y.flatten(), Z.flatten()

def elbow_mesh(r,n_points=20,angle_deg=90):
    """Build 90 deg elbow along XY plane"""
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.deg2rad(angle_deg), n_points)
    Phi, Theta = np.meshgrid(phi,theta)
    Rc = 3*r
    X = Rc*(1-np.cos(Phi))
    Y = Rc*np.sin(Phi)
    Z = r*np.sin(Theta)
    return X.flatten(), Y.flatten(), Z.flatten()

def reducer_mesh(x,r_start,r_end,n_points=20):
    """Tapering cylinder"""
    theta = np.linspace(0, 2*np.pi, n_points)
    radius = np.linspace(r_start,r_end,len(x))
    X, Theta = np.meshgrid(x, theta)
    Rc = np.tile(radius,(n_points,1))
    Y = Rc*np.cos(Theta)
    Z = Rc*np.sin(Theta)
    return X.flatten(), Y.flatten(), Z.flatten()

def tee_mesh(pipe_length,pipe_radius,branch_radius,n_points=20):
    """Straight main + branch along Y axis"""
    X_main, Y_main, Z_main = cylinder_mesh(0,pipe_length,pipe_radius,n_points)
    branch_x = np.ones(n_points*n_points)*pipe_length/2
    branch_y = np.tile(np.linspace(-branch_radius,branch_radius,n_points),n_points)
    branch_z = np.zeros_like(branch_x)
    X = np.concatenate([X_main,branch_x])
    Y = np.concatenate([Y_main,branch_y])
    Z = np.concatenate([Z_main,branch_z])
    return X, Y, Z

# ------------------- EXECUTION -------------------
if run_analysis:
    # UT Data
    years = np.array([int(y) for y in years_input.split(",")])
    thicknesses = np.array([float(t) for t in thk_input.split(",")])
    if len(years)!=len(thicknesses):
        st.error("Years and thickness mismatch"); st.stop()
    if thicknesses[-1]<=min_allow_t:
        st.error("Pipe below minimum thickness"); st.stop()

    # Corrosion rate
    time_span = years[-1]-years[0]
    corrosion_rate = (thicknesses[0]-thicknesses[-1])/time_span
    remaining_life = (thicknesses[-1]-min_allow_t)/corrosion_rate
    next_inspection = min(remaining_life*0.5,5.0)

    # CFD simplified fields
    nx, ny = 140, 70
    x = np.linspace(0,pipe_length,nx)
    y = np.linspace(-pipe_ID/2,pipe_ID/2,ny)
    X,Y = np.meshgrid(x,y)
    velocity_field = velocity*(1-(Y/(pipe_ID/2))**2)
    geo_factor = {"Straight":1.0,"Elbow":1.8,"Tee":2.3,"Reducer":1.6}[fitting]
    rho=1000
    tau_field = rho*velocity**2*geo_factor*np.exp(-((X-pipe_length/2)**2)/(0.1*pipe_length)**2)
    risk_score = tau_field/np.max(tau_field)

    # Monte Carlo
    n_sim = 5000
    cr_dist = np.random.normal(corrosion_rate,0.15*corrosion_rate,n_sim)
    cr_dist = cr_dist[cr_dist>0]
    life_dist = (thicknesses[-1]-min_allow_t)/cr_dist
    failure_prob_5yr = np.mean(life_dist<5)

    # ------------------- 3D Geometry + Hotspot -------------------
    st.subheader("3D Pipe Geometry with Corrosion Hotspots")
    if fitting=="Straight":
        Xc,Yc,Zc = cylinder_mesh(0,pipe_length,pipe_ID/2)
    elif fitting=="Elbow":
        Xc,Yc,Zc = elbow_mesh(pipe_ID/2)
    elif fitting=="Reducer":
        Xc,Yc,Zc = reducer_mesh(x,r_start,r_end)
    elif fitting=="Tee":
        Xc,Yc,Zc = tee_mesh(pipe_length,pipe_ID/2,pipe_ID/2)

    fig3d = go.Figure(data=[go.Mesh3d(
        x=Xc, y=Yc, z=Zc,
        intensity=risk_score.flatten() if len(risk_score.flatten())==len(Xc) else np.linspace(0,1,len(Xc)),
        colorscale='YlOrRd',
        opacity=0.7,
        colorbar=dict(title="Risk Index")
    )])
    fig3d.update_layout(scene=dict(aspectmode="data",xaxis_title="X (m)",yaxis_title="Y (m)",zaxis_title="Z (m)"),
                        height=600)
    st.plotly_chart(fig3d,use_container_width=True)

    # ------------------- Thickness & CFD plots -------------------
    future_years = np.arange(years[-1],years[-1]+15)
    future_thk = thicknesses[-1]-corrosion_rate*(future_years-years[-1])
    fig, ax = plt.subplots()
    ax.plot(years,thicknesses,'o-',label="Measured")
    ax.plot(future_years,future_thk,'--',label="Predicted")
    ax.axhline(min_allow_t,color='r',linestyle='--',label="Min Allowable")
    ax.set_xlabel("Year"); ax.set_ylabel("Thickness (mm)")
    ax.legend(); st.pyplot(fig)

    fig, ax = plt.subplots()
    c = ax.contourf(X,Y,velocity_field,50,cmap="viridis")
    plt.colorbar(c); ax.set_title("Velocity Contour (m/s)"); st.pyplot(fig)

    fig, ax = plt.subplots()
    c = ax.contourf(X,Y,tau_field,50,cmap="plasma")
    plt.colorbar(c); ax.set_title("Wall Shear Stress τ (Pa)"); st.pyplot(fig)

    # Monte Carlo
    st.subheader("Monte Carlo RBI – Remaining Life Distribution")
    fig, ax = plt.subplots()
    ax.hist(life_dist,bins=50,density=True)
    ax.axvline(5,color='r',linestyle='--',label="5-year threshold")
    ax.set_xlabel("Remaining Life (years)")
    ax.legend(); st.pyplot(fig)

    # Results
    st.success("Analysis Completed")
    st.write("### Key Results")
    st.write(f"**Corrosion Rate:** {corrosion_rate:.3f} mm/year")
    st.write(f"**Remaining Useful Life:** {remaining_life:.2f} years")
    st.write(f"**Recommended Next Inspection (API 570):** {next_inspection:.2f} years")
    st.write(f"**Failure Probability within 5 years:** {failure_prob_5yr*100:.1f}%")
else:
    st.info("Enter inputs and click **Execute Analysis**")

