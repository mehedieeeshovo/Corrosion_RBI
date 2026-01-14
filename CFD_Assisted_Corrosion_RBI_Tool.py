import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="CFD Assisted Corrosion and RBI Tool", layout="wide")
st.title("CFD Assisted Corrosion and RBI Tool – Level 1 (API 570)")

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("Pipe & Flow Inputs (SI Units)")
pipe_length = st.sidebar.number_input("Pipe Length (m)", 1.0, 100.0, 10.0)
pipe_ID = st.sidebar.number_input("Pipe Inner Diameter (m)", 0.05, 2.0, 0.30)
velocity = st.sidebar.number_input("Flow Velocity (m/s)", 0.1, 25.0, 2.5)
pressure = st.sidebar.number_input("Operating Pressure (MPa)", 0.1, 30.0, 5.0)

fitting = st.sidebar.selectbox("Pipe Geometry", ["Straight", "Elbow", "Tee", "Reducer"])
fluid = st.sidebar.selectbox("Flowing Fluid", ["Water", "Glycol", "Steam", "Acetate", "Gas Condensate"])

# Reducer input
if fitting == "Reducer":
    reducer_option = st.sidebar.selectbox("Reducer Size (inch)", ["4-2", "2-1", "6-4", "3-2"])
    r_start_inch, r_end_inch = map(int, reducer_option.split("-"))
    r_start = r_start_inch * 0.0254  # convert inch to m
    r_end = r_end_inch * 0.0254
else:
    r_start = r_end = pipe_ID / 2

nominal_t = st.sidebar.number_input("Nominal Thickness (mm)", 1.0, 25.0, 6.02)
min_allow_t = st.sidebar.number_input("Minimum Allowable Thickness (mm)", 1.0, 10.0, 3.0)

st.sidebar.header("UT Thickness History")
years_input = st.sidebar.text_input("Inspection Years (comma separated)", "2019,2021,2023,2025")
thk_input = st.sidebar.text_input("Measured Thickness (mm)", "7.0,6.52,5.73,4.02")

run_analysis = st.sidebar.button("▶ Execute Analysis")

# ================================
# 3D PIPE BUILDERS
# ================================
def build_straight_pipe(x, r):
    theta = np.linspace(0, 2*np.pi, 60)
    Xc, Theta = np.meshgrid(x, theta)
    Yc = r * np.cos(Theta)
    Zc = r * np.sin(Theta)
    return Xc, Yc, Zc

def build_elbow_pipe(r, n_length=140, angle_deg=90):
    theta = np.linspace(0, 2*np.pi, 60)
    bend_angle = np.linspace(0, np.deg2rad(angle_deg), n_length)
    Theta, Phi = np.meshgrid(theta, bend_angle)
    Rc = 3 * r
    Xc = Rc * (1 - np.cos(Phi))
    Yc = Rc * np.sin(Phi)
    Zc =
