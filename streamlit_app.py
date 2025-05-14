import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Import your functions
from jobshop_scheduler import (
    load_job_types_from_excel,
    load_job_instances_from_excel,
    solve_jobshop,
    plot_job_view_gantt_grouped,build_schedule_dataframe,plot_machine_view_gantt)

# --------------------------
# Streamlit App
# --------------------------

st.set_page_config(page_title="Job Shop Scheduler", layout="wide")
st.title("⚙️ Job Shop Scheduling Dashboard")

st.sidebar.header("📂 Upload Input Files")
job_types_file = st.sidebar.file_uploader("Upload Job Types (.xlsx)", type=["xlsx"])
job_instances_file = st.sidebar.file_uploader("Upload Job Instances (.xlsx)", type=["xlsx"])
max_solver_time = st.sidebar.slider("⏱ Solver max time (seconds)", 10, 300, 60)
solve_clicked = st.sidebar.button("🚀 Solve Schedule")

# Storage for session state
if "assigned_tasks" not in st.session_state:
    st.session_state.assigned_tasks = None
if "job_types" not in st.session_state:
    st.session_state.job_types = None
if "job_instances" not in st.session_state:
    st.session_state.job_instances = None
if "instance_to_source" not in st.session_state:
    st.session_state.instance_to_source = None

# --------------------------
# Load Uploaded Files
# --------------------------

if job_types_file:
    job_types = load_job_types_from_excel(job_types_file)
    st.session_state.job_types = job_types
    st.subheader("📄 Job Types")
    st.dataframe(pd.read_excel(job_types_file))

if job_instances_file:
    job_instances, instance_to_source = load_job_instances_from_excel(job_instances_file)
    st.session_state.job_instances = job_instances
    st.session_state.instance_to_source = instance_to_source
    st.subheader("📄 Job Instances")
    st.dataframe(pd.read_excel(job_instances_file))

# --------------------------
# Solve Button Logic
# --------------------------

if solve_clicked:
    if st.session_state.job_types and st.session_state.job_instances:
        st.write("🔍 Solving... Please wait...")
        assigned_tasks = solve_jobshop(
            st.session_state.job_types,
            st.session_state.job_instances,
            max_time=max_solver_time
        )
        if assigned_tasks:
            st.success(f"✅ Solution found! Number of tasks: {len(assigned_tasks)}")
            st.session_state.assigned_tasks = assigned_tasks
        else:
            st.error("❌ No solution found. Check input constraints or increase solver time.")
    else:
        st.error("⚠️ Please upload both Job Types and Job Instances files first.")

# --------------------------
# Show Results
# --------------------------

if st.session_state.assigned_tasks:
    st.subheader("📋 Schedule Results")

    schedule_df = build_schedule_dataframe(
        st.session_state.assigned_tasks,
        st.session_state.job_instances
    )

    st.dataframe(schedule_df)

    # Now you can also make it downloadable easily


    st.subheader("📈 Gantt Charts")

    tab1, tab2 = st.tabs(["📄 Job View (Grouped)", "⚙️ Machine View"])

    with tab1:
        fig_job = plot_job_view_gantt_grouped(
            st.session_state.assigned_tasks,
            st.session_state.job_instances
        )
        st.pyplot(fig_job)

    with tab2:
        fig_machine = plot_machine_view_gantt(
        st.session_state.assigned_tasks,
        st.session_state.job_instances
            )
        st.pyplot(fig_machine)



    # Optional: Download button for schedule
    buffer = BytesIO()
    schedule_df.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="💾 Download Schedule as Excel",
        data=buffer,
        file_name="schedule_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
