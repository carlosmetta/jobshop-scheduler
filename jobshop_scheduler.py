import pandas as pd
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ----------------------
# Loaders
# ----------------------

def load_job_types_from_excel(file_path):
    """
    Load job types (templates) from an Excel file.
    Returns:
        dict: {type: [(machine_id, base_duration), ...]}
    """
    job_types_df = pd.read_excel(file_path)
    job_types = {}

    for job_type, group in job_types_df.groupby('type'):
        operations = []
        for _, row in group.iterrows():
            operations.append((int(row['machine']), int(row['duration'])))
        job_types[job_type] = operations

    return job_types


def load_job_instances_from_excel(file_path):
    """
    Load job instances from Excel, with multiplier support.
    Returns:
        - job_instances: {job_id: {type, release_time, deadline, multiplier}}
        - instance_to_source: {job_id: row_id}
    """
    job_instances_df = pd.read_excel(file_path)
    job_instances = {}
    instance_to_source = {}

    real_job_id = 0

    for idx, row in job_instances_df.iterrows():
        job_type = str(row['type']).strip().lower()  # lowercase to be safe
        release_time = int(row['release_time'])
        deadline = int(row['deadline'])
        multiplier = int(row.get('multiplier', 1))  # default 1 if missing

        job_instances[real_job_id] = {
            "type": job_type,
            "release_time": release_time,
            "deadline": deadline,
            "multiplier": multiplier
        }
        instance_to_source[real_job_id] = idx
        real_job_id += 1

    return job_instances, instance_to_source

# ----------------------
# Combine Types + Instances (Multiplier Scaling)
# ----------------------

def get_scaled_operations(job_types, job_instance):
    """
    Get operations for a job, scaling durations according to multiplier.

    Args:
        job_types (dict): {type_name: [(machine_id, base_duration)]}
        job_instance (dict): Single job instance with 'multiplier'

    Returns:
        list: [(machine_id, scaled_duration), ...]
    """
    base_operations = job_types[job_instance['type']]
    multiplier = job_instance.get('multiplier', 1)

    scaled_operations = [(machine, duration * multiplier) for machine, duration in base_operations]
    return scaled_operations

# ----------------------
# Solver
# ----------------------

def solve_jobshop(job_types, job_instances, max_time=60):
    """
    Solve the Job Shop Scheduling Problem using Google OR-Tools CP-SAT.

    Args:
        job_types (dict): Job templates.
        job_instances (dict): Job instances with multipliers.
        max_time (int): Max solve time in seconds.

    Returns:
        list: Assigned tasks [(start_time, job_id, task_id, machine_id, duration)]
    """
    model = cp_model.CpModel()

    horizon = min(
        max(
            job_info['deadline']
            for job_info in job_instances.values()
            ),
        sum(
            duration * job_info.get('multiplier', 1)
            for job_info in job_instances.values()
            for _, duration in job_types[job_info['type']]
             )
        )       
    task_starts = {}
    task_ends = {}
    all_tasks = {}

    all_machines = set()
    for template in job_types.values():
        for machine, _ in template:
            all_machines.add(machine)

    # Create Variables
    for job_id, job_info in job_instances.items():
        job_operations = get_scaled_operations(job_types, job_info)

        for task_id, (machine, duration) in enumerate(job_operations):
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
            task_starts[(job_id, task_id)] = start_var
            task_ends[(job_id, task_id)] = end_var
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval_var)

    # NoOverlap on machines
    machine_to_intervals = {machine: [] for machine in all_machines}
    for job_id, job_info in job_instances.items():
        job_operations = get_scaled_operations(job_types, job_info)

        for task_id, (machine, _) in enumerate(job_operations):
            machine_to_intervals[machine].append(all_tasks[(job_id, task_id)][2])

    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Task ordering within jobs
    for job_id, job_info in job_instances.items():
        job_operations = get_scaled_operations(job_types, job_info)

        for task_id in range(len(job_operations) - 1):
            model.Add(task_starts[(job_id, task_id + 1)] >= task_ends[(job_id, task_id)])

    # Release time and deadline constraints
    for job_id, job_info in job_instances.items():
        release_time = job_info['release_time']
        deadline = job_info['deadline']
        num_tasks = len(get_scaled_operations(job_types, job_info))

        model.Add(task_starts[(job_id, 0)] >= release_time)
        model.Add(task_ends[(job_id, num_tasks - 1)] <= deadline)

    # Minimize makespan
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        task_ends[(job_id, len(get_scaled_operations(job_types, job_info)) - 1)]
        for job_id, job_info in job_instances.items()
    ])
    model.Minimize(obj_var)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)

    assigned_tasks = []

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for job_id, job_info in job_instances.items():
            job_operations = get_scaled_operations(job_types, job_info)

            for task_id, (machine, duration) in enumerate(job_operations):
                start = solver.Value(task_starts[(job_id, task_id)])
                assigned_tasks.append((start, job_id, task_id, machine, duration))

        assigned_tasks.sort()
        return assigned_tasks
    else:
        return None

# ----------------------
# Gantt Chart
# ----------------------

def plot_job_view_gantt_grouped(assigned_tasks, job_instances):
    """
    Plot Gantt chart grouped by (type + deadline).
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Step 1: Create group mapping
    group_mapping = {}
    label_counter = defaultdict(int)

    for job_id, info in job_instances.items():
        key = (info['type'], info['deadline'])
        label_counter[key] += 1
        group_mapping[job_id] = key

    group_keys = sorted(label_counter.keys())
    label_mapping = {key: idx for idx, key in enumerate(group_keys)}

    # Step 2: Machine colors
    machine_ids = sorted({task[3] for task in assigned_tasks})
    machine_colors = {machine: plt.cm.tab20(machine % 20) for machine in machine_ids}

    # Step 3: Plot bars
    for start, job_id, task_id, machine_id, duration in assigned_tasks:
        group_key = group_mapping[job_id]
        y_pos = label_mapping[group_key]
        color = machine_colors[machine_id]
        ax.barh(y_pos, duration, left=start, color=color, edgecolor='black')
        ax.text(start + duration / 2, y_pos, f'M{machine_id}', ha='center', va='center', fontsize=7)

    # Step 4: Axis setup
    ax.set_xlabel('Time')
    ax.set_ylabel('Job Groups')
    y_labels = [f"{job_type}_{deadline}_{count}" for (job_type, deadline), count in label_counter.items()]
    ax.set_yticks(list(label_mapping.values()))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()

    # Step 5: Legend
    patches = [mpatches.Patch(color=color, label=f'Machine {machine}') for machine, color in machine_colors.items()]
    ax.legend(handles=patches, loc='upper right')

    plt.title('Job-centric Gantt Chart (Grouped by Type + Deadline)')
    plt.tight_layout()
    return fig


def build_schedule_dataframe(assigned_tasks, job_instances):
    """
    Build a clean, user-friendly DataFrame from assigned tasks.

    Args:
        assigned_tasks (list): List of (start_time, job_id, task_id, machine_id, duration)
        job_instances (dict): {job_id: {type, release_time, deadline, multiplier}}

    Returns:
        pd.DataFrame: Cleaned and labeled schedule DataFrame
    """
    rows = []

    for start_time, job_id, task_id, machine_id, duration in assigned_tasks:
        job_info = job_instances[job_id]
        job_type = job_info['type']
        deadline = job_info['deadline']
        multiplier = job_info.get('multiplier', 1)

        # Build friendly label
        friendly_label = f"{job_type}_DL{deadline}_M{multiplier}"

        rows.append({
            "start_time": start_time,
            "type": job_type,               # Job type (A, B, etc.)
            "job_label": friendly_label,    # Full label
            "deadline": deadline,           # 🆕 Added deadline column
            "task_seq": task_id,             # Task sequence inside the job
            "machine_id": machine_id,        # Assigned machine
            "duration": duration             # Duration
        })

    # Create the DataFrame
    schedule_df = pd.DataFrame(rows)

    # Optional: sort nicely
    schedule_df = schedule_df.sort_values(by=["start_time", "type", "task_seq"])

    return schedule_df




def plot_machine_view_gantt(assigned_tasks, job_instances):
    """
    Plot a Gantt chart grouped by machines, with job names.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Collect job IDs to assign colors
    job_ids = sorted({task[1] for task in assigned_tasks})
    job_colors = {job_id: plt.cm.tab20(job_id % 20) for job_id in job_ids}

    # Machines to plot
    machine_ids = sorted({task[3] for task in assigned_tasks})

    # Plot each task on its machine
    for start_time, job_id, task_id, machine_id, duration in assigned_tasks:
        color = job_colors[job_id]

        # Get the friendly label from job_instances
        job_info = job_instances[job_id]
        job_type = job_info['type']
        deadline = job_info['deadline']
        multiplier = job_info.get('multiplier', 1)
        friendly_label = f"{job_type}_DL{deadline}_M{multiplier}"

        ax.barh(machine_id, duration, left=start_time, color=color, edgecolor='black')
        ax.text(start_time + duration/2, machine_id, friendly_label, va='center', ha='center', fontsize=6)

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(machine_ids)
    ax.set_yticklabels([f"Machine {m}" for m in machine_ids])
    ax.invert_yaxis()

    # Legend
    patches = [mpatches.Patch(color=color, label=f'Job {job_id}') for job_id, color in job_colors.items()]
    ax.legend(handles=patches, loc='upper right', title="Jobs")

    plt.title("Machine-centric Gantt Chart")
    plt.tight_layout()
    return fig


