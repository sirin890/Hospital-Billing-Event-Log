

import pm4py
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Load the event log
# =========================================================
log = pm4py.read_xes("Event Log.xes")


# =========================================================
# 1. ACTIVITY ANALYSIS
# =========================================================

activity_table = (
    log["concept:name"]
    .value_counts()
    .reset_index()
)

activity_table.columns = ["Activity", "Frequency"]

# Calculate percentage from total events
total_events = activity_table["Frequency"].sum()

activity_table["Percentage"] = (
    activity_table["Frequency"] / total_events * 100
)

# Draw Activity Frequency chart
plt.figure(figsize=(12, 7))

bars = plt.barh(
    activity_table["Activity"],
    activity_table["Frequency"]
)

plt.xlabel("Frequency (Number of Events)")
plt.ylabel("Activity")
plt.title("Activity Frequency Analysis")

plt.gca().invert_yaxis()

# Print frequency and percentage
for bar, percentage in zip(bars, activity_table["Percentage"]):

    value = bar.get_width()

    plt.text(
        value,
        bar.get_y() + bar.get_height() / 2,
        f" {int(value):,} ({percentage:.2f}%)",
        va="center",
        ha="left"
    )

# Add space on the right
plt.xlim(
    0,
    activity_table["Frequency"].max() * 1.20
)

plt.tight_layout()
plt.show()


# Print total number of events
total_events = len(log)

print("Total Number of Events:", total_events)

# =========================================================
# 2. CASE DURATION ANALYSIS
# =========================================================

# Make sure timestamp is datetime
log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])

# Get first and last timestamp for each case
case_duration = (
    log.groupby("case:concept:name")["time:timestamp"]  
    .agg(["min", "max"])
)

# Calculate duration in days
case_duration["Duration_Days"] = (
    case_duration["max"] - case_duration["min"]
).dt.total_seconds() / (60 * 60 * 24)


# # =========================================================
# # Print Case Duration Statistics
# # =========================================================

# print("\nCase Duration Analysis:")
# print("----------------------------")

# print(
#     "Number of Cases:",
#     len(case_duration)
# )

# print(
#     "Mean Duration:",
#     round(case_duration["Duration_Days"].mean(), 2),
#     "days"
# )

# print(
#     "Median Duration:",
#     round(case_duration["Duration_Days"].median(), 2),
#     "days"
# )

# print(
#     "Minimum Duration:",
#     round(case_duration["Duration_Days"].min(), 2),
#     "days"
# )

# print(
#     "Maximum Duration:",
#     round(case_duration["Duration_Days"].max(), 2),
#     "days"
# )

# print(
#     "Standard Deviation:",
#     round(case_duration["Duration_Days"].std(), 2),
#     "days"
# )



# =========================================================
# Draw Case Duration Distribution
# =========================================================

plt.figure(figsize=(14, 6))

plt.hist(
    case_duration["Duration_Days"],
    bins=50,
    edgecolor="black"
)

plt.title("Case Duration Distribution")
plt.xlabel("Case Duration (Days)")
plt.ylabel("Number of Cases")

# X-axis: 0, 100, 200, 300, ...
max_duration = case_duration["Duration_Days"].max()

plt.xticks(
    range(0, int(max_duration) + 100, 100),
    rotation=45
)

# Start X-axis from 0
plt.xlim(left=0)

plt.grid(
    True,
    alpha=0.2
)

plt.tight_layout()
plt.show()


# =========================================================
# PROCESS BOTTLENECK MAP
# =========================================================

# Calculate performance DFG
performance_dfg, start_activities, end_activities = (
    pm4py.discover_performance_dfg(log)
)

# Print bottleneck information
print("\n========== BOTTLENECK INDICATORS ==========\n")

# Sort connections by average duration
sorted_bottlenecks = sorted(
    performance_dfg.items(),
    key=lambda x: x[1],
    reverse=True
)

for path, duration in sorted_bottlenecks:

    source = path[0]
    target = path[1]

    duration_days = duration / (60 * 60 * 24)

    print(
        f"{source} -> {target}: "
        f"{duration_days:.2f} days"
    )

