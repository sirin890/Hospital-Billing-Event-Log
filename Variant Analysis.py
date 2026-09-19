import pm4py
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Event Log
# ==========================================

log = pm4py.read_xes("Event Log.xes")

# Make sure events are sorted correctly
log = log.sort_values(
    by=["case:concept:name", "time:timestamp"]
)

# ==========================================
# 2. Create Process Variant for each Case
# ==========================================

case_variants = (
    log.groupby("case:concept:name")["concept:name"]
    .apply(tuple)
)

# ==========================================
# 3. Count each Variant
# ==========================================

variant_counts = case_variants.value_counts()

total_cases = len(case_variants)

print("\nTotal Number of Cases:", total_cases)
print("Total Number of Variants:", len(variant_counts))


# ==========================================
# 4. Create Variant Table
# ==========================================

variant_table = variant_counts.reset_index()

variant_table.columns = [
    "Process_Variant",
    "Number_of_Cases"
]

# Calculate percentage
variant_table["Percentage"] = (
    variant_table["Number_of_Cases"]
    / total_cases
    * 100
)

# Convert tuple to readable path
variant_table["Process_Variant"] = (
    variant_table["Process_Variant"]
    .apply(lambda x: " → ".join(x))
)

# Add Variant numbers
variant_table.insert(
    0,
    "Variant",
    [
        f"Variant {i}"
        for i in range(1, len(variant_table) + 1)
    ]
)


# ==========================================
# 5. Get Top 10 Variants
# ==========================================

top10 = variant_table.head(10).copy()

print("\n========== TOP 10 PROCESS VARIANTS ==========\n")

print(
    top10[
        [
            "Variant",
            "Process_Variant",
            "Number_of_Cases",
            "Percentage"
        ]
    ].to_string(index=False)
)


# ==========================================
# 6. Print each Variant clearly
# ==========================================

print("\n========== VARIANT DETAILS ==========\n")

for _, row in top10.iterrows():

    print(row["Variant"])
    print("Path:", row["Process_Variant"])
    print(
        "Number of Cases:",
        f'{row["Number_of_Cases"]:,}'
    )
    print(
        "Percentage:",
        f'{row["Percentage"]:.3f}%'
    )
    print("-" * 80)


# ==========================================
# 7. Draw Top 10 Process Variants
# ==========================================

plt.figure(figsize=(12, 7))

bars = plt.barh(
    top10["Variant"],
    top10["Number_of_Cases"]
)

# Labels
plt.xlabel("Number of Cases")
plt.ylabel("Process Variant")
plt.title("Top 10 Process Variants")

# Variant 1 at the top
plt.gca().invert_yaxis()

# Grid
plt.grid(
    axis="x",
    linestyle="-",
    alpha=0.25
)

# ==========================================
# 8. Add Number of Cases + Percentage
# ==========================================

for bar, cases, percentage in zip(
    bars,
    top10["Number_of_Cases"],
    top10["Percentage"]
):

    plt.text(
        bar.get_width() + 300,
        bar.get_y() + bar.get_height() / 2,

        f"{cases:,} ({percentage:.2f}%)",

        va="center",
        ha="left",
        fontsize=10
    )


# Give enough space for labels
plt.xlim(
    0,
    top10["Number_of_Cases"].max() * 1.05
)

plt.tight_layout()

plt.show()