import os
import csv
import sys
import subprocess
from collections import defaultdict

# User-configurable parameters
EXECUTABLE = "./output4"     # compiled binary file
NUM_RUNS = 5                   # number of repetitions
OUTPUT_CSV = "results1/energy.csv"

# Ensure results directory exists
output_dir = os.path.dirname(OUTPUT_CSV)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Storage for results, keyed by problem size
results = defaultdict(lambda: {
    "runs": None,
    "total_particles": None,
    "e2e_times": [],
    "algo_times": []
})

# Run executable multiple times
for run_id in range(NUM_RUNS):
    print("=== Run {}/{} ===".format(run_id + 1, NUM_RUNS))

    proc = subprocess.Popen(
        [EXECUTABLE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # universal_newlines=True
    )

    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        print("Error running executable:")
        print(stderr)
        sys.exit(1)

    lines = stdout.strip().split("\n")

    # Print raw output for this run
    for line in lines:
        print(line)

    # Skip header, parse data
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue

        problem_size = int(parts[0])
        runs = int(parts[1])
        total_particles = int(parts[2])
        e2e_time = float(parts[3])
        algo_time = float(parts[4])

        entry = results[problem_size]
        entry["runs"] = runs
        entry["total_particles"] = total_particles
        entry["e2e_times"].append(e2e_time)
        entry["algo_times"].append(algo_time)

    print("")

# Write averaged results to CSV
with open(OUTPUT_CSV, "w") as f:
    writer = csv.writer(f)
    writer.writerow([
        "ProblemSize",
        "RUNS",
        "TotalParticles",
        "AvgE2ETime",
        "AvgAlgoTime"
    ])

    for problem_size in sorted(results.keys()):
        entry = results[problem_size]

        avg_e2e = sum(entry["e2e_times"]) / float(len(entry["e2e_times"]))
        avg_algo = sum(entry["algo_times"]) / float(len(entry["algo_times"]))

        writer.writerow([
            problem_size,
            entry["runs"],
            entry["total_particles"],
            "{:.9f}".format(avg_e2e),
            "{:.9f}".format(avg_algo)
        ])

print("Averaged results written to: {}".format(OUTPUT_CSV))
