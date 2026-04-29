from __future__ import print_function
import subprocess
import re
import os
import sys

# --- CLUSTER CONFIGURATION ---
ALL_NODES = ["gics0", "gics1", "gics2", "gics3"]
MPICXX = "/usr/mpi/gcc/openmpi-1.8.8/bin/mpic++"
MPIRUN = "/usr/mpi/gcc/openmpi-1.8.8/bin/mpirun"
HOSTFILE = "sources.txt"
INPUT_FILE = "input.bin"
BINARY = "./output"

# Search space for configurations
# Total cores will be calculated as P * T
PROCESSORS_LIST = [1, 2, 4]              # Number of MPI Nodes
THREADS_PER_PROC_LIST = [1, 2, 4, 8, 16] # Threads per Node
TARGET_TOTAL_THREADS = [1, 2, 4, 8, 16, 32, 64]

def update_hostfile(p):
    """Writes sources.txt with p processes distributed across nodes."""
    with open(HOSTFILE, "w") as f:
        # Distribute ranks across nodes to maximize memory bandwidth
        if p <= len(ALL_NODES):
            for i in range(p):
                f.write("{0} slots=1\n".format(ALL_NODES[i]))
        else:
            per_node = p // len(ALL_NODES)
            for node in ALL_NODES:
                f.write("{0} slots={1}\n".format(node, per_node))

def compile_code():
    print(">>> Compiling main.cpp, utils.cpp, init.cpp...")
    cmd = "{0} -O3 -fopenmp main.cpp utils.cpp init.cpp -lm -o output".format(MPICXX)
    result = subprocess.call(cmd, shell=True)
    if result != 0:
        print("!!! Compilation failed. Check your C++ code.")
        sys.exit(1)

def run_benchmark(p, t):
    update_hostfile(p)
    
    # Environment variable for OpenMP
    run_env = os.environ.copy()
    run_env["OMP_NUM_THREADS"] = str(t)
    
    cmd = [
        MPIRUN, "-np", str(p), 
        "--hostfile", HOSTFILE,
        BINARY, INPUT_FILE, str(t)
    ]
    
    try:
        # stderr=subprocess.STDOUT merges warnings and errors into one stream to clean up
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)
        stdout, _ = process.communicate()
        
        output = stdout.decode('utf-8') if isinstance(stdout, bytes) else stdout
        
        # Regex to find "Wall Time: 0.000338 s" - ignores the "s" and warnings
        match = re.search(r"Wall Time:\s+([\d.e-]+)", output)
        
        if match:
            return float(match.group(1))
        else:
            # If parsing fails, show the user the output to debug
            print("\n[DEBUG] P={0}, T={1} output was:".format(p, t))
            print(output.strip())
            return None
    except Exception as e:
        print("Execution failed for P={0}, T={1}: {2}".format(p, t, e))
        return None

def main():
    if not os.path.exists(INPUT_FILE):
        print("Error: {0} not found. Ensure it is in the codes directory.".format(INPUT_FILE))
        return

    compile_code()
    
    results_log = []

    header = "\n{0:^12} | {1:^10} | {2:^12} | {3:^10}".format("Total Cores", "MPI (P)", "OMP (T)", "Time (s)")
    print(header)
    print("-" * len(header))

    for total in TARGET_TOTAL_THREADS:
        for p in PROCESSORS_LIST:
            for t in THREADS_PER_PROC_LIST:
                if p * t == total:
                    val = run_benchmark(p, t)
                    if val is not None:
                        print("{0:<12} | {1:<10} | {2:<12} | {3:<10.6f}".format(total, p, t, val))
                        results_log.append({"total": total, "p": p, "t": t, "time": val})

    # Summary of the fastest configurations
    print("\n" + "="*60)
    print("{0:^60}".format("OPTIMAL CONFIGURATIONS PER CORE COUNT"))
    print("="*60)
    print("{0:<15} | {1:<12} | {2:<12} | {3:<10}".format("Total Cores", "Best P", "Best T", "Best Time"))
    print("-" * 60)

    for total in TARGET_TOTAL_THREADS:
        configs = [r for r in results_log if r["total"] == total]
        if configs:
            best = min(configs, key=lambda x: x["time"])
            print("{0:<15} | {1:<12} | {2:<12} | {3:<10.6f}".format(total, best['p'], best['t'], best['time']))

if __name__ == "__main__":
    main()
