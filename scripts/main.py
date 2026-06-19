#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from config import *

# --- Configuration ---
# Ensure we are executing scripts in the same directory as main.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name, args=None):
    """Safely executes another Python script and waits for it to finish."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    
    if not os.path.exists(script_path):
        print(f"\n[!] CRITICAL: Could not find {script_name} in {SCRIPT_DIR}")
        return False

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
        
    print(f"\n>> Executing: {script_name} {' '.join(args if args else [])}")
    print("-" * 60)
    
    try:
        # Run the script and stream its output to the console directly
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
        print("-" * 60)
        print(f">> Successfully completed: {script_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"[!] FAILED: {script_name} exited with error code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print("\n[!] Pipeline interrupted by user.")
        sys.exit(1)

# --- Workflow Definitions ---
def run_base():
    print("\n=== Starting BASE Workflow ===")
    if not run_script("base_scheduler.py"): return

def run_vertical():
    print("\n=== Starting VERTICAL Workflow ===")
    if not run_script("doped_generator.py"): return
    if not run_script("vertical_scheduler.py"): return
    if not run_script("difference_plotter.py", ["--mode", "1", "--overwrite", "n"]): return
    if not run_script("vertical_analyzer.py"): return

def run_additivity():
    print("\n=== Starting ADDITIVITY Workflow ===")
    if not run_script("double_doped_generator.py"): return
    if not run_script("additivity_scheduler.py"): return
    if not run_script("difference_plotter.py", ["--mode", "2", "--overwrite", "n"]): return
    if not run_script("additivity_analyzer.py"): return

def run_stress():
    print("\n=== Starting STRESS TEST Workflow ===")
    if not run_script("stress_test_generator.py"): return
    if not run_script("stress_scheduler.py"): return
    if not run_script("difference_plotter.py", ["--mode", "3", "--overwrite", "n"]): return
    if not run_script("stress_analyzer.py"): return

def run_adiabatic():
    print("\n=== Starting ADIABATIC Workflow ===")
    if not run_script("adiabatic_scheduler.py"): return
    if not run_script("difference_plotter.py", ["--mode", "4", "--overwrite", "n"]): return
    if not run_script("adiabatic_analyzer.py"): return

def run_all_schedulers_only():
    print("\n=== Running ALL Schedulers ===")
    run_script("base_scheduler.py")
    run_script("vertical_scheduler.py")
    run_script("additivity_scheduler.py")
    run_script("stress_scheduler.py")
    run_script("adiabatic_scheduler.py")

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Controller for PAH DFT Project")
    parser.add_argument('--run', type=str, choices=['base', 'vertical', 'additivity', 'stress', 'adiabatic', 'all_schedulers', 'full_pipeline'], help="Bypass menu and run a specific workflow.")
    args = parser.parse_args()

    if args.run:
        choice_map = {
            'base': run_base,
            'vertical': run_vertical,
            'additivity': run_additivity,
            'stress': run_stress,
            'adiabatic': run_adiabatic,
            'all_schedulers': run_all_schedulers_only,
            'full_pipeline': lambda: [run_base(), run_vertical(), run_additivity(), run_stress(), run_adiabatic()]
        }
        choice_map[args.run]()
        return

    # Interactive Menu Fallback
    print("=====================================================")
    print("--- Master Automated Pipeline Controller ---")
    print("=====================================================")
    print("  [1] Base Calculations (Pristine Geometries)")
    print("  [2] Vertical Workflow (Generate -> Schedule -> Plot)")
    print("  [3] Additivity Workflow (Generate -> Schedule -> Plot)")
    print("  [4] Stress Test Workflow (Generate -> Schedule -> Plot)")
    print("  [5] Adiabatic Workflow (Schedule -> Plot)")
    print("  [6] Execute ALL Schedulers Only")
    print("  [7] Execute FULL Pipeline (Run Everything)")
    print("  [0] Exit")
    
    while True:
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == '1': run_base()
        elif choice == '2': run_vertical()
        elif choice == '3': run_additivity()
        elif choice == '4': run_stress()
        elif choice == '5': run_adiabatic()
        elif choice == '6': run_all_schedulers_only()
        elif choice == '7':
            run_base()
            run_vertical()
            run_additivity()
            run_stress()
            run_adiabatic()
        elif choice == '0':
            print("Exiting pipeline.")
            break
        else:
            print("Invalid input. Please enter a number between 0 and 7.")

if __name__ == "__main__":
    main()