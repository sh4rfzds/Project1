#!/usr/bin/env python3

import os
import time
import subprocess
import signal
import shutil
import re
from datetime import datetime
from config import *

# --- Configuration ---
CALC_ADIABATIC_DIR = CALC_DIRS["adiabatic"]
DOPED_MOLECULES_DIR = os.path.join(PROJECT_DIR, "doped_molecules")

OPT_TEMPLATE = os.path.join(TEMPLATES_DIR, "opttemplate.inp")
SP_TEMPLATE = os.path.join(TEMPLATES_DIR, "sptemplate.inp")
LOG_FILE = os.path.join(CALC_ADIABATIC_DIR, "adiabatic_workflow.log")

# Global State
shutdown_requested = False

def handle_sigint(signum, frame):
    """Catches Ctrl+C to stop NEW submissions."""
    global shutdown_requested
    if not shutdown_requested:
        log_message("\n[!] CAUTION: Ctrl+C detected.")
        log_message("[!] Stopping NEW job scheduling.")
        log_message("[!] EXISTING jobs are protected and will finish.")
        shutdown_requested = True

signal.signal(signal.SIGINT, handle_sigint)

def log_message(message):
    """Writes to log file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def get_short_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def natural_keys(text):
    """
    Splits strings into text and integer parts for numerical sorting.
    Ensures P2 comes before P10.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def check_if_successful(out_path):
    """Checks for 'ORCA TERMINATED NORMALLY'."""
    if not os.path.exists(out_path): return False
    try:
        with open(out_path, "r", errors='ignore') as f:
            return "ORCA TERMINATED NORMALLY" in f.read()
    except: return False

def parse_stats(out_file):
    """Parses Time and RAM from output."""
    t = "Unknown"
    ram = "0 MB"
    if os.path.exists(out_file):
        with open(out_file, "r", errors='replace') as f:
            for line in f:
                if "TOTAL RUN TIME" in line: 
                    parts = line.split(":", 1)
                    if len(parts) > 1: t = parts[1].strip()
                if "Maximum memory used" in line: 
                    parts = line.split()
                    ram = parts[-2] + " MB"
    return t, ram

def find_homo(out_file):
    """Finds HOMO index from ORCA output."""
    if not os.path.exists(out_file): return None
    last_occ = -1
    reading = False
    try:
        with open(out_file, "r", errors='replace') as f:
            for line in f:
                if "ORBITAL ENERGIES" in line: reading = True; continue
                if reading:
                    if not line.strip() or "----" in line:
                        if last_occ != -1: break
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            if float(parts[1]) > 0: last_occ = int(parts[0])
                        except: continue
    except: return None
    return last_occ

# --- Preparation Functions ---

def prepare_opt(base_mol, mol_name):
    work_dir = os.path.join(CALC_ADIABATIC_DIR, base_mol, mol_name, "opt")
    os.makedirs(work_dir, exist_ok=True)
    
    xyz_src = os.path.join(DOPED_MOLECULES_DIR, base_mol, f"{mol_name}.xyz")
    if not os.path.exists(xyz_src): return None, f"Missing {xyz_src}"

    with open(xyz_src, 'r') as f:
        xyz_lines = f.readlines()
        coords = "".join(xyz_lines[2:]) if len(xyz_lines)>2 else "".join(xyz_lines)

    if not os.path.exists(OPT_TEMPLATE): return None, f"Missing {OPT_TEMPLATE}"
    with open(OPT_TEMPLATE, 'r') as f: template_content = f.read()

    coord_block = f"* xyz {CHARGE} {MULT}\n{coords}*\n"
    final_inp = f"{template_content.strip()}\n\n{coord_block}"
    
    inp_path = os.path.join(work_dir, f"{mol_name}_opt.inp")
    with open(inp_path, "w") as f: f.write(final_inp)
    return inp_path, "Ready"

def prepare_sp(base_mol, mol_name):
    opt_dir = os.path.join(CALC_ADIABATIC_DIR, base_mol, mol_name, "opt")
    work_dir = os.path.join(CALC_ADIABATIC_DIR, base_mol, mol_name, "sp")
    os.makedirs(work_dir, exist_ok=True)
    
    opt_out = os.path.join(opt_dir, f"{mol_name}_opt.out")
    if not check_if_successful(opt_out): return None, "OPT not successfully completed"

    opt_xyz = os.path.join(opt_dir, f"{mol_name}_opt.xyz")
    if not os.path.exists(opt_xyz): return None, "Missing optimized .xyz file"
    
    with open(opt_xyz, 'r') as f:
        xyz_lines = f.readlines()
        coords = "".join(xyz_lines[2:]) if len(xyz_lines)>2 else "".join(xyz_lines)

    gbw_src = os.path.join(opt_dir, f"{mol_name}_opt.gbw")
    gbw_dest = os.path.join(work_dir, f"{mol_name}_guess.gbw")
    if os.path.exists(gbw_src):
        shutil.copy2(gbw_src, gbw_dest)
    else:
        return None, "Missing OPT .gbw file"

    if not os.path.exists(SP_TEMPLATE): return None, f"Missing {SP_TEMPLATE}"
    with open(SP_TEMPLATE, 'r') as f: template_content = f.read()

    moinp_block = f'%moinp "{mol_name}_guess.gbw"'
    coord_block = f"* xyz {CHARGE} {MULT}\n{coords}*\n"
    final_inp = f"{template_content.strip()}\n\n{moinp_block}\n\n{coord_block}"
    
    inp_path = os.path.join(work_dir, f"{mol_name}_sp.inp")
    with open(inp_path, "w") as f: f.write(final_inp)
    return inp_path, "Ready"

def run_plotting(base_mol, mol_name):
    work_dir = os.path.join(CALC_ADIABATIC_DIR, base_mol, mol_name, "sp")
    gbw = f"{mol_name}_sp.gbw"
    out = f"{mol_name}_sp.out"
    
    if not os.path.exists(os.path.join(work_dir, gbw)): 
        log_message(f"Plotting Failed: Missing {gbw}")
        return

    homo_idx = find_homo(os.path.join(work_dir, out))
    if homo_idx is None:
        log_message(f"Plotting Failed: Could not find HOMO in {out}")
        return

    targets = [homo_idx - 1, homo_idx, homo_idx + 1, homo_idx + 2]

    inp_content = f"1\n1\n5\n7\n4\n{GRID_SIZE}\n"
    for t in targets:
        inp_content += f"2\n{t}\n11\n"
    inp_content += "12\n"

    log_message(f"Starting: {mol_name} (PLOT)")
    inp_file = os.path.join(work_dir, "plot.in")
    with open(inp_file, "w") as f: f.write(inp_content)

    try:
        with open(inp_file, "r") as inputs:
            subprocess.run([ORCA_PLOT_CMD, gbw, "-i"], cwd=work_dir, stdin=inputs, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        log_message(f"Job Completed: {mol_name} (PLOT) | SUCCESS")
    except Exception as e:
        log_message(f"Plotting Failed for {mol_name}: {e}")
        
    if os.path.exists(inp_file): os.remove(inp_file)

# --- Main Execution Engine ---

def run_phase(phase_name, suffix, prepare_func, base_mol):
    files = sorted([f for f in os.listdir(os.path.join(DOPED_MOLECULES_DIR, base_mol)) if f.endswith(".xyz")], key=natural_keys)
    job_queue = [get_short_name(f) for f in files]
    
    pending_jobs = []
    for job in job_queue:
        out_path = os.path.join(CALC_ADIABATIC_DIR, base_mol, job, suffix.lower(), f"{job}_{suffix.lower()}.out")
        if not check_if_successful(out_path):
            pending_jobs.append(job)

    if not pending_jobs:
        log_message(f"--- Phase {phase_name} ({base_mol}): Queue (0 jobs) - All completed ---")
        return

    log_message(f"--- Phase {phase_name} ({base_mol}): Queue ({len(pending_jobs)} jobs) ---")
    
    running_jobs = []
    while (pending_jobs or running_jobs) and not shutdown_requested:
        for i in range(len(running_jobs) - 1, -1, -1):
            job = running_jobs[i]
            if job['proc'].poll() is not None:
                name = job['name']
                t, ram = parse_stats(job['out'])
                status = "SUCCESS" if check_if_successful(job['out']) else "FAILED"
                log_message(f"Job Completed: {name} ({suffix}) | {status} | Time: {t} | MaxRAM: {ram}")
                running_jobs.pop(i)

        while len(running_jobs) < MAX_CONCURRENT_JOBS and pending_jobs:
            next_name = pending_jobs.pop(0)
            inp_path, status = prepare_func(base_mol, next_name)
            
            if not inp_path:
                log_message(f"Skipping {next_name} ({suffix}): {status}")
                continue
                
            work_dir = os.path.dirname(inp_path)
            inp_file = os.path.basename(inp_path)
            out_path = os.path.join(work_dir, f"{next_name}_{suffix.lower()}.out")
            
            log_message(f"Starting: {next_name} ({suffix})")
            
            try:
                with open(out_path, "w") as outfile:
                    p = subprocess.Popen([ORCA_CMD, inp_file],
                                         cwd=work_dir,
                                         stdout=outfile,
                                         stderr=subprocess.STDOUT,
                                         start_new_session=True)
                    running_jobs.append({'proc': p, 'name': next_name, 'out': out_path})
            except Exception as e:
                log_message(f"Failed to launch {next_name} ({suffix}): {e}")

        time.sleep(5)

    if shutdown_requested and running_jobs:
        log_message(f"--- Pausing: Waiting for {len(running_jobs)} active jobs to finish ---")
        for job in running_jobs:
            job['proc'].wait()
            log_message(f"Job Completed (Drain): {job['name']} ({suffix})")

def main():
    if not os.path.exists(DOPED_MOLECULES_DIR):
        log_message(f"CRITICAL: {DOPED_MOLECULES_DIR} does not exist.")
        return
    if not os.path.exists(TEMPLATES_DIR):
        log_message(f"CRITICAL: {TEMPLATES_DIR} does not exist. Please add opttemplate.inp and sptemplate.inp")
        return

    os.makedirs(CALC_ADIABATIC_DIR, exist_ok=True)
    log_message("=====================================================")
    log_message("--- Adiabatic (OPT -> SP -> PLOT) Scheduler Started ---")
    log_message("=====================================================")

    base_mols = sorted([d for d in os.listdir(DOPED_MOLECULES_DIR) if os.path.isdir(os.path.join(DOPED_MOLECULES_DIR, d))], key=natural_keys)
    
    for base_mol in base_mols:
        if shutdown_requested: break
            
        log_message(f"\n=====================================================")
        log_message(f"--- Processing Base Molecule: {base_mol} ---")
        log_message(f"=====================================================")

        run_phase("OPT", "OPT", prepare_opt, base_mol)
        if shutdown_requested: break

        run_phase("SP", "SP", prepare_sp, base_mol)
        if shutdown_requested: break
        
        doped_files = sorted([f for f in os.listdir(os.path.join(DOPED_MOLECULES_DIR, base_mol)) if f.endswith(".xyz")], key=natural_keys)
        pending_analysis = []
        
        for f in doped_files:
            mol = get_short_name(f)
            out_path = os.path.join(CALC_ADIABATIC_DIR, base_mol, mol, "sp", f"{mol}_sp.out")
            if check_if_successful(out_path):
                sp_dir = os.path.join(CALC_ADIABATIC_DIR, base_mol, mol, "sp")
                cube_count = len([c for c in os.listdir(sp_dir) if c.endswith(".cube")])
                if cube_count < 4:
                    pending_analysis.append(mol)
                    
        if not pending_analysis:
            log_message(f"--- Phase PLOT ({base_mol}): Queue (0 jobs) - All completed ---")
        else:
            log_message(f"--- Phase PLOT ({base_mol}): Queue ({len(pending_analysis)} jobs) ---")
            for mol in pending_analysis:
                if shutdown_requested: break
                run_plotting(base_mol, mol)

    log_message("\n--- Scheduler Finished ---")

if __name__ == "__main__":
    main()