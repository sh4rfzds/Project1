#!/usr/bin/env python3

import os
import sys
import re
import subprocess
from config import *

try:
    import numpy as np
except ImportError:
    print("CRITICAL: numpy is not installed. Please run: pip install numpy")
    sys.exit(1)

# --- Configuration ---
CALC_BASE_DIR = CALC_DIRS["base"]

def natural_keys(text):
    """
    Splits strings into text and integer parts for numerical sorting.
    Ensures P2 comes before P10.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def get_short_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def check_if_successful(out_path):
    if not os.path.exists(out_path): return False
    try:
        with open(out_path, "r", errors='ignore') as f:
            return "ORCA TERMINATED NORMALLY" in f.read()
    except: return False

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

def rename_raw_cubes(directory, mol_name, homo_idx):
    """Scans for raw ORCA cube files and permanently renames them to standard format."""
    if not homo_idx or not os.path.exists(directory): return
    
    targets = {
        homo_idx - 1: "h-1",
        homo_idx: "h",
        homo_idx + 1: "l",
        homo_idx + 2: "l+1"
    }
    
    renamed_count = 0
    for f in os.listdir(directory):
        if f.endswith(".cube") and ("mo" in f.lower()):
            match = re.search(r'mo_?0*(\d+)a?\.cube$', f, re.IGNORECASE)
            if match:
                mo_num = int(match.group(1))
                if mo_num in targets:
                    old_path = os.path.join(directory, f)
                    new_path = os.path.join(directory, f"{mol_name}_{targets[mo_num]}.cube")
                    if not os.path.exists(new_path): 
                        os.rename(old_path, new_path)
                        renamed_count += 1
                        
    if renamed_count > 0:
        print(f"      [i] Auto-renamed {renamed_count} raw ORCA .cube files.")

def generate_base_opt_cubes(base_mol):
    """Dynamically generates pristine base cubes from the r2scan opt.gbw file."""
    opt_dir = os.path.join(CALC_BASE_DIR, base_mol, "opt")
    out_file = os.path.join(opt_dir, f"{base_mol}_opt.out")
    gbw_file = os.path.join(opt_dir, f"{base_mol}_opt.gbw")
    
    if not os.path.exists(gbw_file) or not os.path.exists(out_file):
        print(f"      [!] Missing {base_mol}_opt.gbw or .out in {opt_dir}")
        return False
        
    homo_idx = find_homo(out_file)
    if homo_idx is None:
        print(f"      [!] Could not parse HOMO from {out_file}")
        return False
        
    targets = [homo_idx - 1, homo_idx, homo_idx + 1, homo_idx + 2]
    inp_content = f"1\n1\n5\n7\n4\n80\n" # GRID_SIZE = 80
    for t in targets:
        inp_content += f"2\n{t}\n11\n"
    inp_content += "12\n"
    
    inp_file = os.path.join(opt_dir, "plot.in")
    with open(inp_file, "w") as f: f.write(inp_content)
    
    try:
        with open(inp_file, "r") as inputs:
            subprocess.run([ORCA_PLOT_CMD, f"{base_mol}_opt.gbw", "-i"], cwd=opt_dir, stdin=inputs, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"      [!] orca_plot failed: {e}")
        return False
    finally:
        if os.path.exists(inp_file): os.remove(inp_file)
        
    rename_raw_cubes(opt_dir, f"{base_mol}_opt", homo_idx)
    return True

def read_cube(filepath):
    """Parses a Gaussian/ORCA .cube file safely handling the negative natoms MO flag."""
    with open(filepath, 'r') as f: lines = f.readlines()
        
    natoms_raw = int(lines[2].split()[0])
    natoms = abs(natoms_raw)
    
    header_end = 6 + natoms + (1 if natoms_raw < 0 else 0)
    header = lines[:header_end]
    
    nx = abs(int(lines[3].split()[0]))
    ny = abs(int(lines[4].split()[0]))
    nz = abs(int(lines[5].split()[0]))
    
    dx = float(lines[3].split()[1])
    dy = float(lines[4].split()[2])
    dz = float(lines[5].split()[3])
    dv = dx * dy * dz
    
    data_lines = lines[header_end:]
    data = []
    for line in data_lines:
        data.extend([float(x) for x in line.split()])
        
    data_array = np.array(data).reshape((nx, ny, nz))
    return header, data_array, dv

def write_cube(filepath, header, data_array):
    """Writes a 3D numpy array back to .cube format, scrubbed for Avogadro."""
    clean_header = header.copy()
    parts = clean_header[2].split()
    natoms = int(parts[0])
    
    if natoms < 0:
        clean_header[2] = clean_header[2].replace(str(natoms), f" {abs(natoms)}", 1)
        clean_header.pop() 
        
    clean_header[0] = "Difference Density Map\n"
    clean_header[1] = "Generated by difference_plotter.py\n"

    with open(filepath, 'w') as f:
        for line in clean_header: 
            f.write(line)
        flat_data = data_array.flatten()
        for i in range(0, len(flat_data), 6):
            f.write(" ".join(f"{val:13.5E}" for val in flat_data[i:i+6]) + "\n")

def is_analysis_done(calc_dir, doped_mol):
    """Checks if the overlap text and all 4 difference cubes already exist."""
    if not os.path.exists(os.path.join(calc_dir, f"{doped_mol}_overlap.txt")): 
        return False
    for orb in ORBITALS:
        if not os.path.exists(os.path.join(calc_dir, f"{doped_mol}_difference_{orb}.cube")): 
            return False
    return True

def analyze_molecule(base_mol, doped_mol, calc_dir, custom_base_dir=None, custom_base_prefix=None):
    """Calculates overlaps and writes difference maps using Block-Diagonal Logic."""
    if custom_base_dir:
        base_dir = custom_base_dir
        base_out_prefix = custom_base_prefix
        base_cube_prefix = custom_base_prefix
    else:
        base_dir = os.path.join(CALC_BASE_DIR, base_mol, "sp")
        base_out_prefix = f"{base_mol}_sp"
        base_cube_prefix = base_mol

    base_out = os.path.join(base_dir, f"{base_out_prefix}.out")
    doped_out = os.path.join(calc_dir, f"{doped_mol}_sp.out")
    if not os.path.exists(doped_out): doped_out = os.path.join(calc_dir, f"{doped_mol}.out")

    base_homo = find_homo(base_out)
    doped_homo = find_homo(doped_out)

    if not base_homo or not doped_homo:
        print(f"      [!] Error: Could not parse HOMO index from outputs.")
        return False

    rename_raw_cubes(base_dir, base_cube_prefix, base_homo)
    rename_raw_cubes(calc_dir, doped_mol, doped_homo)

    base_cubes = {}
    doped_cubes = {}
    dv = None
    header = None

    try:
        for orb in ORBITALS:
            base_cube_path = os.path.join(base_dir, f"{base_cube_prefix}_{orb}.cube")
            doped_cube_path = os.path.join(calc_dir, f"{doped_mol}_{orb}.cube")
            
            if not os.path.exists(base_cube_path) or not os.path.exists(doped_cube_path):
                print(f"      [!] Error: Missing standardized .cube files for {orb}")
                return False

            b_head, b_data, b_dv = read_cube(base_cube_path)
            d_head, d_data, d_dv = read_cube(doped_cube_path)
            
            norm_b = np.sqrt(np.sum(b_data**2) * b_dv)
            norm_d = np.sqrt(np.sum(d_data**2) * d_dv)
            
            if norm_b > 1e-9: b_data /= norm_b
            if norm_d > 1e-9: d_data /= norm_d
            
            base_cubes[orb] = b_data
            doped_cubes[orb] = d_data
            if dv is None: dv = b_dv
            if header is None: header = b_head
    except Exception as e:
        print(f"      [!] Error processing cube files: {e}")
        return False

    final_map = {}
    overlaps = {}

    def solve_block(orbs):
        orb1, orb2 = orbs
        ov_11 = np.sum(base_cubes[orb1] * doped_cubes[orb1]) * dv
        ov_12 = np.sum(base_cubes[orb1] * doped_cubes[orb2]) * dv
        ov_21 = np.sum(base_cubes[orb2] * doped_cubes[orb1]) * dv
        ov_22 = np.sum(base_cubes[orb2] * doped_cubes[orb2]) * dv
        
        score_direct = abs(ov_11) + abs(ov_22)
        score_swap = abs(ov_12) + abs(ov_21)
        
        if score_direct >= score_swap:
            final_map[orb1] = orb1
            final_map[orb2] = orb2
            overlaps[orb1] = abs(ov_11)
            overlaps[orb2] = abs(ov_22)
        else:
            final_map[orb1] = orb2
            final_map[orb2] = orb1
            overlaps[orb1] = abs(ov_12)
            overlaps[orb2] = abs(ov_21)

    solve_block(["h", "h-1"])
    solve_block(["l", "l+1"])

    overlap_txt_path = os.path.join(calc_dir, f"{doped_mol}_overlap.txt")
    with open(overlap_txt_path, "w") as f:
        f.write(f"Overlap tracking for {base_mol} (Base) vs {doped_mol} (Doped)\n")
        f.write("Base_Orbital\tMatches_Doped\tOverlap_Value\n")
        f.write("-" * 50 + "\n")
        for b_orb in ORBITALS:
            d_orb = final_map[b_orb]
            val = min(overlaps[b_orb], 1.0)
            f.write(f"{b_orb}\t\t{d_orb}\t\t{val:.5f}\n")

    for b_orb, d_orb in final_map.items():
        diff_data = np.abs(doped_cubes[d_orb]) - np.abs(base_cubes[b_orb])
        diff_cube_path = os.path.join(calc_dir, f"{doped_mol}_difference_{b_orb}.cube")
        write_cube(diff_cube_path, header, diff_data)

    return True

def run_vertical_sp_mode(overwrite=False):
    DOPED_MOLECULES_DIR = os.path.join(PROJECT_DIR, "doped_molecules")
    TARGET_DIR = os.path.join(PROJECT_DIR, "calculations/vertical_sp")
    
    if not os.path.exists(DOPED_MOLECULES_DIR):
        print("No doped molecules directory found.")
        return

    base_mols = sorted([d for d in os.listdir(DOPED_MOLECULES_DIR) if os.path.isdir(os.path.join(DOPED_MOLECULES_DIR, d))], key=natural_keys)

    for base_mol in base_mols:
        print(f"\n--- Checking Base Molecule: {base_mol} ---")
        base_sp_dir = os.path.join(CALC_BASE_DIR, base_mol, "sp")
        base_out = os.path.join(base_sp_dir, f"{base_mol}_sp.out")
        
        if check_if_successful(base_out):
            base_homo = find_homo(base_out)
            rename_raw_cubes(base_sp_dir, base_mol, base_homo)

        doped_variants = sorted([get_short_name(f) for f in os.listdir(os.path.join(DOPED_MOLECULES_DIR, base_mol)) if f.endswith(".xyz")], key=natural_keys)
        
        for doped_mol in doped_variants:
            calc_dir = os.path.join(TARGET_DIR, base_mol, doped_mol, "sp")
            if not overwrite and is_analysis_done(calc_dir, doped_mol):
                print(f"  -> [{doped_mol}] Analysis complete. Skipping.")
                continue
                
            out_path = os.path.join(calc_dir, f"{doped_mol}_sp.out")
            if not check_if_successful(out_path):
                print(f"  -> [{doped_mol}] Waiting for ORCA SP to complete...")
                continue
                
            print(f"  -> [{doped_mol}] Calculating Difference Maps...", end="", flush=True)
            if analyze_molecule(base_mol, doped_mol, calc_dir):
                print(" Done.")
            else:
                print(" Failed.")
                
        combined_path = os.path.join(TARGET_DIR, base_mol, f"{base_mol}_overlaps.txt")
        try:
            with open(combined_path, "w") as out_f:
                header_str = f"{'Molecule':<25}"
                for orb in ORBITALS:
                    header_str += f"{orb+'_match':<12}{orb+'_overlap':<15}"
                out_f.write(header_str + "\n")
                
                for doped_mol in doped_variants:
                    calc_dir = os.path.join(TARGET_DIR, base_mol, doped_mol, "sp")
                    overlap_file = os.path.join(calc_dir, f"{doped_mol}_overlap.txt")
                    if os.path.exists(overlap_file):
                        data = {}
                        with open(overlap_file, "r") as in_f:
                            for line in in_f:
                                parts = line.split()
                                if len(parts) == 3 and parts[0] in ORBITALS:
                                    data[parts[0]] = (parts[1], parts[2])
                        if len(data) == len(ORBITALS):
                            row_str = f"{doped_mol:<25}"
                            for orb in ORBITALS:
                                row_str += f"{data[orb][0]:<12}{data[orb][1]:<15}"
                            out_f.write(row_str + "\n")
            print(f"  -> Compiled summary to {base_mol}/{base_mol}_overlaps.txt")
        except Exception as e:
            print(f"  -> [!] Failed to compile combined overlaps: {e}")

def run_additivity_sp_mode(overwrite=False):
    ADDITIVITY_DIR = os.path.join(PROJECT_DIR, "calculations/additivity_sp")
    print(f"\n--- Checking Additivity Runs ---")
    
    if not os.path.exists(ADDITIVITY_DIR):
        print(f"No additivity directory found at {ADDITIVITY_DIR}")
        return

    base_mols = sorted([d for d in os.listdir(ADDITIVITY_DIR) if os.path.isdir(os.path.join(ADDITIVITY_DIR, d))], key=natural_keys)

    for base_mol in base_mols:
        TARGET_DIR = os.path.join(ADDITIVITY_DIR, base_mol)
        print(f"\n  -> Processing Base Molecule: {base_mol}")
        
        base_sp_dir = os.path.join(CALC_BASE_DIR, base_mol, "sp")
        base_out = os.path.join(base_sp_dir, f"{base_mol}_sp.out")
        if check_if_successful(base_out):
            base_homo = find_homo(base_out)
            rename_raw_cubes(base_sp_dir, base_mol, base_homo)

        calc_folders = sorted([d for d in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, d))], key=natural_keys)
        doped_variants = []
        
        for doped_mol in calc_folders:
            calc_dir = os.path.join(TARGET_DIR, doped_mol, "sp")
            if not os.path.exists(calc_dir): calc_dir = os.path.join(TARGET_DIR, doped_mol)

            if not overwrite and is_analysis_done(calc_dir, doped_mol):
                print(f"    -> [{doped_mol}] Analysis complete. Skipping.")
                doped_variants.append(doped_mol)
                continue
                
            out_path = os.path.join(calc_dir, f"{doped_mol}_sp.out")
            if not os.path.exists(out_path): out_path = os.path.join(calc_dir, f"{doped_mol}.out")

            if not check_if_successful(out_path):
                print(f"    -> [{doped_mol}] Waiting for ORCA SP to complete...")
                continue
                
            print(f"    -> [{doped_mol}] Calculating Difference Maps...", end="", flush=True)
            if analyze_molecule(base_mol, doped_mol, calc_dir):
                print(" Done.")
                doped_variants.append(doped_mol)
            else:
                print(" Failed.")

        combined_path = os.path.join(TARGET_DIR, f"{base_mol}_overlaps.txt")
        try:
            with open(combined_path, "w") as out_f:
                header_str = f"{'Molecule':<30}"
                for orb in ORBITALS:
                    header_str += f"{orb+'_match':<12}{orb+'_overlap':<15}"
                out_f.write(header_str + "\n")
                
                for doped_mol in doped_variants:
                    calc_dir = os.path.join(TARGET_DIR, doped_mol, "sp")
                    if not os.path.exists(calc_dir): calc_dir = os.path.join(TARGET_DIR, doped_mol)
                    overlap_file = os.path.join(calc_dir, f"{doped_mol}_overlap.txt")
                    if os.path.exists(overlap_file):
                        data = {}
                        with open(overlap_file, "r") as in_f:
                            for line in in_f:
                                parts = line.split()
                                if len(parts) == 3 and parts[0] in ORBITALS:
                                    data[parts[0]] = (parts[1], parts[2])
                        if len(data) == len(ORBITALS):
                            row_str = f"{doped_mol:<30}"
                            for orb in ORBITALS:
                                row_str += f"{data[orb][0]:<12}{data[orb][1]:<15}"
                            out_f.write(row_str + "\n")
            print(f"    -> Compiled summary to calculations/additivity_sp/{base_mol}/{base_mol}_overlaps.txt")
        except Exception as e:
            print(f"    -> [!] Failed to compile combined overlaps: {e}")

def run_stress_sp_mode(overwrite=False):
    """Mode 3: Processes the haywire stress test molecules using pure r2scan baselines."""
    base_mol = "CirCor" 
    TARGET_DIR = os.path.join(PROJECT_DIR, f"calculations/stress_sp/{base_mol}")
    
    print(f"\n--- Checking Stress Test Runs (Base: {base_mol}) ---")
    
    if not os.path.exists(TARGET_DIR):
        print(f"No stress test directory found at {TARGET_DIR}")
        return

    base_opt_dir = os.path.join(CALC_BASE_DIR, base_mol, "opt")
    base_prefix = f"{base_mol}_opt"
    
    print("  -> Generating temporary r2scan-3c baseline cubes from opt.gbw...")
    if not generate_base_opt_cubes(base_mol):
        print("  -> [!] Failed to generate base cubes from opt.gbw. Cannot proceed.")
        return

    calc_folders = sorted([d for d in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, d))], key=natural_keys)
    doped_variants = []
    
    for doped_mol in calc_folders:
        calc_dir = os.path.join(TARGET_DIR, doped_mol, "sp")
        if not os.path.exists(calc_dir): calc_dir = os.path.join(TARGET_DIR, doped_mol)

        if not overwrite and is_analysis_done(calc_dir, doped_mol):
            print(f"  -> [{doped_mol}] Analysis complete. Skipping.")
            doped_variants.append(doped_mol)
            continue
            
        out_path = os.path.join(calc_dir, f"{doped_mol}_sp.out")
        if not os.path.exists(out_path): out_path = os.path.join(calc_dir, f"{doped_mol}.out")

        if not check_if_successful(out_path):
            print(f"  -> [{doped_mol}] Waiting for ORCA SP to complete...")
            continue
            
        print(f"  -> [{doped_mol}] Calculating Overlaps & Difference Maps...", end="", flush=True)
        # Pass the custom opt base directories to strictly compare r2scan to r2scan
        if analyze_molecule(base_mol, doped_mol, calc_dir, custom_base_dir=base_opt_dir, custom_base_prefix=base_prefix):
            print(" Done.")
            doped_variants.append(doped_mol)
        else:
            print(" Failed.")

    combined_path = os.path.join(TARGET_DIR, f"{base_mol}_overlaps.txt")
    try:
        with open(combined_path, "w") as out_f:
            header_str = f"{'Molecule':<45}"
            for orb in ORBITALS:
                header_str += f"{orb+'_match':<12}{orb+'_overlap':<15}"
            out_f.write(header_str + "\n")
            
            for doped_mol in doped_variants:
                calc_dir = os.path.join(TARGET_DIR, doped_mol, "sp")
                if not os.path.exists(calc_dir): calc_dir = os.path.join(TARGET_DIR, doped_mol)
                overlap_file = os.path.join(calc_dir, f"{doped_mol}_overlap.txt")
                if os.path.exists(overlap_file):
                    data = {}
                    with open(overlap_file, "r") as in_f:
                        for line in in_f:
                            parts = line.split()
                            if len(parts) == 3 and parts[0] in ORBITALS:
                                data[parts[0]] = (parts[1], parts[2])
                    if len(data) == len(ORBITALS):
                        row_str = f"{doped_mol:<45}"
                        for orb in ORBITALS:
                            row_str += f"{data[orb][0]:<12}{data[orb][1]:<15}"
                        out_f.write(row_str + "\n")
        print(f"  -> Compiled summary to calculations/stress_sp/{base_mol}/{base_mol}_overlaps.txt")
    except Exception as e:
        print(f"  -> [!] Failed to compile combined overlaps: {e}")

    # --- CLEANUP OF BASELINE ORBITALS ---
    print("  -> Cleaning up temporary baseline opt cubes...")
    for orb in ORBITALS:
        cube_path = os.path.join(base_opt_dir, f"{base_prefix}_{orb}.cube")
        if os.path.exists(cube_path):
            try:
                os.remove(cube_path)
            except Exception as e:
                print(f"      [!] Could not remove {cube_path}: {e}")

def run_adiabatic_rename_mode(overwrite=False):
    """Mode 4: Only renames raw mo_*.cube files for the adiabatic runs without calculating overlaps."""
    TARGET_DIR = os.path.join(PROJECT_DIR, "calculations/adiabatic_sp")
    
    print(f"\n--- Checking Adiabatic Runs for MO Renaming ---")
    
    if not os.path.exists(TARGET_DIR):
        print(f"No adiabatic directory found at {TARGET_DIR}")
        return

    base_mols = sorted([d for d in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, d))], key=natural_keys)

    for base_mol in base_mols:
        print(f"\n--- Base Molecule: {base_mol} ---")
        base_dir_path = os.path.join(TARGET_DIR, base_mol)
        doped_folders = sorted([d for d in os.listdir(base_dir_path) if os.path.isdir(os.path.join(base_dir_path, d))], key=natural_keys)
        
        for doped_mol in doped_folders:
            sp_dir = os.path.join(base_dir_path, doped_mol, "sp")
            out_path = os.path.join(sp_dir, f"{doped_mol}_sp.out")
            
            if not os.path.exists(sp_dir) or not os.path.exists(out_path):
                continue
                
            if check_if_successful(out_path):
                homo_idx = find_homo(out_path)
                if homo_idx:
                    # Quick check to see if there are raw cubes to process so we don't spam the console unnecessarily 
                    raw_cubes = [f for f in os.listdir(sp_dir) if f.endswith(".cube") and "mo" in f.lower()]
                    if raw_cubes:
                        print(f"  -> [{doped_mol}] Renaming raw cubes...")
                        rename_raw_cubes(sp_dir, doped_mol, homo_idx)
            else:
                print(f"  -> [{doped_mol}] SP calculation not complete. Skipping.")

import argparse

def main():
    print("=====================================================")
    print("--- Difference Plotter and Overlap Calculator ---")
    print("=====================================================")
    
    # Set up command-line arguments for automated pipelines
    parser = argparse.ArgumentParser(description="Automated Cube Difference Plotter")
    parser.add_argument('--mode', type=str, choices=['1', '2', '3', '4', '5'], help="1:Vertical, 2:Additivity, 3:Stress, 4:Adiabatic, 5:All")
    parser.add_argument('--overwrite', type=str, choices=['y', 'n'], help="Overwrite existing? (y/n)")
    args = parser.parse_args()

    mode_choice = args.mode
    overwrite_choice = True if args.overwrite == 'y' else False

    # If arguments weren't provided in the terminal, fall back to the interactive menu
    if not mode_choice:
        print("\nPlease select the calculation directory to analyze:")
        print("  [1] calculations/vertical_sp   (Single-site permutations)")
        print("  [2] calculations/additivity_sp (Complex co-doping combinations)")
        print("  [3] calculations/stress_sp     (Haywire stress test combinations)")
        print("  [4] calculations/adiabatic_sp  (MO renaming only)")
        print("  [5] Run ALL modes sequentially")
        
        while True:
            choice = input("Enter 1, 2, 3, 4, or 5: ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                mode_choice = choice
                break
            print("Invalid input. Please enter 1, 2, 3, 4, or 5.")
            
    if not args.overwrite:
        while True:
            ow = input("Overwrite existing analyses? (y/n): ").strip().lower()
            if ow == 'y':
                overwrite_choice = True
                break
            elif ow == 'n':
                overwrite_choice = False
                break
            print("Invalid input. Please enter 'y' or 'n'.")

    if mode_choice in ['1', '5']: run_vertical_sp_mode(overwrite=overwrite_choice)
    if mode_choice in ['2', '5']: run_additivity_sp_mode(overwrite=overwrite_choice)
    if mode_choice in ['3', '5']: run_stress_sp_mode(overwrite=overwrite_choice)
    if mode_choice in ['4', '5']: run_adiabatic_rename_mode(overwrite=overwrite_choice)

    print("\n=====================================================")
    print("--- Process Complete ---")

if __name__ == "__main__":
    main()