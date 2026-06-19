#!/usr/bin/env python3

import os
import sys
import glob
import subprocess
from config import *

# --- Configuration ---
CALC_BASE_DIR = CALC_DIRS["base"]
TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "spmoprinttemplate.inp")

def check_if_successful(out_path):
    if not os.path.exists(out_path): return False
    try:
        with open(out_path, "r", errors='ignore') as f:
            return "ORCA TERMINATED NORMALLY" in f.read()
    except: return False

def main():
    if not os.path.exists(CALC_BASE_DIR):
        print(f"CRITICAL: Directory not found: {CALC_BASE_DIR}")
        sys.exit(1)

    if not os.path.exists(TEMPLATE_PATH):
        print(f"CRITICAL: Template not found: {TEMPLATE_PATH}")
        sys.exit(1)

    with open(TEMPLATE_PATH, 'r') as f:
        template_content = f.read().strip()

    # Find all base molecule directories
    base_mols = sorted([d for d in os.listdir(CALC_BASE_DIR) if os.path.isdir(os.path.join(CALC_BASE_DIR, d))])

    for mol_name in base_mols:
        sp_dir = os.path.join(CALC_BASE_DIR, mol_name, "sp")
        if not os.path.exists(sp_dir):
            continue

        gbw_path = os.path.join(sp_dir, f"{mol_name}_sp.gbw")
        sp_inp_path = os.path.join(sp_dir, f"{mol_name}_sp.inp")
        
        if not os.path.exists(gbw_path) or not os.path.exists(sp_inp_path):
            print(f"[{mol_name}] Missing _sp.gbw or _sp.inp. Skipping.")
            continue

        out_path = os.path.join(sp_dir, f"{mol_name}_MO.out")
        if check_if_successful(out_path):
            print(f"[{mol_name}] MO print already completed. Skipping.")
            continue

        # 1. Extract the exact * xyz block from the original _sp.inp to guarantee coordinate consistency
        with open(sp_inp_path, 'r') as f:
            sp_inp_lines = f.readlines()
            
        xyz_block = ""
        in_xyz = False
        for line in sp_inp_lines:
            if line.strip().startswith("* xyz"):
                in_xyz = True
            if in_xyz:
                xyz_block += line
                if line.strip() == "*":
                    break

        if not xyz_block:
            print(f"[{mol_name}] Could not find '* xyz' block in {mol_name}_sp.inp. Skipping.")
            continue

        # 2. Build the new sandboxed input file
        moinp_line = f'%moinp "{mol_name}_sp.gbw"\n\n'
        inp_content = f"{template_content}\n\n{moinp_line}{xyz_block}"

        inp_path = os.path.join(sp_dir, f"{mol_name}_MO.inp")
        with open(inp_path, 'w') as f:
            f.write(inp_content)

        print(f"[{mol_name}] Running Zero-Iteration MO Print...")
        
        # 3. Execute ORCA
        with open(out_path, 'w') as out_f:
            subprocess.run([ORCA_CMD, inp_path], stdout=out_f, stderr=subprocess.STDOUT, cwd=sp_dir)
            
        # 4. Cleanup: Delete everything matching _MO.* EXCEPT the .out file
        print(f"[{mol_name}] Cleaning up temporary ORCA files...")
        for temp_file in glob.glob(os.path.join(sp_dir, f"{mol_name}_MO.*")):
            if not temp_file.endswith(".out"):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"  -> [!] Could not remove {temp_file}: {e}")
                    
        print(f"[{mol_name}] Finished MO Print.")

if __name__ == "__main__":
    main()