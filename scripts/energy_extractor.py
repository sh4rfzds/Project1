#!/usr/bin/env python3

import os
import sys
import re
from config import *

# --- Configuration ---
CALC_BASE_DIR = CALC_DIRS["base"]
CALC_VERT_DIR = CALC_DIRS["vertical"]

def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def get_total_energy(filepath):
    """Parses the FINAL SINGLE POINT ENERGY (Eh) from an ORCA output file."""
    if not os.path.exists(filepath): return None
    try:
        with open(filepath, 'r', errors='ignore') as f:
            for line in reversed(f.readlines()):
                if "FINAL SINGLE POINT ENERGY" in line:
                    return float(line.split()[-1])
    except: pass
    return None

def get_energies(filepath):
    """Parses frontier orbital energies (eV) from an ORCA output file."""
    if not os.path.exists(filepath): return None
    homo_idx = -1
    energies = {}
    reading = False
    try:
        with open(filepath, 'r', errors='replace') as f:
            for line in f:
                if "ORBITAL ENERGIES" in line:
                    reading = True; continue
                if reading:
                    if not line.strip() or "----" in line:
                        if energies: break
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            idx, occ, ev = int(parts[0]), float(parts[1]), float(parts[3])
                            energies[idx] = ev
                            if occ > 0: homo_idx = idx
                        except ValueError: pass
    except: return None
    
    if homo_idx != -1:
        return {
            "h-1": energies.get(homo_idx - 1),
            "h":   energies.get(homo_idx),
            "l":   energies.get(homo_idx + 1),
            "l+1": energies.get(homo_idx + 2)
        }
    return None

def get_overlap_mapping(filepath):
    """Reads the _overlap.txt file to determine orbital tracking."""
    if not os.path.exists(filepath): return None
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] in ORBITALS:
                mapping[parts[0]] = parts[1]
    return mapping

def main():
    print("=========================================================")
    print("--- Matched Energy Extractor (Vertical SP) ---")
    print("=========================================================")
    
    if not os.path.exists(CALC_VERT_DIR):
        print(f"CRITICAL: Could not find {CALC_VERT_DIR}")
        sys.exit(1)

    base_mols = sorted([d for d in os.listdir(CALC_VERT_DIR) if os.path.isdir(os.path.join(CALC_VERT_DIR, d))], key=natural_keys)

    for base_mol in base_mols:
        print(f"\n[+] Processing Base Molecule: {base_mol}")
        
        # 1. Get Base Molecule Energies
        base_sp_dir = os.path.join(CALC_BASE_DIR, base_mol, "sp")
        base_out_file = os.path.join(base_sp_dir, f"{base_mol}_sp.out")
        
        base_tot_E = get_total_energy(base_out_file)
        base_frontiers = get_energies(base_out_file)
        
        if base_tot_E is None or base_frontiers is None:
            print(f"  -> [!] Could not parse base energies for {base_mol}. Skipping.")
            continue
            
        base_gap = base_frontiers['l'] - base_frontiers['h']
        
        # Prepare Master Summary List
        master_data = []

        base_doped_dir = os.path.join(CALC_VERT_DIR, base_mol)
        doped_mols = sorted([d for d in os.listdir(base_doped_dir) if os.path.isdir(os.path.join(base_doped_dir, d))], key=natural_keys)
        
        for d_mol in doped_mols:
            calc_dir = os.path.join(base_doped_dir, d_mol, "sp")
            if not os.path.exists(calc_dir): calc_dir = os.path.join(base_doped_dir, d_mol)
            
            out_file = os.path.join(calc_dir, f"{d_mol}_sp.out")
            if not os.path.exists(out_file): out_file = os.path.join(calc_dir, f"{d_mol}.out")
            overlap_file = os.path.join(calc_dir, f"{d_mol}_overlap.txt")
            
            tot_E = get_total_energy(out_file)
            frontiers = get_energies(out_file)
            mapping = get_overlap_mapping(overlap_file)
            
            if tot_E is None or frontiers is None or mapping is None:
                continue # Calculation likely unfinished or unanalyzed
                
            standard_gap = frontiers['l'] - frontiers['h']
            
            # --- Write Individual Doped Summary ---
            d_summary_path = os.path.join(calc_dir, f"{d_mol}_energies.txt")
            with open(d_summary_path, 'w') as f:
                f.write(f"Energy Summary for: {d_mol}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Total Single Point Energy : {tot_E:15.6f} Eh\n")
                f.write(f"Standard HOMO-LUMO Gap    : {standard_gap:15.6f} eV\n\n")
                f.write("--- Frontier Orbital Tracking (Matched via Overlap) ---\n")
                f.write(f"{'Base_Orb':<10} | {'Base_E (eV)':<12} | {'Doped_Orb':<10} | {'Doped_E (eV)':<12} | {'Shift (eV)':<10}\n")
                f.write("-" * 65 + "\n")
                
                tracked_energies = {}
                for b_orb in ORBITALS:
                    d_orb = mapping[b_orb]
                    d_E = frontiers.get(d_orb, 0.0)
                    b_E = base_frontiers.get(b_orb, 0.0)
                    shift = d_E - b_E
                    tracked_energies[b_orb] = (d_orb, d_E)
                    f.write(f"{b_orb:<10} | {b_E:<12.5f} | {d_orb:<10} | {d_E:<12.5f} | {shift:<10.5f}\n")
            
            # Append to master data
            master_data.append({
                'name': d_mol,
                'tot_E': tot_E,
                'gap': standard_gap,
                'tracked': tracked_energies
            })
            
        # --- Write Master Summary for Base Molecule ---
        if master_data:
            master_summary_path = os.path.join(base_doped_dir, f"{base_mol}_energy_summary.txt")
            with open(master_summary_path, 'w') as f:
                f.write(f"=========================================================================================================================\n")
                f.write(f"                               Master Energy Summary: {base_mol} (Vertical SP)                                  \n")
                f.write(f"=========================================================================================================================\n\n")
                
                f.write(f"Base Total Energy: {base_tot_E:.6f} Eh\n")
                f.write(f"Base HOMO-LUMO Gap: {base_gap:.5f} eV\n\n")
                
                # Header
                header = f"{'Molecule':<25} | {'Total_E (Eh)':<15} | {'Gap (eV)':<10} | "
                for orb in ORBITALS:
                    header += f"{orb+'_match':<10} | {orb+'_E(eV)':<12} | "
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                
                # Write Base Row for reference
                base_row = f"{base_mol+' (Base)':<25} | {base_tot_E:<15.6f} | {base_gap:<10.5f} | "
                for orb in ORBITALS:
                    base_row += f"{orb:<10} | {base_frontiers[orb]:<12.5f} | "
                f.write(base_row + "\n")
                
                # Write Doped Rows
                for data in master_data:
                    row = f"{data['name']:<25} | {data['tot_E']:<15.6f} | {data['gap']:<10.5f} | "
                    for orb in ORBITALS:
                        d_orb, d_E = data['tracked'][orb]
                        row += f"{d_orb:<10} | {d_E:<12.5f} | "
                    f.write(row + "\n")
            
            print(f"  -> Generated {len(master_data)} individual summaries.")
            print(f"  -> Master summary saved to {master_summary_path}")

if __name__ == "__main__":
    main()