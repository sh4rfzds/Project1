#!/usr/bin/env python3

import os
import math
import itertools
from config import *

# --- Configuration ---
OPT_XYZ_PATH = os.path.join(CALC_DIRS["base"], "CirCor", "opt", "CirCor_opt.xyz")
OUTPUT_DIR = os.path.join(STRESS_MOLS_DIR, "CirCor")

def read_xyz(filepath):
    atoms = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append((parts[0], [float(parts[1]), float(parts[2]), float(parts[3])]))
    return atoms

def write_xyz(filepath, atoms):
    with open(filepath, 'w') as f:
        f.write(f"{len(atoms)}\nGenerated Haywire Combination\n")
        for atom, coords in atoms:
            f.write(f"{atom:2} {coords[0]:15.6f} {coords[1]:15.6f} {coords[2]:15.6f}\n")

def distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def get_carbon_center(atoms):
    """Calculates center of geometry strictly using Carbon atoms."""
    c_coords = [c for a, c in atoms if a == 'C']
    if not c_coords: return [0.0, 0.0, 0.0]
    cx = sum(c[0] for c in c_coords) / len(c_coords)
    cy = sum(c[1] for c in c_coords) / len(c_coords)
    cz = sum(c[2] for c in c_coords) / len(c_coords)
    return [cx, cy, cz]

def main():
    if not os.path.exists(OPT_XYZ_PATH):
        print(f"[!] Error: Could not find pristine optimized geometry at {OPT_XYZ_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    atoms = read_xyz(OPT_XYZ_PATH)
    
    # Center the molecule using ONLY the carbon lattice
    cog = get_carbon_center(atoms)
    centered = [[c[0]-cog[0], c[1]-cog[1], c[2]-cog[2]] for _, c in atoms]

    # 1. Map Edge Carbons to their Hydrogens
    edge_carbons = {} 
    for i, (atom_i, coords_i) in enumerate(atoms):
        if atom_i == 'C':
            for j, (atom_j, coords_j) in enumerate(atoms):
                if atom_j == 'H' and distance(coords_i, coords_j) < CH_BOND_MAX:
                    edge_carbons[i] = j
                    break 

    # 2. Group Symmetrical Sites (Assuming full X and Y symmetry for CirCor)
    unique_groups = []
    visited_carbons = set()

    for c_idx in edge_carbons.keys():
        if c_idx in visited_carbons: continue
        cx, cy, _ = centered[c_idx]
        
        # Reflection targets for X and Y symmetry
        targets = [(cx, cy), (cx, -cy), (-cx, cy), (-cx, -cy)]
        
        current_group = set()
        for tx, ty in targets:
            best_dist = SYM_TOLERANCE
            best_idx = -1
            for search_idx in edge_carbons.keys():
                dist = distance([tx, ty, 0], [centered[search_idx][0], centered[search_idx][1], 0])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = search_idx
            
            if best_idx != -1:
                current_group.add(best_idx)
                visited_carbons.add(best_idx)
        
        if current_group:
            unique_groups.append(list(current_group))

    # 3. Sort Groups Clockwise
    def group_angle(group):
        min_angle = 2 * math.pi
        for idx in group:
            x, y, _ = centered[idx]
            angle = math.atan2(x, y)
            if angle < 0: angle += 2 * math.pi 
            if angle < min_angle: min_angle = angle
        return min_angle
    
    unique_groups.sort(key=group_angle)
    
    print(f"--- CirCor Haywire Generator ---")
    print(f"Discovered {len(unique_groups)} topological sites.")
    
    # 4. Generate all possible combinations (C = un-doped, N = Nitrogen, B = Boron)
    options = ['C', 'N', 'B']
    all_combos = list(itertools.product(options, repeat=len(unique_groups)))
    
    count = 0
    for combo in all_combos:
        # Skip the pristine 'C, C, C, C, C' combination
        if all(dopant == 'C' for dopant in combo):
            continue
            
        # Build the filename (e.g., CirCor_P1N_P3B_P5N.xyz)
        name_parts = []
        for p_idx, dopant in enumerate(combo, start=1):
            if dopant != 'C':
                name_parts.append(f"P{p_idx}{dopant}")
        
        out_name = f"CirCor_{'_'.join(name_parts)}.xyz"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # Collect all hydrogens that need to be deleted
        h_to_delete = set()
        for p_idx, dopant in enumerate(combo):
            if dopant != 'C':
                group = unique_groups[p_idx]
                h_to_delete.update([edge_carbons[c] for c in group])
        
        # Build the new atom list
        new_atoms = []
        for idx, (atom_type, coords) in enumerate(atoms):
            if idx in h_to_delete:
                continue # Strip Hydrogen
            
            # Check if this atom belongs to a group that is being doped
            swapped = False
            for p_idx, dopant in enumerate(combo):
                if dopant != 'C' and idx in unique_groups[p_idx]:
                    new_atoms.append((dopant, coords))
                    swapped = True
                    break
            
            if not swapped:
                new_atoms.append((atom_type, coords))
                
        write_xyz(out_path, new_atoms)
        count += 1

    print(f"\nSuccessfully brute-forced {count} molecules into {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()