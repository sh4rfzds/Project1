#!/usr/bin/env python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from config import *

# --- Configuration ---
CALC_BASE_DIR = CALC_DIRS["base"]
CALC_VERT_DIR = CALC_DIRS["vertical"]
CALC_ADD_DIR = CALC_DIRS["additivity"]
OUTPUT_DIR = os.path.join(DATA_ANALYSIS_DIR, "additivity_sp")

regression_stats = []

def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def read_file_to_ram(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r', errors='ignore') as f: return f.readlines()

def check_if_successful(lines):
    if not lines: return False
    for line in reversed(lines[-100:]): 
        if "ORCA TERMINATED NORMALLY" in line: return True
    return False

def get_xyz_coords(filepath):
    if not os.path.exists(filepath): return []
    atoms = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                try: atoms.append((parts[0], np.array([float(parts[1]), float(parts[2]), float(parts[3])])))
                except ValueError: pass
    return atoms

def find_all_doped_indices(base_atoms, doped_atoms):
    indices = []
    for i, (b_type, b_coord) in enumerate(base_atoms):
        if b_type == 'C':
            for d_type, d_coord in doped_atoms:
                if np.linalg.norm(b_coord - d_coord) < 0.2:
                    if d_type in ['N', 'B']: indices.append((i, d_type))
                    break
    return indices

def get_energies(lines):
    homo_idx, energies, reading = -1, {}, False
    for line in lines:
        if "ORBITAL ENERGIES" in line: reading = True; continue
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
    if homo_idx != -1:
        return {"h-1": (homo_idx-1, energies.get(homo_idx-1)), "h": (homo_idx, energies.get(homo_idx)), "l": (homo_idx+1, energies.get(homo_idx+1)), "l+1": (homo_idx+2, energies.get(homo_idx+2))}
    return None

def parse_mo_pz_amplitudes(lines, target_mos):
    pops = {mo: {} for mo in target_mos}
    start_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if "MOLECULAR ORBITALS" in lines[i] or "INITIAL GUESS ORBITALS" in lines[i]: start_idx = i; break
    if start_idx == -1: return pops
    mo_cols = []
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or "----" in line: continue
        if any(term in line for term in ["MULLIKEN", "LOEWDIN", "MAYER", "DIPOLE", "ORBITAL ENERGIES"]): break
        parts = line.split()
        if not parts: continue
        if all(p.isdigit() for p in parts): mo_cols = [int(p) for p in parts]; continue
        match = re.search(r'^\s*(\d+)\s*([A-Za-z]+)\s+(\d*[A-Za-z]+)', line)
        if match:
            atom_idx, elem, orb_type = int(match.group(1)), match.group(2), match.group(3)
            if elem == 'C' and 'pz' in orb_type:
                vals = line[match.end():].split()
                for col_i, mo in enumerate(mo_cols):
                    if mo in target_mos and col_i < len(vals):
                        try: pops[mo][atom_idx] = pops[mo].get(atom_idx, 0.0) + (float(vals[col_i])**2)
                        except ValueError: pass
    return pops

def parse_overlap_file(filepath):
    if not os.path.exists(filepath): return None
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[0] in ORBITALS: mapping[parts[0]] = {'match': parts[1], 'overlap': float(parts[2])}
    return mapping

def main():
    print("=========================================================")
    print("--- Additivity Analyzer (LOOCV Static Analysis) ---")
    print("=========================================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(CALC_ADD_DIR): return
    base_folders = sorted([d for d in os.listdir(CALC_ADD_DIR) if os.path.isdir(os.path.join(CALC_ADD_DIR, d))], key=natural_keys)
    if not base_folders: return

    # 1. Parse Vertical Summary & Get Unique Molecules
    fits = {'N': {o: {} for o in ORBITALS}, 'B': {o: {} for o in ORBITALS}}
    unique_mols = set()
    vert_summary_path = os.path.join(PROJECT_DIR, "data_analysis", "vertical_sp", "amp_vs_en", "amp_vs_en_summary.txt")
    
    if os.path.exists(vert_summary_path):
        with open(vert_summary_path, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 8 and "y =" in parts[7]:
                    d_type = 'B' if 'Boron' in parts[0] else 'N'
                    mol_name = parts[1]
                    orb = parts[2].lower()
                    unique_mols.add(mol_name)
                    
                    r2_val = 0.0
                    for p in parts:
                        if p.startswith('R2:'): r2_val = float(p.replace('R2:', '').strip())
                        
                    eq_str = parts[7].replace("y =", "").strip()
                    match = re.search(r'([-.\d]+)x\s*\+\s*([-.\d]+)', eq_str)
                    if match:
                        fits[d_type][orb][mol_name] = {'slope': float(match.group(1)), 'inter': float(match.group(2)), 'r2': r2_val}
    else:
        print(f"Warning: Could not find {vert_summary_path}. Run vertical_analyzer.py first.")

    # Calculate 1/N_c for unique molecules
    mol_inv_nc = {}
    for m in unique_mols:
        m_xyz = os.path.join(CALC_BASE_DIR, m, "sp", f"{m}_sp.xyz")
        if not os.path.exists(m_xyz):
            m_xyz = os.path.join(PROJECT_DIR, "base_molecules", f"{m}.xyz")
        m_coords = get_xyz_coords(m_xyz)
        if m_coords:
            num_carbons = sum(1 for atom, _ in m_coords if atom.upper() == 'C')
            if num_carbons > 0:
                mol_inv_nc[m] = 1.0 / num_carbons

    parity_data = {orb: [] for orb in ORBITALS}
    amp_data = {'N+N': {orb: {} for orb in ORBITALS}, 'B+B': {orb: {} for orb in ORBITALS}}
    loocv_lines = []

    # 2. Iterate through base molecules dynamically
    for base_mol in base_folders:
        combo_dir = os.path.join(CALC_ADD_DIR, base_mol)
        
        b_out_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_MO.out")
        if not os.path.exists(b_out_path): b_out_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_sp.out")
        b_xyz_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_sp.xyz")
        if not os.path.exists(b_xyz_path): b_xyz_path = os.path.join(PROJECT_DIR, "base_molecules", f"{base_mol}.xyz")

        b_lines, b_coords = read_file_to_ram(b_out_path), get_xyz_coords(b_xyz_path)
        if not b_lines or not b_coords: continue
            
        base_energies = get_energies(b_lines)
        b_pops = parse_mo_pz_amplitudes(b_lines, [v[0] for v in base_energies.values() if v])

        # Construct LOOCV Equations strictly for this target base_mol
        print(f"[+] Generating LOOCV Size-Adaptive Equations for {base_mol}...")
        eqs = {'N': {}, 'B': {}}
        for el in ['N', 'B']:
            for orb in ORBITALS:
                valid_mols = list(fits[el][orb].keys())
                train_mols = [m for m in valid_mols if m != base_mol] 
                
                if len(train_mols) > 0:
                    sum_r2 = sum(fits[el][orb][m]['r2'] for m in train_mols)
                    
                    if sum_r2 > 0:
                        w_slope = sum(fits[el][orb][m]['slope'] * fits[el][orb][m]['r2'] for m in train_mols) / sum_r2
                        fallback_inter = sum(fits[el][orb][m]['inter'] * fits[el][orb][m]['r2'] for m in train_mols) / sum_r2
                    else:
                        w_slope = np.mean([fits[el][orb][m]['slope'] for m in train_mols])
                        fallback_inter = np.mean([fits[el][orb][m]['inter'] for m in train_mols])
                    
                    train_inv_nc = [mol_inv_nc[m] for m in train_mols if m in mol_inv_nc]
                    train_inters = [fits[el][orb][m]['inter'] for m in train_mols if m in mol_inv_nc]
                    
                    if len(train_inv_nc) >= 2 and len(set(train_inv_nc)) > 1:
                        slope_size, inter_size, _, _, _ = linregress(train_inv_nc, train_inters)
                        pred_inter = slope_size * mol_inv_nc.get(base_mol, 0) + inter_size
                    else:
                        pred_inter = fallback_inter
                    
                    eqs[el][orb] = (w_slope, pred_inter)
                    loocv_lines.append(f"{el:<2} | {orb.upper():<4} | Left out: {base_mol:<10} | y = {w_slope:0.3f}x + {pred_inter:0.3f}")
                else:
                    eqs[el][orb] = (None, None)

        # Iterate through Additivity Combinations for this base_mol
        for d_mol in sorted(os.listdir(combo_dir), key=natural_keys):
            parts = d_mol.split('_')
            if len(parts) != 3: continue 
            
            site1, site2 = parts[1], parts[2]
            type1, type2 = site1[-1], site2[-1]
            combo_type = 'N+N' if type1 == 'N' and type2 == 'N' else ('B+B' if type1 == 'B' and type2 == 'B' else 'B+N')

            c_sp_path = os.path.join(combo_dir, d_mol, "sp")
            c_out_lines = read_file_to_ram(os.path.join(c_sp_path, f"{d_mol}_sp.out"))
            c_ov = parse_overlap_file(os.path.join(c_sp_path, f"{d_mol}_overlap.txt"))
            c_xyz_path = os.path.join(c_sp_path, f"{d_mol}_sp.xyz")
            if not os.path.exists(c_xyz_path): c_xyz_path = os.path.join(PROJECT_DIR, "double_doped_molecules", base_mol, f"{d_mol}.xyz")
            c_coords = get_xyz_coords(c_xyz_path)

            s1_out_lines = read_file_to_ram(os.path.join(CALC_VERT_DIR, base_mol, f"{base_mol}_{site1}", "sp", f"{base_mol}_{site1}_sp.out"))
            s1_ov = parse_overlap_file(os.path.join(CALC_VERT_DIR, base_mol, f"{base_mol}_{site1}", "sp", f"{base_mol}_{site1}_overlap.txt"))
            s2_out_lines = read_file_to_ram(os.path.join(CALC_VERT_DIR, base_mol, f"{base_mol}_{site2}", "sp", f"{base_mol}_{site2}_sp.out"))
            s2_ov = parse_overlap_file(os.path.join(CALC_VERT_DIR, base_mol, f"{base_mol}_{site2}", "sp", f"{base_mol}_{site2}_overlap.txt"))

            if not (c_out_lines and s1_out_lines and s2_out_lines and c_coords and c_ov and s1_ov and s2_ov): continue
            if not (check_if_successful(c_out_lines) and check_if_successful(s1_out_lines) and check_if_successful(s2_out_lines)): continue

            c_energies, s1_energies, s2_energies = get_energies(c_out_lines), get_energies(s1_out_lines), get_energies(s2_out_lines)
            doped_indices = find_all_doped_indices(b_coords, c_coords)
            if not doped_indices: continue
            n_total = len(doped_indices)

            for orb in ORBITALS:
                b_idx, b_E = base_energies.get(orb, [None, None])
                if b_E is None: continue

                c_ov_data, s1_ov_data, s2_ov_data = c_ov.get(orb), s1_ov.get(orb), s2_ov.get(orb)
                if not c_ov_data or c_ov_data['overlap'] < OVERLAP_THRESHOLD: continue
                if not s1_ov_data or s1_ov_data['overlap'] < OVERLAP_THRESHOLD: continue
                if not s2_ov_data or s2_ov_data['overlap'] < OVERLAP_THRESHOLD: continue

                c_match, s1_match, s2_match = c_ov_data['match'], s1_ov_data['match'], s2_ov_data['match']
                if c_match not in c_energies or s1_match not in s1_energies or s2_match not in s2_energies: continue

                c_E, s1_E, s2_E = c_energies[c_match][1], s1_energies[s1_match][1], s2_energies[s2_match][1]
                
                # Direct Calculation Parity
                dE_pred, dE_act = (s1_E - b_E) + (s2_E - b_E), c_E - b_E
                pred_norm, act_norm = dE_pred / n_total, dE_act / n_total
                
                # LOOCV Equation-Based Parity
                pred_tot_eq = 0.0
                valid_eq = True
                for idx, d_type in doped_indices:
                    slope, inter = eqs[d_type].get(orb, (None, None))
                    if slope is None:
                        valid_eq = False
                        break
                    amp = b_pops[b_idx].get(idx, 0.0)
                    pred_tot_eq += (slope * amp + inter)
                    
                pred_norm_eq = pred_tot_eq / n_total if valid_eq else None
                
                parity_data[orb].append({
                    'base_mol': base_mol,
                    'combo': d_mol, 'type': combo_type, 'n_tot': n_total, 
                    'pred_norm': pred_norm, 'act_norm': act_norm, 'error_norm': act_norm - pred_norm, 
                    'pred_tot': dE_pred, 'act_tot': dE_act,
                    'pred_norm_eq': pred_norm_eq, 'error_norm_eq': act_norm - pred_norm_eq if pred_norm_eq is not None else None,
                    'pred_tot_eq': pred_tot_eq if valid_eq else None
                })

                if combo_type in ['N+N', 'B+B']:
                    if base_mol not in amp_data[combo_type][orb]:
                        amp_data[combo_type][orb][base_mol] = {'x': [], 'y': []}
                    norm_amp = sum(b_pops[b_idx].get(idx, 0.0) for idx, _ in doped_indices) / n_total
                    amp_data[combo_type][orb][base_mol]['x'].append(norm_amp)
                    amp_data[combo_type][orb][base_mol]['y'].append(act_norm)

    # --- PLOT 1: DIRECT PARITY ---
    print("[+] Generating Direct Parity y=x plots...")
    fig_p, ax_p = plt.subplots(2, 2, figsize=(14, 12))
    fig_p.suptitle(f"Additivity Parity: Directly Calculated vs Actual Normalized Shifts", fontsize=16, fontweight='bold')
    
    global_parity_errors = []
    for i, orb in enumerate(ORBITALS):
        ax = ax_p[i//2, i%2]
        pts = parity_data[orb]
        if not pts: continue

        x_vals, y_vals, types = [p['pred_norm'] for p in pts], [p['act_norm'] for p in pts], [p['type'] for p in pts]
        for t in ['N+N', 'B+B', 'B+N']:
            tx, ty = [x for x, typ in zip(x_vals, types) if typ == t], [y for y, typ in zip(y_vals, types) if typ == t]
            if tx: ax.scatter(tx, ty, label=t, color=COLORS_MULTI[t], edgecolor='k', s=55, alpha=0.8, zorder=3)
        
        errors = np.array(y_vals) - np.array(x_vals)
        mae, rmse = np.mean(np.abs(errors)) if len(errors) > 0 else 0, np.sqrt(np.mean(errors**2)) if len(errors) > 0 else 0
        global_parity_errors.append(f"{orb.upper():<4} | Direct Parity MAE: {mae:.4f} eV | Direct Parity RMSE: {rmse:.4f} eV")

        if x_vals and y_vals:
            min_val, max_val = min(x_vals + y_vals), max(x_vals + y_vals)
            pad = abs(max_val - min_val) * 0.1
            if pad == 0: pad = 0.1
            ax.plot([min_val-pad, max_val+pad], [min_val-pad, max_val+pad], 'k--', zorder=2, label="y = x")
            
        ax.set_title(f"Orbital: {orb.upper()} (MAE: {mae:.3f} eV)")
        ax.set_xlabel(r"Directly Calculated Normalized $\Delta E$ (eV/n)")
        ax.set_ylabel(r"Actual Normalized $\Delta E$ (eV/n)")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        leg = ax.legend(facecolor='white', framealpha=0.9, edgecolor='black')
        leg.set_zorder(5)
        
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(OUTPUT_DIR, "additivity_parity.svg"), format='svg')

    # --- PLOT 2: EQUATION PARITY (LOOCV) ---
    print("[+] Generating LOOCV Equation-Predicted Parity y=x plots...")
    fig_pe, ax_pe = plt.subplots(2, 2, figsize=(14, 12))
    fig_pe.suptitle(f"Additivity Parity: LOOCV-Predicted vs Actual Normalized Shifts", fontsize=16, fontweight='bold')
    
    global_eq_parity_errors = []
    for i, orb in enumerate(ORBITALS):
        ax = ax_pe[i//2, i%2]
        pts = [p for p in parity_data[orb] if p['pred_norm_eq'] is not None]
        if not pts: 
            ax.set_title(f"Orbital: {orb.upper()}")
            ax.text(0.5, 0.5, "States Destroyed\n($S < 0.90$)", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes, weight='bold')
            continue

        x_vals, y_vals, types = [p['pred_norm_eq'] for p in pts], [p['act_norm'] for p in pts], [p['type'] for p in pts]
        for t in ['N+N', 'B+B', 'B+N']:
            tx, ty = [x for x, typ in zip(x_vals, types) if typ == t], [y for y, typ in zip(y_vals, types) if typ == t]
            if tx: ax.scatter(tx, ty, label=t, color=COLORS_MULTI[t], edgecolor='k', s=55, alpha=0.8, zorder=3)
        
        errors = np.array(y_vals) - np.array(x_vals)
        mae, rmse = np.mean(np.abs(errors)) if len(errors) > 0 else 0, np.sqrt(np.mean(errors**2)) if len(errors) > 0 else 0
        global_eq_parity_errors.append(f"{orb.upper():<4} | LOOCV-Parity MAE: {mae:.4f} eV | LOOCV-Parity RMSE: {rmse:.4f} eV")

        if x_vals and y_vals:
            min_val, max_val = min(x_vals + y_vals), max(x_vals + y_vals)
            pad = abs(max_val - min_val) * 0.1
            if pad == 0: pad = 0.1
            ax.plot([min_val-pad, max_val+pad], [min_val-pad, max_val+pad], 'k--', zorder=2, label="y = x")
            
        ax.set_title(f"Orbital: {orb.upper()} (MAE: {mae:.3f} eV)")
        ax.set_xlabel(r"LOOCV Predicted Normalized $\Delta E$ (eV/n)")
        ax.set_ylabel(r"Actual Normalized $\Delta E$ (eV/n)")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        leg = ax.legend(facecolor='white', framealpha=0.9, edgecolor='black')
        leg.set_zorder(5)
        
    fig_pe.tight_layout()
    fig_pe.savefig(os.path.join(OUTPUT_DIR, "additivity_parity_loocv.svg"), format='svg')

    # --- PLOT 3: AMPLITUDE VS ENERGY ---
    print("[+] Generating Additivity Amplitude vs Energy plots...")
    fig_a, ax_a = plt.subplots(2, 2, figsize=(14, 12))
    fig_a.suptitle(f"Additivity Amplitude vs Energy (N+N and B+B Only)", fontsize=16, fontweight='bold')
    
    COLORS_MOL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, orb in enumerate(ORBITALS):
        ax = ax_a[i//2, i%2]
        all_x = []
        
        # Extract unique base_mols across both N+N and B+B to maintain consistent colors
        base_mols_plot = set()
        for ct in ['N+N', 'B+B']:
            base_mols_plot.update(amp_data[ct][orb].keys())
        sorted_mols = sorted(list(base_mols_plot), key=natural_keys)
        
        for combo_type in ['N+N', 'B+B']:
            marker = 'o' if combo_type == 'N+N' else 's'
            ls = '--' if combo_type == 'N+N' else ':'
            
            for base_mol in sorted(amp_data[combo_type][orb].keys(), key=natural_keys):
                c = COLORS_MOL[sorted_mols.index(base_mol) % len(COLORS_MOL)]
                x = amp_data[combo_type][orb][base_mol]['x']
                y = amp_data[combo_type][orb][base_mol]['y']
                
                if len(x) > 2:
                    slope, intercept, r, p, se = linregress(x, y)
                    preds = np.poly1d([slope, intercept])(x)
                    res = np.array(y) - preds
                    mae, rmse, r2 = np.mean(np.abs(res)), np.sqrt(np.mean(res**2)), r**2
                    
                    x_line = np.linspace(min(x), max(x), 100)
                    preds_line = slope * x_line + intercept
                    ax.plot(x_line, preds_line, color=c, ls=ls, lw=1.5, zorder=4)
                    ax.scatter(x, y, color=c, marker=marker, label=f"{combo_type} {base_mol} ($R^2={r2:.3f}$)", s=55, edgecolors='k', zorder=3)
                    regression_stats.append(f"{combo_type:<5} | {base_mol:<15} | {orb.upper():<4} | R2: {r2:0.4f} | p: {p:8.2e} | MAE: {mae:0.4f} | RMSE: {rmse:0.4f} | y = {slope:0.3f}x + {intercept:0.3f}")
                    all_x.extend(x)
                elif x:
                    ax.scatter(x, y, color=c, marker=marker, label=f"{combo_type} {base_mol} (<3 pts)", s=55, edgecolors='k', zorder=3)
                    all_x.extend(x)
                
        if all_x:
            ax.set_title(f"Orbital: {orb.upper()}")
            ax.set_xlabel(r"Combined Normalized Amplitude ($\sum \psi^2 / n_{tot}$)")
            ax.set_ylabel(r"Actual Normalized $\Delta E$ (eV / $n_{tot}$)")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
            leg = ax.legend(fontsize=7, facecolor='white', framealpha=0.9, edgecolor='black')
            leg.set_zorder(5)
        else:
            ax.set_title(f"Orbital: {orb.upper()}")
            ax.text(0.5, 0.5, "States Destroyed\n($S < 0.90$)", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes, weight='bold')

    fig_a.tight_layout()
    fig_a.savefig(os.path.join(OUTPUT_DIR, "additivity_amp_vs_en.svg"), format='svg')

    # --- SUMMARY TEXT ---
    with open(os.path.join(OUTPUT_DIR, "additivity_summary.txt"), "w") as f:
        f.write("=" * 105 + "\n")
        f.write(f"{'Additivity Analysis Summary':^105}\n")
        f.write("=" * 105 + "\n\n")
        
        # 1. Global Averages
        f.write("--- Global Averages (All Combinations) ---\n")
        for orb in ORBITALS:
            dir_errs = [p['error_norm'] for p in parity_data[orb] if p['error_norm'] is not None]
            eq_errs = [p['error_norm_eq'] for p in parity_data[orb] if p['error_norm_eq'] is not None]
            if dir_errs:
                dir_mae = np.mean(np.abs(dir_errs))
                dir_rmse = np.sqrt(np.mean(np.array(dir_errs)**2))
                eq_str = ""
                if eq_errs:
                    eq_mae = np.mean(np.abs(eq_errs))
                    eq_rmse = np.sqrt(np.mean(np.array(eq_errs)**2))
                    eq_str = f" | LOOCV MAE: {eq_mae:.4f} eV | LOOCV RMSE: {eq_rmse:.4f} eV"
                f.write(f"[{orb.upper():<4}] Direct MAE: {dir_mae:.4f} eV | Direct RMSE: {dir_rmse:.4f} eV{eq_str}\n")
        f.write("\n")

        # 2. Specific Molecule Averages
        f.write("--- Specific Molecule Averages (All Groups) ---\n")
        for orb in ORBITALS:
            pts = parity_data[orb]
            if not pts: continue
            f.write(f"\n[{orb.upper()}]\n")
            mols = sorted(list(set(p['base_mol'] for p in pts)), key=natural_keys)
            for m in mols:
                m_pts = [p for p in pts if p['base_mol'] == m]
                m_dir = [p['error_norm'] for p in m_pts if p['error_norm'] is not None]
                m_eq = [p['error_norm_eq'] for p in m_pts if p['error_norm_eq'] is not None]
                
                dir_mae = f"{np.mean(np.abs(m_dir)):.4f}" if m_dir else "N/A"
                eq_mae = f"{np.mean(np.abs(m_eq)):.4f}" if m_eq else "N/A"
                f.write(f"  -> {m:<15} | Direct MAE: {dir_mae:>6} eV | LOOCV MAE: {eq_mae:>6} eV\n")
        f.write("\n")

        # 3. Group Averages
        f.write("--- Group Averages (N+N, B+B, B+N) ---\n")
        for t in ['N+N', 'B+B', 'B+N']:
            for orb in ORBITALS:
                t_pts = [p for p in parity_data[orb] if p['type'] == t]
                if not t_pts: continue
                
                dir_errs = [p['error_norm'] for p in t_pts if p['error_norm'] is not None]
                eq_errs = [p['error_norm_eq'] for p in t_pts if p['error_norm_eq'] is not None]
                
                dir_mae = f"{np.mean(np.abs(dir_errs)):.4f}" if dir_errs else "N/A"
                eq_mae = f"{np.mean(np.abs(eq_errs)):.4f}" if eq_errs else "N/A"
                
                f.write(f"\n[{t} Doping - {orb.upper()}] Global Direct MAE: {dir_mae:>6} eV | Global LOOCV MAE: {eq_mae:>6} eV\n")
                
                mols = sorted(list(set(p['base_mol'] for p in t_pts)), key=natural_keys)
                for m in mols:
                    m_pts = [p for p in t_pts if p['base_mol'] == m]
                    m_dir = [p['error_norm'] for p in m_pts if p['error_norm'] is not None]
                    m_eq = [p['error_norm_eq'] for p in m_pts if p['error_norm_eq'] is not None]
                    
                    m_dir_mae = f"{np.mean(np.abs(m_dir)):.4f}" if m_dir else "N/A"
                    m_eq_mae = f"{np.mean(np.abs(m_eq)):.4f}" if m_eq else "N/A"
                    
                    f.write(f"       -> {m:<15} | Direct MAE: {m_dir_mae:>6} eV | LOOCV MAE: {m_eq_mae:>6} eV\n")
        f.write("\n")

        # 4. LOOCV Equations
        f.write("--- Leave-One-Out Equations Used (Trained on N-1 non-Base molecules) ---\n")
        header_loocv = f"{'El':<2} | {'Orb':<4} | {'Left out':<20} | {'Pred Eq (Size Adapted)':<25}"
        f.write(header_loocv + "\n" + "-" * len(header_loocv) + "\n")
        f.write("\n".join(list(dict.fromkeys(loocv_lines))) + "\n\n")

        # 5. Linear Regressions
        f.write("--- Independent Amplitude Linear Regression (N+N and B+B Only) ---\n")
        header_reg = f"{'Type':<5} | {'Base_Mol':<15} | {'Orb':<4} | {'R2 Score':<10} | {'p-value':<10} | {'MAE':<8} | {'RMSE':<8} | {'Equation'}"
        f.write(header_reg + "\n" + "-" * len(header_reg) + "\n")
        for stat in sorted(regression_stats): f.write(stat + "\n")
        f.write("\n")
        
        # 6. Detailed Points
        f.write("--- Detailed Parity Data Points ---\n")
        for orb in ORBITALS:
            if not parity_data[orb]: continue
            f.write(f"\n[{orb.upper()}]\n")
            header_data = f"{'Base_Mol':<15} | {'Combo_Mol':<20} | {'Type':<5} | {'N_tot':<5} | {'DirPred_dE_n':<13} | {'EqPred_dE_n':<13} | {'Act_dE_n':<13} | {'DirErr_n':<10} | {'EqErr_n':<10} | {'DirPred_Tot':<12} | {'EqPred_Tot':<12} | {'Act_Tot'}"
            f.write(header_data + "\n" + "-" * len(header_data) + "\n")
            for p in sorted(parity_data[orb], key=lambda k: (k['base_mol'], k['type'], k['combo'])):
                eq_pred_n = p['pred_norm_eq'] if p['pred_norm_eq'] is not None else float('nan')
                eq_err_n = p['error_norm_eq'] if p['error_norm_eq'] is not None else float('nan')
                eq_pred_t = p['pred_tot_eq'] if p['pred_tot_eq'] is not None else float('nan')
                f.write(f"{p['base_mol']:<15} | {p['combo']:<20} | {p['type']:<5} | {p['n_tot']:<5} | {p['pred_norm']:<13.6f} | {eq_pred_n:<13.6f} | {p['act_norm']:<13.6f} | {p['error_norm']:<10.6f} | {eq_err_n:<10.6f} | {p['pred_tot']:<12.6f} | {eq_pred_t:<12.6f} | {p['act_tot']:.6f}\n")

    print(f"\n[+] Analysis complete! Check '{OUTPUT_DIR}' for the final plots and summary.")

if __name__ == "__main__":
    main()