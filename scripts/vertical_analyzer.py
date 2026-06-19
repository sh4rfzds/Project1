#!/usr/bin/env python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from config import *

# --- Configuration ---
CALC_BASE_DIR = CALC_DIRS["base"]
CALC_DOPED_DIR = CALC_DIRS["vertical"]

OUTPUT_DIR = os.path.join(DATA_ANALYSIS_DIR, "vertical_sp")
AMP_DIR = os.path.join(OUTPUT_DIR, "amp_vs_en")
SIZE_DIR = os.path.join(OUTPUT_DIR, "size_vs_en")
regression_stats = []

def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def read_file_to_ram(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r', errors='ignore') as f:
        return f.readlines()

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
                try:
                    coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    atoms.append((parts[0], coords))
                except ValueError: pass
    return atoms

def find_doped_indices(base_atoms, doped_atoms):
    indices = []
    element_found = None
    for i, (b_type, b_coord) in enumerate(base_atoms):
        if b_type == 'C':
            for d_type, d_coord in doped_atoms:
                if np.linalg.norm(b_coord - d_coord) < 0.2:
                    if d_type in ['N', 'B']:
                        indices.append(i)
                        element_found = d_type
                    break
    return indices, element_found

def get_energies(lines):
    homo_idx = -1
    energies = {}
    reading = False
    for line in lines:
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
    if homo_idx != -1:
        return {"h-1": (homo_idx-1, energies.get(homo_idx-1)),
                "h":   (homo_idx,   energies.get(homo_idx)),
                "l":   (homo_idx+1, energies.get(homo_idx+1)),
                "l+1": (homo_idx+2, energies.get(homo_idx+2))}
    return None

def parse_mo_pz_amplitudes(lines, target_mos):
    pops = {mo: {} for mo in target_mos}
    start_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if "MOLECULAR ORBITALS" in lines[i] or "INITIAL GUESS ORBITALS" in lines[i]:
            start_idx = i; break
            
    if start_idx == -1: return pops

    mo_cols = []
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or "----" in line: continue
        if any(term in line for term in ["MULLIKEN", "LOEWDIN", "MAYER", "DIPOLE", "ORBITAL ENERGIES"]): break
        
        parts = line.split()
        if not parts: continue
        if all(p.isdigit() for p in parts):
            mo_cols = [int(p) for p in parts]
            continue
            
        match = re.search(r'^\s*(\d+)\s*([A-Za-z]+)\s+(\d*[A-Za-z]+)', line)
        if match:
            atom_idx = int(match.group(1))
            elem = match.group(2)
            orb_type = match.group(3)
            if elem == 'C' and 'pz' in orb_type:
                vals = line[match.end():].split()
                for col_i, mo in enumerate(mo_cols):
                    if mo in target_mos and col_i < len(vals):
                        try:
                            coeff = float(vals[col_i])
                            pops[mo][atom_idx] = pops[mo].get(atom_idx, 0.0) + (coeff**2)
                        except ValueError: pass
    return pops

def parse_overlap_file(filepath):
    if not os.path.exists(filepath): return None
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[0] in ORBITALS:
                mapping[parts[0]] = {'match': parts[1], 'overlap': float(parts[2])}
    return mapping

def plot_4x4(fig, axes, data, title, is_boron=False):
    fig.suptitle(title, fontsize=16, fontweight='bold')
    elem_str = "Boron" if is_boron else "Nitrogen"
    
    for i, orb in enumerate(ORBITALS):
        ax = axes[i//2, i%2]
        all_x = []
        all_y = []
        
        for m_idx, base_mol in enumerate(data.keys()):
            x, y = data[base_mol][orb]['x'], data[base_mol][orb]['y']
            if x and len(x) > 2:
                color = COLORS[m_idx % len(COLORS)]
                if len(set(x)) > 1:
                    slope, intercept, r, p, se = linregress(x, y)
                    r2 = r**2
                    preds = np.poly1d([slope, intercept])(x)
                    residuals = np.array(y) - preds
                    mae = np.mean(np.abs(residuals))
                    rmse = np.sqrt(np.mean(residuals**2))
                    
                    x_line = np.linspace(min(x), max(x), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=color, ls='--', lw=1.5, zorder=4)
                    
                    regression_stats.append(f"{elem_str:<9} | {base_mol:<10} | {orb.upper():<4} | R2: {r2:0.4f} | p: {p:8.2e} | MAE: {mae:0.4f} | RMSE: {rmse:0.4f} | y = {slope:0.3f}x + {intercept:0.3f}")
                    label = f"{base_mol} ($R^2={r2:.3f}$)"
                else:
                    label = f"{base_mol} (No fit)"
                
                ax.scatter(x, y, color=color, label=label, s=55, edgecolors='k', alpha=0.7, zorder=3)
                all_x.extend(x)
                all_y.extend(y)
            elif x:
                color = COLORS[m_idx % len(COLORS)]
                ax.scatter(x, y, color=color, label=f"{base_mol} (<3 pts)", s=55, edgecolors='k', alpha=0.7, zorder=3)
                all_x.extend(x)
                all_y.extend(y)
        
        ax.set_title(f"Orbital: {orb.upper()}")
        ax.set_xlabel(r"Normalized Amplitude ($\sum \psi^2 / n$)")
        ax.set_ylabel(r"Normalized $\Delta E$ (eV / $n$)")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        if all_x:
            leg = ax.legend(fontsize=7, facecolor='white', framealpha=0.9, edgecolor='black')
            leg.set_zorder(5)
        else:
            ax.text(0.5, 0.5, "States Destroyed\n($S < 0.90$)", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes, weight='bold')

def main():
    print("=========================================================")
    print("--- Vertical SP Analyzer (Amplitude & Size Scaling) ---")
    print("=========================================================")
    
    os.makedirs(AMP_DIR, exist_ok=True)
    os.makedirs(SIZE_DIR, exist_ok=True)
    
    base_folders = sorted([d for d in os.listdir(CALC_DOPED_DIR) if os.path.isdir(os.path.join(CALC_DOPED_DIR, d))], key=natural_keys)
    master_data = {'N': {}, 'B': {}}
    mol_inv_nc = {}

    for base_mol in base_folders:
        for el in master_data: master_data[el][base_mol] = {o: {'x': [], 'y': [], 'details': []} for o in ORBITALS}
        
        b_out_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_MO.out")
        if not os.path.exists(b_out_path): b_out_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_sp.out")
        b_xyz_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_sp.xyz")
        if not os.path.exists(b_xyz_path): b_xyz_path = os.path.join(PROJECT_DIR, "base_molecules", f"{base_mol}.xyz")
            
        b_lines = read_file_to_ram(b_out_path)
        b_coords = get_xyz_coords(b_xyz_path)
        if not b_lines or not b_coords: continue
        
        num_carbons = sum(1 for atom, _ in b_coords if atom.upper() == 'C')
        mol_inv_nc[base_mol] = 1.0 / num_carbons if num_carbons > 0 else 0

        b_energies = get_energies(b_lines)
        b_pops = parse_mo_pz_amplitudes(b_lines, [v[0] for v in b_energies.values()])

        doped_root = os.path.join(CALC_DOPED_DIR, base_mol)
        for d_mol in sorted(os.listdir(doped_root), key=natural_keys):
            d_sp_path = os.path.join(doped_root, d_mol, "sp")
            d_out = read_file_to_ram(os.path.join(d_sp_path, f"{d_mol}_sp.out"))
            d_ov  = parse_overlap_file(os.path.join(d_sp_path, f"{d_mol}_overlap.txt"))
            d_coords = get_xyz_coords(os.path.join(PROJECT_DIR, "doped_molecules", base_mol, f"{d_mol}.xyz"))
            
            if not check_if_successful(d_out) or not d_ov: continue
            
            d_energies = get_energies(d_out)
            indices, element = find_doped_indices(b_coords, d_coords)
            if not indices or not element: continue

            n_dopants = len(indices)

            for orb in ORBITALS:
                ov_data = d_ov.get(orb)
                if not ov_data or ov_data['overlap'] < OVERLAP_THRESHOLD: continue
                
                b_idx, b_E = b_energies[orb]
                d_match_orb = ov_data['match']
                if d_match_orb not in d_energies: continue
                
                d_E = d_energies[d_match_orb][1]
                
                components = {idx: b_pops[b_idx].get(idx, 0.0) for idx in indices}
                total_amp = sum(components.values())
                norm_amp = total_amp / n_dopants
                norm_dE = (d_E - b_E) / n_dopants
                
                master_data[element][base_mol][orb]['x'].append(norm_amp)
                master_data[element][base_mol][orb]['y'].append(norm_dE)
                master_data[element][base_mol][orb]['details'].append({
                    'd_mol': d_mol, 'n_dopants': n_dopants, 'norm_amp': norm_amp,
                    'components': components, 'norm_dE': norm_dE
                })

    print("[+] Generating Amp vs En SVG plots...")
    fig_n, ax_n = plt.subplots(2, 2, figsize=(14, 10))
    plot_4x4(fig_n, ax_n, master_data['N'], "Nitrogen Doping Correlation (Normalized)")
    fig_n.tight_layout()
    fig_n.savefig(os.path.join(AMP_DIR, "amp_vs_en_N.svg"), format='svg')
    
    fig_b, ax_b = plt.subplots(2, 2, figsize=(14, 10))
    plot_4x4(fig_b, ax_b, master_data['B'], "Boron Doping Correlation (Normalized)", is_boron=True)
    fig_b.tight_layout()
    fig_b.savefig(os.path.join(AMP_DIR, "amp_vs_en_B.svg"), format='svg')

    print("[+] Calculating Amp vs En Errors and Residuals...")
    amp_residuals = {'N': [], 'B': []}
    amp_preds = {'N': [], 'B': []}
    
    # Store fits for R2-weighted LOOCV
    fits = {'N': {o: {} for o in ORBITALS}, 'B': {o: {} for o in ORBITALS}}
    
    for stat in regression_stats:
        parts = [p.strip() for p in stat.split('|')]
        if len(parts) < 6: continue
        el_str = 'B' if 'Boron' in parts[0] else 'N'
        mol = parts[1]
        orb = parts[2].lower()
        eq_idx = -1
        
        # Extract R2
        r2_val = 0.0
        for p in parts:
            if p.startswith('R2:'): r2_val = float(p.replace('R2:', '').strip())
        
        for i, p in enumerate(parts):
            if p.startswith('y ='): eq_idx = i
        if eq_idx == -1: continue
        
        eq = parts[eq_idx].replace("y =", "").strip()
        match = re.search(r'([-.\d]+)x\s*\+\s*([-.\d]+)', eq)
        if match:
            slope, inter = float(match.group(1)), float(match.group(2))
            fits[el_str][orb][mol] = {'slope': slope, 'inter': inter, 'r2': r2_val}
            
            if mol in master_data[el_str]:
                x_data = master_data[el_str][mol][orb]['x']
                y_data = master_data[el_str][mol][orb]['y']
                if len(x_data) > 2:
                    for x_v, y_v in zip(x_data, y_data):
                        pred = slope * x_v + inter
                        amp_preds[el_str].append(pred)
                        amp_residuals[el_str].append(y_v - pred)

    # R2-Weighted Leave-One-Out Cross Validation for Amp vs En
    amp_loocv_lines = []
    loocv_summary_stats = {'N': {orb: [] for orb in ORBITALS}, 'B': {orb: [] for orb in ORBITALS}}
    loocv_parity_data = {orb: [] for orb in ORBITALS}
    
    for el in ['N', 'B']:
        for orb in ORBITALS:
            valid_mols = list(fits[el][orb].keys())
            if len(valid_mols) >= 2:
                for test_mol in valid_mols:
                    train_mols = [m for m in valid_mols if m != test_mol]
                    sum_r2 = sum(fits[el][orb][m]['r2'] for m in train_mols)
                    
                    # 1. R2-Weighted Average for the Slope (Amplitude Dependence)
                    if sum_r2 > 0:
                        w_slope = sum(fits[el][orb][m]['slope'] * fits[el][orb][m]['r2'] for m in train_mols) / sum_r2
                        fallback_inter = sum(fits[el][orb][m]['inter'] * fits[el][orb][m]['r2'] for m in train_mols) / sum_r2
                    else:
                        w_slope = np.mean([fits[el][orb][m]['slope'] for m in train_mols])
                        fallback_inter = np.mean([fits[el][orb][m]['inter'] for m in train_mols])
                        
                    # 2. Linear Regression for the Intercept (Size Dependence)
                    train_inv_nc = [mol_inv_nc[m] for m in train_mols if m in mol_inv_nc and mol_inv_nc[m] > 0]
                    train_inters = [fits[el][orb][m]['inter'] for m in train_mols if m in mol_inv_nc and mol_inv_nc[m] > 0]
                    
                    if len(train_inv_nc) >= 2 and len(set(train_inv_nc)) > 1:
                        slope_size, inter_size, _, _, _ = linregress(train_inv_nc, train_inters)
                        pred_inter = slope_size * mol_inv_nc.get(test_mol, 0) + inter_size
                    else:
                        pred_inter = fallback_inter
                    
                    x_data = master_data[el][test_mol][orb]['x']
                    y_data = master_data[el][test_mol][orb]['y']
                    if not x_data: continue
                    
                    preds = w_slope * np.array(x_data) + pred_inter
                    errors = np.array(y_data) - preds
                    mae = np.mean(np.abs(errors))
                    rmse = np.sqrt(np.mean(errors**2))
                    
                    loocv_summary_stats[el][orb].append({'mol': test_mol, 'mae': mae, 'rmse': rmse})
                    for act, pred in zip(y_data, preds):
                        loocv_parity_data[orb].append({'el': el, 'mol': test_mol, 'act': act, 'pred': pred})
                    
                    eq_str = f"y = {w_slope:0.3f}x + {pred_inter:0.3f}"
                    amp_loocv_lines.append(f"{el:<2} | {orb.upper():<4} | Left out: {test_mol:<15} | {eq_str:<25} | {mae:>8.4f}   | {rmse:>8.4f}")

    print("[+] Generating Amp vs En LOOCV Parity plots...")
    fig_p, ax_p = plt.subplots(2, 2, figsize=(14, 12))
    fig_p.suptitle("Vertical LOOCV Parity: Predicted vs Actual Normalized Shifts", fontsize=16, fontweight='bold')
    
    for i, orb in enumerate(ORBITALS):
        ax = ax_p[i//2, i%2]
        pts = loocv_parity_data[orb]
        if not pts:
            ax.set_title(f"Orbital: {orb.upper()}")
            ax.text(0.5, 0.5, "States Destroyed\n($S < 0.90$)", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes, weight='bold')
            continue
            
        x_vals, y_vals, els = [p['pred'] for p in pts], [p['act'] for p in pts], [p['el'] for p in pts]
        
        for el, color in zip(['N', 'B'], ['#1f77b4', '#d62728']):
            tx = [x for x, e in zip(x_vals, els) if e == el]
            ty = [y for y, e in zip(y_vals, els) if e == el]
            if tx:
                ax.scatter(tx, ty, label=f"{'Nitrogen' if el=='N' else 'Boron'}", color=color, edgecolor='k', s=55, alpha=0.8, zorder=3)
                
        errors = np.array(y_vals) - np.array(x_vals)
        mae = np.mean(np.abs(errors)) if len(errors) > 0 else 0
        
        if x_vals and y_vals:
            min_val, max_val = min(x_vals + y_vals), max(x_vals + y_vals)
            pad = abs(max_val - min_val) * 0.1
            if pad == 0: pad = 0.1
            ax.plot([min_val-pad, max_val+pad], [min_val-pad, max_val+pad], 'k--', zorder=2, label="y = x")
            
        ax.set_title(f"Orbital: {orb.upper()} (Global MAE: {mae:.3f} eV)")
        ax.set_xlabel(r"LOOCV Predicted Normalized $\Delta E$ (eV/n)")
        ax.set_ylabel(r"Actual Normalized $\Delta E$ (eV/n)")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        leg = ax.legend(facecolor='white', framealpha=0.9, edgecolor='black')
        leg.set_zorder(5)
        
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(AMP_DIR, "amp_vs_en_loocv_parity.svg"), format='svg')
    plt.close(fig_p)

    amp_error_lines = []
    fig_res, axes_res = plt.subplots(1, 2, figsize=(14, 6))
    fig_res.suptitle("Amp vs En: Residual Analysis", fontsize=14, fontweight='bold')
    for i, el in enumerate(['N', 'B']):
        if amp_residuals[el]:
            res_arr = np.array(amp_residuals[el])
            mae = np.mean(np.abs(res_arr))
            rmse = np.sqrt(np.mean(res_arr**2))
            err_str = f"[{el} Doping Global] MAE: {mae:.4f} eV | RMSE: {rmse:.4f} eV"
            print(f"  -> {err_str}")
            amp_error_lines.append(err_str)
            
            ax = axes_res[i]
            ax.scatter(amp_preds[el], amp_residuals[el], alpha=0.6, s=55, edgecolors='k', zorder=3)
            ax.axhline(0, color='red', linestyle='--', linewidth=2, zorder=2)
            ax.set_title(f"{'Nitrogen' if el=='N' else 'Boron'} Residuals")
            ax.set_xlabel("Predicted Shift (eV)")
            ax.set_ylabel("Residual Error (eV)")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
            
    fig_res.tight_layout()
    fig_res.savefig(os.path.join(AMP_DIR, "amp_vs_en_residual.svg"), format='svg')
    plt.close(fig_res)

    summary_path = os.path.join(AMP_DIR, "amp_vs_en_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 105 + "\n")
        f.write(f"{'Amplitude vs Energy Summary':^105}\n")
        f.write("=" * 105 + "\n\n")
        
        if amp_error_lines:
            f.write("--- Global Error Metrics ---\n")
            f.write("\n".join(amp_error_lines) + "\n\n")
            
        if amp_loocv_lines:
            f.write("--- Leave-One-Out Cross Validation (Size-Adaptive Intercept Models) ---\n")
            for el in ['N', 'B']:
                for orb in ORBITALS:
                    stats = loocv_summary_stats[el][orb]
                    if stats:
                        avg_mae = np.mean([s['mae'] for s in stats])
                        avg_rmse = np.mean([s['rmse'] for s in stats])
                        f.write(f"[{'Nitrogen' if el=='N' else 'Boron'} - {orb.upper()}] Averaged LOOCV MAE: {avg_mae:.4f} eV | Averaged RMSE: {avg_rmse:.4f} eV\n")
            f.write("\n")
            header = f"{'El':<2} | {'Orb':<4} | {'Left out':<20} | {'Pred Eq (Size Adapted)':<25} | {'Test MAE':<10} | {'Test RMSE'}"
            f.write(header + "\n" + "-" * len(header) + "\n")
            f.write("\n".join(amp_loocv_lines) + "\n\n")

        f.write("--- Linear Regression Models ---\n")
        header = f"{'Dopant':<9} | {'Molecule':<10} | {'Orb':<4} | {'R2 Score':<10} | {'p-value':<10} | {'MAE':<8} | {'RMSE':<8} | {'Equation'}"
        f.write(header + "\n" + "-" * len(header) + "\n")
        for stat in sorted(regression_stats): f.write(stat + "\n")
        
        f.write("\n--- Detailed Data Points ---\n")
        for el in ['N', 'B']:
            for base_mol in sorted(master_data[el].keys(), key=natural_keys):
                for orb in ORBITALS:
                    details = master_data[el][base_mol][orb].get('details', [])
                    if not details: continue
                    f.write(f"\n[{el}] {base_mol} - {orb.upper()}\n")
                    header = f"{'Doped_Mol':<25} | {'N':<2} | {'Norm_Amp (x)':<15} | {'Norm_dE (y)':<12} | {'Components (Index: Amplitude)'}"
                    f.write(header + "\n" + "-" * len(header) + "\n")
                    for d in details:
                        comp_str = ", ".join([f"C{k}: {v:.4f}" for k, v in d['components'].items()])
                        f.write(f"{d['d_mol']:<25} | {d['n_dopants']:<2} | {d['norm_amp']:<15.6f} | {d['norm_dE']:<12.6f} | {comp_str}\n")

    print("[+] Generating Size vs Intercept analysis...")
    size_data = {'B': {orb: [] for orb in ORBITALS}, 'N': {orb: [] for orb in ORBITALS}}
    for stat in regression_stats:
        parts = [p.strip() for p in stat.split('|')]
        if len(parts) < 8: continue
        elem_str = 'B' if 'Boron' in parts[0] else 'N'
        base_mol_name = parts[1]
        orb = parts[2].lower()
        eq = parts[7]
        match = re.search(r'x\s*\+\s*([-\d\.]+)', eq)
        if match and base_mol_name in mol_inv_nc and mol_inv_nc[base_mol_name] > 0:
            intercept = float(match.group(1))
            size_data[elem_str][orb].append((base_mol_name, mol_inv_nc[base_mol_name], intercept))

    fig_size, axes_size = plt.subplots(2, 4, figsize=(24, 12))
    header_size_models = f"{'Dopant':<10} | {'Orb':<5} | {'R2 Score':<10} | {'p-value':<10} | {'MAE':<8} | {'RMSE':<8} | {'Equation'}"
    size_summary_lines = [header_size_models, "-" * len(header_size_models)]
    size_details_lines, size_residuals, size_preds, loocv_lines = [], {'N': [], 'B': []}, {'N': [], 'B': []}, []

    for row, el in enumerate(['B', 'N']):
        for col, orb in enumerate(ORBITALS):
            ax = axes_size[row, col]
            data_points = size_data[el][orb]
            if len(data_points) >= 3: 
                x_vals, y_vals, labels = [d[1] for d in data_points], [d[2] for d in data_points], [d[0] for d in data_points]
                slope, intercept, r_val, p_val, std_err = linregress(x_vals, y_vals)
                r2 = r_val**2
                preds_s = np.poly1d([slope, intercept])(x_vals)
                res_s = np.array(y_vals) - preds_s
                mae_s, rmse_s = np.mean(np.abs(res_s)), np.sqrt(np.mean(res_s**2))

                for x_v, y_v, mol in zip(x_vals, y_vals, labels):
                    pred = slope * x_v + intercept
                    size_preds[el].append(pred)
                    size_residuals[el].append(y_v - pred)

                if len(data_points) >= 4:
                    for i in range(len(data_points)):
                        test_mol, test_x, test_y = data_points[i]
                        train_x = [p[1] for j, p in enumerate(data_points) if j != i]
                        train_y = [p[2] for j, p in enumerate(data_points) if j != i]
                        t_slope, t_inter, _, _, _ = linregress(train_x, train_y)
                        pred_y = t_slope * test_x + t_inter
                        error = test_y - pred_y
                        loocv_lines.append(f"{el:<2} | {orb.upper():<4} | Left out: {test_mol:<10} | Pred: {pred_y:>8.4f} | Act: {test_y:>8.4f} | Err: {error:>8.4f}")

                ax.scatter(x_vals, y_vals, color='black', s=55, edgecolors='k', zorder=3)
                for k, label in enumerate(labels): ax.annotate(label, (x_vals[k], y_vals[k]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=9)
                x_line = np.linspace(min(x_vals)*0.9, max(x_vals)*1.1, 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color='red', linestyle='--', lw=1.5, zorder=4)
                ax.text(0.05, 0.90, f"$R^2 = {r2:.4f}$", transform=ax.transAxes, va='top', fontsize=11, zorder=5, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))
                
                size_summary_lines.append(f"{el:<10} | {orb.upper():<5} | R2: {r2:.4f} | p: {p_val:.2e} | {mae_s:.4f}   | {rmse_s:.4f}   | y = {slope:.3f}x + {intercept:.3f}")
                
                header_raw = f"{'Molecule':<20} | {'N_C':<5} | {'1/N_C (x)':<12} | {'Intercept (y)':<12}"
                size_details_lines.extend([f"\n[{el}] - {orb.upper()}", header_raw, "-" * len(header_raw)])
                for l, x, y in zip(labels, x_vals, y_vals): size_details_lines.append(f"{l:<20} | {int(round(1.0/x)):<5} | {x:<12.6f} | {y:<12.6f}")
            elif len(data_points) > 0:
                x_vals, y_vals, labels = [d[1] for d in data_points], [d[2] for d in data_points], [d[0] for d in data_points]
                ax.scatter(x_vals, y_vals, color='black', s=55, edgecolors='k', zorder=3)
                for k, label in enumerate(labels): ax.annotate(label, (x_vals[k], y_vals[k]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=9)
            else:
                ax.text(0.5, 0.5, "States Destroyed\n($S < 0.90$)", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes, weight='bold')

            ax.set_title(f"{'Boron' if el=='B' else 'Nitrogen'} Doping - {orb.upper()}")
            ax.set_xlabel("1 / N_C")
            ax.set_ylabel("Intercept (eV)")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
            
    plt.tight_layout()
    fig_size.savefig(os.path.join(SIZE_DIR, "size_vs_en.svg"), format='svg')
    plt.close(fig_size)

    size_error_lines = []
    fig_s_res, axes_s_res = plt.subplots(1, 2, figsize=(14, 6))
    fig_s_res.suptitle("Size vs Intercept: Residual Analysis", fontsize=14, fontweight='bold')
    for i, el in enumerate(['N', 'B']):
        if size_residuals[el]:
            res_arr = np.array(size_residuals[el])
            mae = np.mean(np.abs(res_arr))
            rmse = np.sqrt(np.mean(res_arr**2))
            err_str = f"[Size {el} Global] MAE: {mae:.4f} eV | RMSE: {rmse:.4f} eV"
            print(f"  -> {err_str}")
            size_error_lines.append(err_str)
            
            ax = axes_s_res[i]
            ax.scatter(size_preds[el], size_residuals[el], alpha=0.6, s=55, edgecolors='k', zorder=3)
            ax.axhline(0, color='red', linestyle='--', linewidth=2, zorder=2)
            ax.set_title(f"{'Nitrogen' if el=='N' else 'Boron'} Size Residuals")
            ax.set_xlabel("Predicted Intercept (eV)")
            ax.set_ylabel("Residual Error (eV)")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
            
    fig_s_res.tight_layout()
    fig_s_res.savefig(os.path.join(SIZE_DIR, "size_vs_en_residual.svg"), format='svg')
    plt.close(fig_s_res)

    with open(os.path.join(SIZE_DIR, "size_vs_en_summary.txt"), "w") as f:
        f.write("=" * 105 + "\n")
        f.write(f"{'Size vs Intercept Calibration Summary':^105}\n")
        f.write("=" * 105 + "\n\n")
        if size_error_lines: f.write("--- Global Error Metrics ---\n" + "\n".join(size_error_lines) + "\n\n")
        if loocv_lines: f.write("--- Leave-One-Out Cross Validation ---\n" + "\n".join(loocv_lines) + "\n\n")
        f.write("--- Linear Regression Models ---\n" + "\n".join(size_summary_lines) + "\n\n")
        f.write("--- Raw Data ---\n" + "\n".join(size_details_lines) + "\n")

    print(f"\n[+] Vertical Analysis complete! Check '{OUTPUT_DIR}' for the final plots and summary.")

if __name__ == "__main__":
    main()