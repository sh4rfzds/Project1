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
CALC_ADIA_DIR = CALC_DIRS["adiabatic"]
OUTPUT_DIR = os.path.join(DATA_ANALYSIS_DIR, "adiabatic_sp")

def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def read_file_to_ram(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r', errors='ignore') as f: return f.readlines()

def get_inp_xyz(filepath):
    if not os.path.exists(filepath): return []
    atoms = []
    with open(filepath, 'r') as f:
        in_xyz = False
        for line in f:
            stripped = line.strip()
            if stripped.startswith('* xyz') or stripped.startswith('*xyz'):
                in_xyz = True; continue
            if in_xyz:
                if stripped == '*' or stripped == '': break
                parts = stripped.split()
                if len(parts) >= 4:
                    try: atoms.append((parts[0], np.array([float(parts[1]), float(parts[2]), float(parts[3])])))
                    except ValueError: pass
    return atoms

def get_xyz_coords(filepath):
    if not os.path.exists(filepath): return []
    atoms = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if len(lines) < 3: return []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                try: atoms.append((parts[0], np.array([float(parts[1]), float(parts[2]), float(parts[3])])))
                except ValueError: pass
    return atoms

def kabsch_rmsd(coords1, coords2):
    if not coords1 or not coords2 or len(coords1) != len(coords2): return None
    P, Q = np.array([c[1] for c in coords1]), np.array([c[1] for c in coords2])
    Pc, Qc = P - np.mean(P, axis=0), Q - np.mean(Q, axis=0)
    H = np.dot(Pc.T, Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    Qc_rot = np.dot(Qc, R.T)
    return np.sqrt(np.mean(np.sum((Pc - Qc_rot)**2, axis=1)))

def get_energies(lines):
    if not lines: return None
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
        return {
            "h-1": (homo_idx-1, energies.get(homo_idx-1)),
            "h":   (homo_idx,   energies.get(homo_idx)),
            "l":   (homo_idx+1, energies.get(homo_idx+1)),
            "l+1": (homo_idx+2, energies.get(homo_idx+2))
        }
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

def find_all_doped_indices(base_atoms, doped_atoms):
    indices = []
    for i, (b_type, b_coord) in enumerate(base_atoms):
        if b_type == 'C':
            for d_type, d_coord in doped_atoms:
                if np.linalg.norm(b_coord - d_coord) < 0.2:
                    if d_type in ['N', 'B']: indices.append((i, d_type))
                    break
    return indices

def plot_4x4(fig, axes, data, title, is_boron=False):
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, orb in enumerate(ORBITALS):
        ax = axes[i//2, i%2]
        all_x, all_y = [], []

        for m_idx, base_mol in enumerate(data.keys()):
            x, y = data[base_mol][orb]['x'], data[base_mol][orb]['y']
            if x:
                color = COLORS[m_idx % len(COLORS)]
                ax.scatter(x, y, color=color, label=f"{base_mol}", s=55, edgecolors='k', alpha=0.7, zorder=3)
                all_x.extend(x)
                all_y.extend(y)
        
        ax.set_title(f"Orbital: {orb.upper()}")
        ax.set_xlabel("RMSD (\u00c5)")
        ax.set_ylabel(r"Relaxation $\Delta E$ (eV)")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        if all_x:
            ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)
            leg = ax.legend(fontsize=7, facecolor='white', framealpha=0.9, edgecolor='black')
            leg.set_zorder(5)
        else:
            ax.text(0.5, 0.5, "States Destroyed\n($S < 0.90$)", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes, weight='bold')

def main():
    print("=========================================================")
    print("--- Adiabatic SP Analyzer (RMSD & LOOCV Parity) ---")
    print("=========================================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(CALC_ADIA_DIR): return

    base_folders = [d for d in os.listdir(CALC_ADIA_DIR) if os.path.isdir(os.path.join(CALC_ADIA_DIR, d))]
    master_data = {'N': {}, 'B': {}}
    parity_data = {orb: [] for orb in ORBITALS}
    loocv_eqs = {'N': {o: {} for o in ORBITALS}, 'B': {o: {} for o in ORBITALS}}

    # 1. Parse Vertical Summary for LOOCV Pipeline
    fits = {'N': {o: {} for o in ORBITALS}, 'B': {o: {} for o in ORBITALS}}
    unique_mols = set()
    vert_summary_path = os.path.join(PROJECT_DIR, "data_analysis", "vertical_sp", "amp_vs_en", "amp_vs_en_summary.txt")
    
    if os.path.exists(vert_summary_path):
        with open(vert_summary_path, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 8 and "y =" in parts[-1]:
                    d_type = 'B' if 'Boron' in parts[0] else 'N'
                    mol_name = parts[1]
                    orb = parts[2].lower()
                    unique_mols.add(mol_name)
                    r2_val = 0.0
                    for p in parts:
                        if p.startswith('R2:'): r2_val = float(p.replace('R2:', '').strip())
                    eq_str = parts[-1].replace("y =", "").strip()
                    match = re.search(r'([-.\d]+)x\s*\+\s*([-.\d]+)', eq_str)
                    if match:
                        fits[d_type][orb][mol_name] = {'slope': float(match.group(1)), 'inter': float(match.group(2)), 'r2': r2_val}

    # 2. Calculate 1/N_c for unique molecules
    mol_inv_nc = {}
    for m in unique_mols:
        m_xyz = os.path.join(CALC_BASE_DIR, m, "sp", f"{m}_sp.xyz")
        if not os.path.exists(m_xyz): m_xyz = os.path.join(PROJECT_DIR, "base_molecules", f"{m}.xyz")
        coords = get_xyz_coords(m_xyz)
        if coords:
            nc = sum(1 for atom, _ in coords if atom.upper() == 'C')
            if nc > 0: mol_inv_nc[m] = 1.0 / nc

    # 3. Main processing loop
    for base_mol in base_folders:
        for el in master_data: master_data[el][base_mol] = {o: {'x': [], 'y': [], 'details': []} for o in ORBITALS}
        
        # Load Base Molecule Data
        b_out_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_MO.out")
        if not os.path.exists(b_out_path): b_out_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_sp.out")
        b_xyz_path = os.path.join(CALC_BASE_DIR, base_mol, "sp", f"{base_mol}_sp.xyz")
        if not os.path.exists(b_xyz_path): b_xyz_path = os.path.join(PROJECT_DIR, "base_molecules", f"{base_mol}.xyz")
            
        b_lines = read_file_to_ram(b_out_path)
        b_coords = get_xyz_coords(b_xyz_path)
        if not b_lines or not b_coords: continue
        
        b_energies = get_energies(b_lines)
        b_pops = parse_mo_pz_amplitudes(b_lines, [v[0] for v in b_energies.values() if v])

        # Generate LOOCV Equations for this base_mol
        eqs = {'N': {}, 'B': {}}
        for el in ['N', 'B']:
            for orb in ORBITALS:
                valid_mols = list(fits[el][orb].keys())
                train_mols = [m for m in valid_mols if m != base_mol]
                
                fallback_used = False
                if len(train_mols) == 0 and len(valid_mols) > 0:
                    train_mols = valid_mols
                    fallback_used = True
                
                if len(train_mols) >= 1:
                    sum_r2 = sum(fits[el][orb][m]['r2'] for m in train_mols)
                    if sum_r2 > 0:
                        w_slope = sum(fits[el][orb][m]['slope'] * fits[el][orb][m]['r2'] for m in train_mols) / sum_r2
                    else:
                        w_slope = np.mean([fits[el][orb][m]['slope'] for m in train_mols])
                    
                    train_inv_nc = [mol_inv_nc[m] for m in train_mols if m in mol_inv_nc]
                    train_inters = [fits[el][orb][m]['inter'] for m in train_mols if m in mol_inv_nc]
                    
                    if len(train_inv_nc) >= 2 and len(set(train_inv_nc)) > 1:
                        slope_size, inter_size, _, _, _ = linregress(train_inv_nc, train_inters)
                        pred_inter = slope_size * mol_inv_nc.get(base_mol, 0) + inter_size
                    else:
                        pred_inter = np.mean(train_inters) if train_inters else 0.0
                        
                    eqs[el][orb] = (w_slope, pred_inter)
                    
                    log_text = f"Left out: {base_mol}" if not fallback_used else f"Self-Val: {base_mol}"
                    loocv_eqs[el][orb][base_mol] = (log_text, f"y = {w_slope:0.3f}x + {pred_inter:0.3f}")
                else:
                    eqs[el][orb] = (None, None)

        doped_root = os.path.join(CALC_ADIA_DIR, base_mol)
        for d_mol in sorted(os.listdir(doped_root), key=natural_keys):
            dopant_type = d_mol.split('_')[-1][-1]
            if dopant_type not in ['B', 'N']: continue

            adia_opt_dir, adia_sp_dir, vert_sp_dir = os.path.join(doped_root, d_mol, "opt"), os.path.join(doped_root, d_mol, "sp"), os.path.join(CALC_VERT_DIR, base_mol, d_mol, "sp")
            inp_path, xyz_path = os.path.join(adia_opt_dir, f"{d_mol}_opt.inp"), os.path.join(adia_opt_dir, f"{d_mol}_opt.xyz")
            adia_out, vert_out, vert_ov_path = os.path.join(adia_sp_dir, f"{d_mol}_sp.out"), os.path.join(vert_sp_dir, f"{d_mol}_sp.out"), os.path.join(vert_sp_dir, f"{d_mol}_overlap.txt")
            
            if not os.path.exists(xyz_path) or not os.path.exists(adia_out) or not os.path.exists(vert_out): continue

            coords_initial, coords_final = get_inp_xyz(inp_path), get_xyz_coords(xyz_path)
            rmsd = kabsch_rmsd(coords_initial, coords_final)
            if rmsd is None: continue
            
            vert_lines, adia_lines = read_file_to_ram(vert_out), read_file_to_ram(adia_out)
            vert_energies, adia_energies, vert_ov = get_energies(vert_lines), get_energies(adia_lines), parse_overlap_file(vert_ov_path)
            if not vert_energies or not adia_energies or not vert_ov: continue

            for orb in ORBITALS:
                ov_data = vert_ov.get(orb)
                
                match_label = orb
                if ov_data:
                    match_label = ov_data['match'] if ov_data else orb
                
                overlap_val = ov_data['overlap'] if ov_data else 0.0
                if overlap_val < OVERLAP_THRESHOLD: continue
                
                v_E = vert_energies.get(match_label, [None, None])[1]
                a_E = adia_energies.get(match_label, [None, None])[1]
                
                if v_E is not None and a_E is not None:
                    dE_relax = a_E - v_E
                    master_data[dopant_type][base_mol][orb]['x'].append(rmsd)
                    master_data[dopant_type][base_mol][orb]['y'].append(dE_relax)
                    master_data[dopant_type][base_mol][orb]['details'].append({'d_mol': d_mol, 'rmsd': rmsd, 'dE_relax': dE_relax, 'v_E': v_E, 'a_E': a_E, 'overlap': overlap_val})

                # --- LOOCV Pipeline Prediction ---
                b_idx, b_E = b_energies.get(orb, [None, None])
                if b_E is None or a_E is None: continue
                
                doped_indices = find_all_doped_indices(b_coords, coords_initial) 
                if not doped_indices: continue
                
                w_slope, pred_inter = eqs[dopant_type].get(orb, (None, None))
                if w_slope is None: continue
                
                pred_tot_eq = 0.0
                for idx, d_type in doped_indices:
                    amp = b_pops[b_idx].get(idx, 0.0)
                    pred_tot_eq += (w_slope * amp + pred_inter)
                    
                n_dopants = len(doped_indices)
                pred_norm_eq = pred_tot_eq / n_dopants
                act_norm = (a_E - b_E) / n_dopants
                
                parity_data[orb].append({
                    'base_mol': base_mol,
                    'd_mol': d_mol, 'el': dopant_type,
                    'pred_norm_eq': pred_norm_eq, 'act_norm': act_norm, 'error': act_norm - pred_norm_eq
                })

    print("[+] Generating RMSD vs Relaxation Energy SVG plots...")
    fig_n, ax_n = plt.subplots(2, 2, figsize=(14, 10))
    plot_4x4(fig_n, ax_n, master_data['N'], "Nitrogen Doping: Structural vs Electronic Relaxation")
    fig_n.tight_layout()
    fig_n.savefig(os.path.join(OUTPUT_DIR, "rmsd_vs_en_N.svg"), format='svg')
    
    fig_b, ax_b = plt.subplots(2, 2, figsize=(14, 10))
    plot_4x4(fig_b, ax_b, master_data['B'], "Boron Doping: Structural vs Electronic Relaxation", is_boron=True)
    fig_b.tight_layout()
    fig_b.savefig(os.path.join(OUTPUT_DIR, "rmsd_vs_en_B.svg"), format='svg')

    print("[+] Generating LOOCV Parity y=x plots...")
    fig_p, ax_p = plt.subplots(2, 2, figsize=(14, 12))
    fig_p.suptitle("Adiabatic LOOCV Parity: Predicted vs Actual Normalized Shifts", fontsize=16, fontweight='bold')
    
    for i, orb in enumerate(ORBITALS):
        ax = ax_p[i//2, i%2]
        pts = parity_data[orb]
        if not pts: 
            ax.set_title(f"Orbital: {orb.upper()}")
            ax.text(0.5, 0.5, "States Destroyed\n($S < 0.90$)", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes, weight='bold')
            continue

        x_vals, y_vals, els = [p['pred_norm_eq'] for p in pts], [p['act_norm'] for p in pts], [p['el'] for p in pts]
        for el, color in zip(['N', 'B'], ['#1f77b4', '#d62728']):
            tx, ty = [x for x, e in zip(x_vals, els) if e == el], [y for y, e in zip(y_vals, els) if e == el]
            if tx: ax.scatter(tx, ty, label=f"{'Nitrogen' if el=='N' else 'Boron'}", color=color, edgecolor='k', s=55, alpha=0.8, zorder=3)
        
        errors = np.array(y_vals) - np.array(x_vals)
        mae, rmse = np.mean(np.abs(errors)) if len(errors) > 0 else 0, np.sqrt(np.mean(errors**2)) if len(errors) > 0 else 0

        if x_vals and y_vals:
            min_val, max_val = min(x_vals + y_vals), max(x_vals + y_vals)
            pad = abs(max_val - min_val) * 0.1
            if pad == 0: pad = 0.1
            ax.plot([min_val-pad, max_val+pad], [min_val-pad, max_val+pad], 'k--', zorder=2, label="y = x")
            
        ax.set_title(f"Orbital: {orb.upper()} (Global MAE: {mae:.3f} eV)")
        ax.set_xlabel(r"LOOCV Predicted Normalized $\Delta E_{adia}$ (eV/n)")
        ax.set_ylabel(r"Actual Normalized $\Delta E_{adia}$ (eV/n)")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        leg = ax.legend(facecolor='white', framealpha=0.9, edgecolor='black')
        leg.set_zorder(5)
        
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(OUTPUT_DIR, "adiabatic_loocv_parity.svg"), format='svg')

    # --- Write Unified Summary Text ---
    with open(os.path.join(OUTPUT_DIR, "adiabatic_summary.txt"), 'w') as f:
        f.write("=" * 105 + "\n")
        f.write(f"{'Adiabatic Analysis Summary':^105}\n")
        f.write("=" * 105 + "\n\n")
        
        # 1. Average RMSD
        global_rmsd = {'N': {}, 'B': {}}
        for el in ['N', 'B']:
            for base_mol in master_data[el]:
                for orb in ORBITALS:
                    for d in master_data[el][base_mol][orb].get('details', []):
                        global_rmsd[el][d['d_mol']] = d['rmsd']
        
        f.write("--- Global Structural Relaxation (RMSD) ---\n")
        for el in ['N', 'B']:
            rmsds = list(global_rmsd[el].values())
            if rmsds:
                f.write(f"[{'Nitrogen' if el=='N' else 'Boron'} Doping Global] Average RMSD: {np.mean(rmsds):.4f} \u00c5\n")
        f.write("\n")
        
        # 2. Global Parity
        f.write("--- Global Parity Error Metrics ---\n")
        for el in ['N', 'B']:
            el_errors = [p['error'] for orb in ORBITALS for p in parity_data[orb] if p['el'] == el]
            if el_errors:
                mae = np.mean(np.abs(el_errors))
                rmse = np.sqrt(np.mean(np.array(el_errors)**2))
                f.write(f"[{'Nitrogen' if el=='N' else 'Boron'} Doping Global] MAE: {mae:.4f} eV | RMSE: {rmse:.4f} eV\n")
        f.write("\n")
        
        # 3. LOOCV Averages and Specifics
        f.write("--- Leave-One-Out Cross Validation (Parity) ---\n")
        for el in ['N', 'B']:
            for orb in ORBITALS:
                pts = [p for p in parity_data[orb] if p['el'] == el]
                if pts:
                    errors = np.array([p['error'] for p in pts])
                    avg_mae = np.mean(np.abs(errors))
                    avg_rmse = np.sqrt(np.mean(errors**2))
                    f.write(f"[{'Nitrogen' if el=='N' else 'Boron'} - {orb.upper()}] Averaged LOOCV Parity MAE: {avg_mae:.4f} eV | Averaged RMSE: {avg_rmse:.4f} eV\n")
        f.write("\n")
        
        header_loocv = f"{'El':<2} | {'Orb':<4} | {'Left out':<20} | {'Pred Eq (Size Adapted)':<25} | {'Test MAE':<10} | {'Test RMSE'}"
        f.write(header_loocv + "\n" + "-" * len(header_loocv) + "\n")
        
        adia_loocv_lines = []
        for el in ['N', 'B']:
            for orb in ORBITALS:
                pts = [p for p in parity_data[orb] if p['el'] == el]
                mols = sorted(list(set([p['base_mol'] for p in pts])), key=natural_keys)
                for m in mols:
                    m_pts = [p for p in pts if p['base_mol'] == m]
                    m_err = np.array([p['error'] for p in m_pts])
                    mae = np.mean(np.abs(m_err))
                    rmse = np.sqrt(np.mean(m_err**2))
                    
                    log_text, eq_str = loocv_eqs[el][orb].get(m, (f"Left out: {m}", "N/A"))
                    adia_loocv_lines.append(f"{el:<2} | {orb.upper():<4} | {log_text:<20} | {eq_str:<25} | {mae:>8.4f}   | {rmse:>8.4f}")
        
        f.write("\n".join(adia_loocv_lines) + "\n\n")
        
        # 4. Detailed Data
        f.write("--- Detailed LOOCV Parity Data Points ---\n")
        for orb in ORBITALS:
            if not parity_data[orb]: continue
            f.write(f"\n[{orb.upper()}]\n")
            header_pts = f"{'Base_Mol':<15} | {'Combo_Mol':<20} | {'El':<2} | {'Pred_dE_norm':<15} | {'Act_dE_norm':<15} | {'Error_norm'}"
            f.write(header_pts + "\n" + "-" * len(header_pts) + "\n")
            for p in sorted(parity_data[orb], key=lambda k: (k['base_mol'], k['d_mol'])):
                f.write(f"{p['base_mol']:<15} | {p['d_mol']:<20} | {p['el']:<2} | {p['pred_norm_eq']:<15.6f} | {p['act_norm']:<15.6f} | {p['error']:<10.6f}\n")
        
        f.write("\n--- Detailed Adiabatic Relaxation Data Points (RMSD vs dE_relax) ---\n")
        for el in ['N', 'B']:
            for base_mol in sorted(master_data[el].keys(), key=natural_keys):
                for orb in ORBITALS:
                    details = master_data[el][base_mol][orb].get('details', [])
                    if not details: continue
                    f.write(f"\n[{el}] {base_mol} - {orb.upper()}   (Avg RMSD: {np.mean([d['rmsd'] for d in details]):.6f} \u00c5 | Avg dE_relax: {np.mean([d['dE_relax'] for d in details]):.6f} eV | Avg Abs dE_relax: {np.mean([abs(d['dE_relax']) for d in details]):.6f} eV)\n")
                    header_relax = f"{'Doped_Mol':<25} | {'RMSD (x)':<15} | {'dE_relax (y)':<12} | {'Vert_E':<10} | {'Adia_E':<10} | {'Vert_Overlap'}"
                    f.write(header_relax + "\n" + "-" * len(header_relax) + "\n")
                    for d in details: f.write(f"{d['d_mol']:<25} | {d['rmsd']:<15.6f} | {d['dE_relax']:<12.6f} | {d['v_E']:<10.4f} | {d['a_E']:<10.4f} | {d['overlap']:.4f}\n")

    print(f"\n[+] Adiabatic Analysis complete! Check '{OUTPUT_DIR}' for the final plots and summary.")

if __name__ == "__main__":
    main()