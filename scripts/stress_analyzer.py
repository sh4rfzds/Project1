#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from config import *

# --- Configuration ---
INPUT_FILE = os.path.join(CALC_DIRS["stress"], "CirCor", "CirCor_overlaps.txt")
OUTPUT_DIR = os.path.join(DATA_ANALYSIS_DIR, "stress_sp")

def config_sort_key(c):
    parts = c.split('+')
    b = int(parts[0].replace('B', '').strip())
    n = int(parts[1].replace('N', '').strip())
    return (b + n, b)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"CRITICAL: Could not find overlap data at {INPUT_FILE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=========================================================")
    print("--- Stress SP Analyzer (Categorical Overlap Plots) ---")
    print("=========================================================")

    data = {orb: {} for orb in ORBITALS}

    with open(INPUT_FILE, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 9 and "Molecule" not in parts[0]:
                mol_name = parts[0]
                b_count = mol_name.count('B')
                n_count = mol_name.count('N')
                config = f"{b_count}B + {n_count}N"
                try:
                    overlaps = {
                        "h-1": float(parts[2]),
                        "h":   float(parts[4]),
                        "l":   float(parts[6]),
                        "l+1": float(parts[8])
                    }
                    for orb in ORBITALS:
                        if config not in data[orb]: data[orb][config] = []
                        data[orb][config].append(overlaps[orb])
                except ValueError: continue

    all_configs = set()
    for orb in ORBITALS: all_configs.update(data[orb].keys())
    sorted_configs = sorted(list(all_configs), key=config_sort_key)

    print(f"[+] Found {len(sorted_configs)} unique dopant configurations.")
    print("[+] Generating overlap clustering plot (.svg)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Orbital Overlap by Dopant Configuration", fontsize=16, fontweight='bold')

    cmap = plt.get_cmap('tab20')
    colors = {cfg: cmap(i % 20) for i, cfg in enumerate(sorted_configs)}

    for idx, orb in enumerate(ORBITALS):
        ax = axes[idx // 2, idx % 2]
        x_ticks, x_labels = [], []
        
        for x_pos, cfg in enumerate(sorted_configs):
            y_vals = data[orb].get(cfg, [])
            if not y_vals: continue
            
            x_jitter = np.random.normal(x_pos, 0.08, size=len(y_vals))
            ax.scatter(x_jitter, y_vals, color=colors[cfg], alpha=0.8, 
                       edgecolors='k', s=55, zorder=3)
            
            x_ticks.append(x_pos)
            x_labels.append(cfg)

        ax.set_title(f"Orbital: {orb.upper()}")
        ax.set_ylabel("Overlap (S)")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        
        ax.axhline(0.90, color='r', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2, label="S=0.90")
        ax.axhline(0.80, color='darkorange', linestyle=':', linewidth=1.5, alpha=0.8, zorder=2, label="S=0.80")
        ax.axhline(0.60, color='gold', linestyle='-.', linewidth=1.5, alpha=0.8, zorder=2, label="S=0.60")
        
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        leg = ax.legend(loc='lower left', facecolor='white', framealpha=0.9, edgecolor='black')
        leg.set_zorder(5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15)
    plt.savefig(os.path.join(OUTPUT_DIR, "overlap_clustering.svg"), format='svg')
    plt.close()

    print("[+] Generating summary text file...")
    with open(os.path.join(OUTPUT_DIR, "stress_summary.txt"), 'w') as f:
        f.write("=" * 105 + "\n")
        f.write(f"{'Stress Test Overlap Summary':^105}\n")
        f.write("=" * 105 + "\n\n")
        
        n_configs = [c for c in sorted_configs if c.split('+')[0].strip() == '0B']
        b_configs = [c for c in sorted_configs if c.split('+')[1].strip() == '0N']
        m_configs = [c for c in sorted_configs if c.split('+')[0].strip() != '0B' and c.split('+')[1].strip() != '0N']

        def write_global_metrics(title, configs_to_include):
            if not configs_to_include: return
            f.write(f"--- Global Survival Metrics ({title}) ---\n")
            header = f"{'Orbital':<10} | {'Total':<6} | {'S >= 0.90':<15} | {'S >= 0.80':<15} | {'S >= 0.60':<15}"
            f.write(header + "\n" + "-" * len(header) + "\n")
            for orb in ORBITALS:
                total_orb = sum(len(data[orb].get(c, [])) for c in configs_to_include)
                s90 = sum(sum(1 for v in data[orb].get(c, []) if v >= 0.90) for c in configs_to_include)
                s80 = sum(sum(1 for v in data[orb].get(c, []) if v >= 0.80) for c in configs_to_include)
                s60 = sum(sum(1 for v in data[orb].get(c, []) if v >= 0.60) for c in configs_to_include)
                
                r90 = (s90 / total_orb * 100) if total_orb > 0 else 0
                r80 = (s80 / total_orb * 100) if total_orb > 0 else 0
                r60 = (s60 / total_orb * 100) if total_orb > 0 else 0
                f.write(f"{orb.upper():<10} | {total_orb:<6} | {r90:>6.2f}% ({s90:<3}) | {r80:>6.2f}% ({s80:<3}) | {r60:>6.2f}% ({s60:<3})\n")
            f.write("\n")

        write_global_metrics("Pure Nitrogen", n_configs)
        write_global_metrics("Pure Boron", b_configs)
        write_global_metrics("Mixed Boron+Nitrogen", m_configs)
        
        f.write("--- Detailed Cluster Breakdown (By Dopant Configuration) ---\n")
        header_cluster = f"{'Config':<12} | {'Orbital':<8} | {'Total':<6} | {'S >= 0.90':<15} | {'S >= 0.80':<15} | {'S >= 0.60':<15}"
        f.write(header_cluster + "\n" + "-" * len(header_cluster) + "\n")
        for cfg in sorted_configs:
            for orb in ORBITALS:
                vals = data[orb].get(cfg, [])
                if not vals: continue
                total = len(vals)
                
                s90 = sum(1 for v in vals if v >= 0.90)
                s80 = sum(1 for v in vals if v >= 0.80)
                s60 = sum(1 for v in vals if v >= 0.60)
                
                r90 = (s90 / total * 100) if total > 0 else 0
                r80 = (s80 / total * 100) if total > 0 else 0
                r60 = (s60 / total * 100) if total > 0 else 0
                f.write(f"{cfg:<12} | {orb.upper():<8} | {total:<6} | {r90:>6.2f}% ({s90:<3}) | {r80:>6.2f}% ({s80:<3}) | {r60:>6.2f}% ({s60:<3})\n")
            f.write("-" * len(header_cluster) + "\n")

    print(f"[+] Success! Plots and summary saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()