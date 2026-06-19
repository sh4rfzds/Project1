#!/usr/bin/env python3

import os

# =====================================================================
# --- SYSTEM CONFIGURATION ---
# =====================================================================
# Change these to match your system before running any calculations.

PROJECT_DIR = os.path.expanduser("~/Project2")
ORCA_CMD = os.path.expanduser("~/orca_6_1_0/orca")
ORCA_PLOT_CMD = os.path.expanduser("~/orca_6_1_0/orca_plot")

# =====================================================================
# --- DIRECTORY MAPPING ---
# =====================================================================

TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates")
DATA_ANALYSIS_DIR = os.path.join(PROJECT_DIR, "data_analysis")

# Input Geometries
BASE_MOLS_DIR = os.path.join(PROJECT_DIR, "base_molecules")
DOPED_MOLS_DIR = os.path.join(PROJECT_DIR, "doped_molecules")
DOUBLE_DOPED_MOLS_DIR = os.path.join(PROJECT_DIR, "double_doped_molecules")
STRESS_MOLS_DIR = os.path.join(PROJECT_DIR, "stress_test_molecules")

# Output Calculations
CALC_DIRS = {
    "base": os.path.join(PROJECT_DIR, "calculations/base_sp"),
    "vertical": os.path.join(PROJECT_DIR, "calculations/vertical_sp"),
    "adiabatic": os.path.join(PROJECT_DIR, "calculations/adiabatic_sp"),
    "additivity": os.path.join(PROJECT_DIR, "calculations/additivity_sp"),
    "stress": os.path.join(PROJECT_DIR, "calculations/stress_sp")
}

# =====================================================================
# --- SCIENTIFIC CONSTANTS & PARAMETERS ---
# =====================================================================

MAX_CONCURRENT_JOBS = 2
CHARGE = 0
MULT = 1
GRID_SIZE = 80

ORBITALS = ["h-1", "h", "l", "l+1"]
OVERLAP_THRESHOLD = 0.90

# Structural thresholds (in Angstroms)
CH_BOND_MAX = 1.25          
CC_ADJACENT_MAX = 1.6
SYM_TOLERANCE = 1.0

# =====================================================================
# --- MOLECULAR SYMMETRY RULES ---
# =====================================================================

SYMMETRY_RULES = {
    "AsymPAH2": {"X_sym": False, "Y_sym": True},
    "AsymPAH3": {"X_sym": False, "Y_sym": True},
    "BenPer":   {"X_sym": False, "Y_sym": True},
    "CirCor":   {"X_sym": True,  "Y_sym": True},
    "CirOv":    {"X_sym": True,  "Y_sym": True},
    "Cor":      {"X_sym": True,  "Y_sym": True},
    "RibPAH":   {"X_sym": True,  "Y_sym": True}
}

# =====================================================================
# --- PLOTTING COLORS ---
# =====================================================================

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
COLORS_MULTI = {'N+N': '#1f77b4', 'B+B': '#d62728', 'B+N': '#2ca02c'}