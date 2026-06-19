# Scripts for: "A Predictive Model for Frontier Orbital Energy Shifts in Symmetrically Edge-Doped Polycyclic Aromatic Hydrocarbons: A DFT Study"

This repository contains the automated Python workflow used to generate, execute, and analyze Density Functional Theory (DFT) calculations for the associated study. 

The pipeline handles strict topological symmetry generation for Nitrogen and Boron doping, automated scheduling of ORCA single-point and optimization calculations, block-diagonal orbital overlap tracking, and the extraction of structure-property relationships (e.g., Leave-One-Out Cross Validation for electronic relaxation).

## Prerequisites

To run this pipeline, you must have the following installed:
* Python 3.8+
* ORCA 6.1.0 (The `orca` and `orca_plot` binaries must be accessible. ORCA is available for free to use for academic purposes and is available at https://orcaforum.kofo.mpg.de/)

## Installation

1. Clone or download this repository.
2. Install the required Python packages by running this in your terminal:
   `pip install -r requirements.txt`

## Configuration

Before running any calculations, you must configure the pipeline to match your local system.

1. Open `scripts/config.py`.
2. Locate the `SYSTEM CONFIGURATION` section at the top of the file.
3. Update the `PROJECT_DIR` and `ORCA_CMD` variables with the absolute paths for your local machine. If ORCA is already in your system PATH, you can leave the command as `"orca"`.

All structural thresholds (e.g., C-H bond length max), grid sizes, and symmetry rules are centralized in `config.py` and can be adjusted there.

## Usage

The entire workflow is managed through a central wrapper script. It can be run interactively or via command-line arguments.

### Interactive Mode
Run the pipeline script to access the interactive menu:
`python scripts/main.py`

This will present a numbered menu allowing you to execute specific workflows (e.g., Base SP, Vertical Doping, Additivity, Stress Testing, or Adiabatic Relaxation). 

### Command Line Interface (CLI) Reference

While both scripts (`main.py` and `difference_plotter.py`) feature an interactive menu if run without arguments, they can be fully automated using the following CLI flags.

#### 1. Master Pipeline (`main.py`)
Use the `--run` flag to bypass the interactive menu and execute a specific, end-to-end workflow.

* `python scripts/main.py --run base` : Runs pristine PAH geometry optimizations and single-point calculations.
* `python scripts/main.py --run vertical` : Runs the vertical excitation workflow (Geometry Generation -> SP Scheduler -> Plotter -> Analyzer).
* `python scripts/main.py --run additivity` : Runs the complex co-doping workflow.
* `python scripts/main.py --run stress` : Runs the haywire multi-site combinatorial workflow.
* `python scripts/main.py --run adiabatic` : Runs the structural relaxation workflow.
* `python scripts/main.py --run all_schedulers` : Only executes the ORCA queue schedulers for all modes without running the analyzers.
* `python scripts/main.py --run full_pipeline` : Executes every workflow sequentially from start to finish.

#### 2. Difference Plotter (`difference_plotter.py`)
This script can be run independently to calculate block-diagonal overlaps and generate difference density `.cube` files. It requires two flags for automation: `--mode` and `--overwrite`.

**Modes:**
* `--mode 1` : Analyze Vertical SP calculations.
* `--mode 2` : Analyze Additivity SP calculations.
* `--mode 3` : Analyze Stress Test calculations.
* `--mode 4` : Rename Adiabatic raw cubes (does not calculate overlaps).
* `--mode 5` : Run all modes sequentially.

**Overwrite Flags:**
* `--overwrite y` : Force recalculation of all difference maps, overwriting existing `.cube` and `_overlap.txt` files.
* `--overwrite n` : Skip molecules that have already been analyzed (Safe mode).

**Example Command:**
```bash
# Analyze vertical calculations and skip already processed molecules
python scripts/difference_plotter.py --mode 1 --overwrite n
```
## Script Descriptions

The `scripts/` directory contains the modular architecture of the pipeline. Each script is designed to perform a specific task, passing its output forward as the input for the next stage.

### Core Configuration
* **`config.py`**
  * **Description:** Centralizes all system paths, math thresholds (e.g., C-H bond limits, overlap thresholds), and strict topological symmetry rules. 
  * **Input:** User-defined environment paths.
  * **Output:** Global variables imported by all other scripts.
* **`main.py`**
  * **Description:** The master CLI wrapper that safely executes the workflow stages in sequence and handles crash protection.
  * **Input:** Interactive menu selections or command-line arguments (e.g., `--run vertical`).
  * **Output:** Sequential subprocess execution of the pipeline scripts.

### Generators (Topological Substitution)
* **`doped_generator.py`**
  * **Description:** Applies mathematical symmetry rules to generate single-site Nitrogen and Boron edge-substitutions.
  * **Input:** Optimized pristine `.xyz` files from the base calculations.
  * **Output:** Single-site doped `.xyz` files saved to `doped_molecules/`.
* **`double_doped_generator.py`**
  * **Description:** Dynamically generates dual-doped configurations, applying structural adjacency limits to create N+N, B+B, and N+B pairings.
  * **Input:** Optimized pristine `.xyz` files.
  * **Output:** Dual-doped `.xyz` files saved to `double_doped_molecules/`.
* **`stress_test_generator.py`**
  * **Description:** Brute-forces "haywire" multi-site dopant combinations to test the physical limits of the predictive models.
  * **Input:** Pristine `CirCor` `.xyz` geometry.
  * **Output:** Highly doped combinatorial `.xyz` files saved to `stress_test_molecules/`.

### Schedulers (ORCA Execution)
*All schedulers handle queue management, crash recovery, and automated `.inp` template generation.*
* **`base_scheduler.py`**
  * **Description:** Establishes the baseline structural and electronic properties for the pristine PAHs.
  * **Input:** Pristine `.xyz` files from `base_molecules/`.
  * **Output:** Full geometry optimizations (`_opt.out`, `.gbw`) and single-point baseline calculations.
* **`vertical_scheduler.py` & `additivity_scheduler.py`**
  * **Description:** Calculates vertical excitations by projecting doped atoms onto the frozen pristine PAH geometry.
  * **Input:** Doped `.xyz` files and pristine guess `.gbw` files.
  * **Output:** Single-point ORCA calculations (`.out`) and raw orbital `.cube` files.
* **`adiabatic_scheduler.py`**
  * **Description:** Calculates full electronic and structural relaxation effects.
  * **Input:** Doped `.xyz` files from the generators.
  * **Output:** Doped geometry optimizations (`_opt.xyz`) and subsequent relaxed single-point calculations (`.out`, `.cube`).
* **`stress_scheduler.py`**
  * **Description:** Executes the high-volume stress test single-point queue.
  * **Input:** Stress test `.xyz` geometries.
  * **Output:** Single-point ORCA calculations (`.out`, `.cube`).

### Analysis & Plotting
* **`difference_plotter.py`**
  * **Description:** Executes a block-diagonal orbital overlap solver to track shifting frontier orbitals across permutations, handling state-crossing.
  * **Input:** Base and doped `.cube` files, and ORCA `.out` files.
  * **Output:** Standardized `.cube` files, block-diagonal overlap matrices (`_overlap.txt`), and normalized difference density maps.
* **`vertical_analyzer.py`**
  * **Description:** Performs Leave-One-Out Cross Validation (LOOCV) regressions to correlate unperturbed $p_z$ MO amplitudes with vertical energy shifts.
  * **Input:** ORCA `.out` files (for MO populations/energies) and `_overlap.txt` files.
  * **Output:** Amplitude vs. Energy SVG plots, size-adaptive regression models, LOOCV parity SVGs, residual plots, and detailed statistical `.txt` summaries.
* **`additivity_analyzer.py`**
  * **Description:** Evaluates co-doping parity by comparing directly calculated DFT shifts against equation-predicted additivity models.
  * **Input:** Double-doped `.out` and `_overlap.txt` files, and trained linear regression parameters.
  * **Output:** Direct parity SVGs, LOOCV parity SVGs, combined amplitude correlation plots, and global error metric summaries.
* **`adiabatic_analyzer.py`**
  * **Description:** Correlates physical structural relaxation (measured via Kabsch RMSD) with adiabatic electronic energy shifts.
  * **Input:** Relaxed adiabatic `.xyz` geometries, vertical SP `.out` files, and adiabatic SP `.out` files.
  * **Output:** RMSD vs. Relaxation Energy SVGs, adiabatic LOOCV parity SVGs, and text summaries detailing structural shifts.
* **`stress_analyzer.py`**
  * **Description:** Evaluates at what doping concentration the PAH frontier orbitals are fundamentally destroyed (S < 0.90).
  * **Input:** Master combined `_overlaps.txt` file from the stress pipeline.
  * **Output:** Categorical overlap clustering SVGs and survival metric `.txt` summaries.

### Utilities & Extras
* **`mo_extractor.py`**
  * **Description:** A utility to ensure clean orbital coefficient extraction.
  * **Input:** Base `.gbw` and `.inp` files.
  * **Output:** Runs a zero-iteration ORCA print job to extract isolated $p_z$ MO coefficients into a dedicated `_MO.out` file.
* **`energy_extractor.py`**
  * **Description:** *(Optional Utility)* A lightweight script for quick data harvesting without running the full mathematical analyzers.
  * **Input:** SP `.out` files and `_overlap.txt` matrices.
  * **Output:** Consolidates Total Energies (Eh), standard HOMO-LUMO gaps (eV), and tracked orbital shifts into a quick-reference `.txt` summary.

## Repository Structure

* `base_molecules/`: Initial pristine `.xyz` geometries (e.g., CirCor, AsymPAH2).
* `templates/`: Input blocks (`.inp`) for ORCA calculations containing the chosen functional and basis sets.
* `scripts/`: The Python scripts driving the workflow.
* `data_analysis_example`: Example `data_analysis` folder with results to test reproducibility.
* `data_analysis/`: Output directory where final plots, LOOCV parity charts, and summary text files are saved. 
* `calculations/`: Output directory where ORCA calculations are saved.
* `doped_molecules/`: Doped molecule `.xyz` geometries (e. g., CirCor_P1N, AsymPAH2_P2B).
* `double_doped_molecules/`: Double doped molecule `.xyz` geometries (e. g., CirCor_P1N_P3N, AsymPAH2_P2B_P6N).
* `stress_test_molecules/`: Doped molecule `.xyz` geometries for Stress calculations (e. g., CirCor_P1N_P2N_P3B_P5N).
* `requirements.txt`: Python prerequisites.
* `.gitignore`
* `README.md`
* `LICENSE`

`calculations/`, `data_analysis/` and molecule folders except for `base_molecules/` are generated and are not part of repository. 

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. See the LICENSE file for details.