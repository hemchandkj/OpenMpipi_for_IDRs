# OpenMpipi_for_IDRs

This repository contains Python scripts for OpenMpipi implementation for intrinsically disordered proteins simulating biomolecular condensates with the residue-resolution coarse-grained force field (OpenMpipi). The scripts are designed for GPU-accelerated, direct-coexistence simulations of charge-rich, intrinsically disordered proteins (e.g., hnRNPA1-LCD, DDX4-NTD), along with post-processing tools for phase diagrams, critical temperatures, density profiles, and performance benchmarking.

---

## Repository structure

- `test.py`  
  Two-stage **direct-coexistence** OpenMM protocol using OpenMpipi:
  1. Build compact initial chains and tile multiple proteins on a 3D grid.  
  2. Compress the system via NPT to a target density.  
  3. Stretch the box along one axis to form a slab and run NVT production.  

- `production_test.py`  
  Continues an Mpipi-Recharged simulation from an equilibrated `final_model.pdb`
  using OpenMM + CUDA (mixed precision), writing an `trajectory.xtc` and state
  outputs.

- `Tc calculations.py`  
  Reads pre-computed coexistence densities from `.csv`, fits a binodal, and
  extracts the critical temperature **Tc** (and fit parameters) using
  non-linear least-squares. Optionally generates publication-quality plots.

- `density_profiles.py`  
  Uses **MDTraj** to load `.xtc` trajectories and corresponding `.pdb` files,
  wraps and centres coordinates, and computes 1D density profiles along the
  slab axis (x) over multiple temperatures. Outputs profiles to `.csv` and
  plots.

- `timesteps per second.py`  
  Parses `state.out` timing files from GPU benchmarks (different system sizes),
  computes timesteps per second, and plots performance scaling.

---

## Requirements

- Python 3.9+  
- [OpenMM](https://openmm.org) (GPU build; CUDA recommended)  
- The **OpenMpipi** / `Mpipi-Recharged` Python package (local or installed)  
- Python libraries:
  - `numpy`, `pandas`, `scipy`, `matplotlib`
  - `mdtraj` (for `density_profiles.py`)

You can install typical dependencies with:

```bash
conda create -n openmpipi python=3.10
conda activate openmpipi
conda install -c conda-forge openmm mdtraj numpy scipy pandas matplotlib
# then install/clone your OpenMpipi package


Note: The original OpenMpipi is available at https://github.com/CollepardoLab/OpenMpipi
