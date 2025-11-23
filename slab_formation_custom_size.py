import os
from OpenMpipi import IDP, get_mpipi_system, PLATFORM, PROPERTIES
import openmm as mm
from openmm import app
import openmm.unit as unit
import numpy as np
import math
# ----------------------------------------------------------------------
# Optional: ensure mixed precision on CUDA
# ----------------------------------------------------------------------
PROPERTIES.setdefault("Precision", "mixed")  # use proper lowercase key

# ----------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------
Nprot = 63
seq = "MASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF"
chainID = "wtA1"
sim_temp = 300          # K
sim_C = 150             # mM NaCl

cube_length = 10.0      # nm, initial cube side
elongation_factor = 6
elongated_length = cube_length * elongation_factor  # 60 nm

# 10 fs timestep → 100,000 steps per ns
# For debugging, you may want: npt_steps = nvt_steps = 100_000 (1 ns)
npt_steps = 25_000_00     # 500 ns (production)
nvt_steps = 500_000_00    # 1000 ns (production)

# ----------------------------------------------------------------------
# SETUP: Single-chain model and coordinates
# ----------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

# Clean old XTC files if they exist (harmless even if we don't write XTC now)
for fname in ["outputs/npt.xtc", "outputs/nvt.xtc"]:
    if os.path.exists(fname):
        os.remove(fname)

prot = IDP(chainID, seq)
single_topology = prot.topology
coords = prot.initial_coords.copy()
coords -= coords.mean(axis=0)   # centre COM at origin

# ----------------------------------------------------------------------
# Build 63-chain system in a 10 nm cube
# ----------------------------------------------------------------------
grid_dim = 4  # 4x4x4 grid → 64 positions; we’ll use the first 63
spacing = (cube_length - np.ptp(coords, axis=0).max()) / (grid_dim - 1)

# Start with one chain at origin
model = app.Modeller(single_topology, coords * unit.nanometer)

# Place remaining chains
positions_grid = [
    (i, j, k)
    for i in range(grid_dim)
    for j in range(grid_dim)
    for k in range(grid_dim)
]

for idx, (i, j, k) in enumerate(positions_grid):
    if idx == 0:
        continue  # first chain already added
    if idx >= Nprot:
        break
    offset = np.array([i, j, k]) * spacing
    model.add(single_topology, (coords + offset) * unit.nanometer)

# ----------------------------------------------------------------------
# Define orthorhombic box (system + context) – 10 nm cube
# ----------------------------------------------------------------------
a_vec = mm.Vec3(cube_length, 0.0, 0.0)
b_vec = mm.Vec3(0.0, cube_length, 0.0)
c_vec = mm.Vec3(0.0, 0.0, cube_length)

box_vectors = [a_vec, b_vec, c_vec] * unit.nanometer
model.topology.setPeriodicBoxVectors(box_vectors)

# ----------------------------------------------------------------------
# Save initial PDB
# ----------------------------------------------------------------------
app.PDBFile.writeFile(
    model.topology,
    model.positions,
    open("outputs/initial_model.pdb", "w"),
)

# ----------------------------------------------------------------------
# Build Mpipi-Recharged system
#   IMPORTANT: pass positions with units (like in the working script)
# ----------------------------------------------------------------------
system = get_mpipi_system(
    model.positions,      # <--- changed from positions_no_unit to unit-bearing positions
    model.topology,
    {chainID: []},
    sim_temp,
    sim_C,
    CM_remover=True,
    periodic=True,
)

# Force the System's default periodic box to be perfectly orthorhombic
system.setDefaultPeriodicBoxVectors(a_vec, b_vec, c_vec)

# Add barostat for NPT
barostat = mm.MonteCarloBarostat(
    0.50 * unit.atmosphere,
    sim_temp * unit.kelvin,
    100,
)
system.addForce(barostat)

# ----------------------------------------------------------------------
# Integrator and Simulation (first context, with barostat)
# ----------------------------------------------------------------------
integrator = mm.LangevinMiddleIntegrator(
    sim_temp * unit.kelvin,
    0.2 / unit.picosecond,
    10.0 * unit.femtosecond,
)

simulation = app.Simulation(
    model.topology,
    system,
    integrator,
    platform=PLATFORM,
    platformProperties=PROPERTIES,
)

# Set box BEFORE positions in the context
simulation.context.setPeriodicBoxVectors(*box_vectors)
simulation.context.setPositions(model.positions)
simulation.minimizeEnergy()

# ----------------------------------------------------------------------
# NPT: compression in 10 nm cube
# ----------------------------------------------------------------------
simulation.reporters.append(
    app.StateDataReporter(
        "outputs/npt_data.out",
        10_000,
        step=True,
        potentialEnergy=True,
        temperature=True,
        elapsedTime=True,
        density=True,
        volume=True,
    )
)
simulation.reporters.append(
    app.CheckpointReporter("outputs/npt.chk", 500_000)
)

print("Running NPT (compression) in 10 nm cube...")
simulation.step(npt_steps)

state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    enforcePeriodicBox=True,
)

# ----------------------------------------------------------------------
# Remove barostat and rebuild context cleanly (NEW integrator for NVT)
# ----------------------------------------------------------------------
for idx in range(system.getNumForces()):
    if isinstance(system.getForce(idx), mm.MonteCarloBarostat):
        system.removeForce(idx)
        break

# Use the final NPT box as new default
box_a, box_b, box_c = state.getPeriodicBoxVectors()
system.setDefaultPeriodicBoxVectors(
    box_a.value_in_unit(unit.nanometer),
    box_b.value_in_unit(unit.nanometer),
    box_c.value_in_unit(unit.nanometer),
)

# Create a BRAND NEW integrator for the NVT phase
new_integrator = mm.LangevinMiddleIntegrator(
    sim_temp * unit.kelvin,
    0.2 / unit.picosecond,
    10.0 * unit.femtosecond,
)

# Build new context using the NEW integrator
context = mm.Context(system, new_integrator, PLATFORM, PROPERTIES)
context.setState(state)

# Update the Simulation object
simulation._integrator = new_integrator
simulation.integrator = new_integrator
simulation._context = context
simulation.context = context

# ----------------------------------------------------------------------
# Stretch box along x to 60 nm
# ----------------------------------------------------------------------
new_a_vec = mm.Vec3(elongated_length, 0.0, 0.0)
new_b_vec = mm.Vec3(0.0, cube_length, 0.0)
new_c_vec = mm.Vec3(0.0, 0.0, cube_length)
new_box = [new_a_vec, new_b_vec, new_c_vec] * unit.nanometer

simulation.context.setPeriodicBoxVectors(*new_box)

# ----------------------------------------------------------------------
# Re-centre chains in elongated box
# ----------------------------------------------------------------------
positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
old_com = np.mean(positions, axis=0)
box_center = np.array(
    [elongated_length / 2.0, cube_length / 2.0, cube_length / 2.0]
)
shift = box_center - old_com
shifted_positions = positions + shift
simulation.context.setPositions(shifted_positions * unit.nanometer)

# ----------------------------------------------------------------------
# Critical stabilisation step:
# Minimise & reset velocities AFTER stretching (to avoid NaNs)
# ----------------------------------------------------------------------
print("Minimizing after stretching...")
simulation.minimizeEnergy(maxIterations=500)

print("Reassigning velocities at target temperature after stretching...")
simulation.context.setVelocitiesToTemperature(sim_temp * unit.kelvin)


# Save elongated, minimized configuration with correct box
state_el = simulation.context.getState(getPositions=True)
box_el = simulation.context.getState().getPeriodicBoxVectors()
simulation.topology.setPeriodicBoxVectors(box_el)

with open("outputs/elongated_model.pdb", "w") as f:
    app.PDBFile.writeFile(
        simulation.topology,
        state_el.getPositions(),
        f,
    )


# ----------------------------------------------------------------------
# NVT in elongated 60 nm box
# ----------------------------------------------------------------------
simulation.reporters = []
simulation.reporters.append(
    app.StateDataReporter(
        "outputs/nvt_data.out",
        10_000,
        step=True,
        potentialEnergy=True,
        temperature=True,
        elapsedTime=True,
        density=True,
        volume=True,
    )
)
simulation.reporters.append(
    app.CheckpointReporter("outputs/nvt.chk", 500_000)
)

print("Running NVT (elongated box)...")
simulation.step(nvt_steps)


# Save final configuration with correct slab box
state_fin = simulation.context.getState(getPositions=True)
box_fin = simulation.context.getState().getPeriodicBoxVectors()
simulation.topology.setPeriodicBoxVectors(box_fin)

with open("final_model.pdb", "w") as f:
    app.PDBFile.writeFile(
        simulation.topology,
        state_fin.getPositions(),
        f,
    )



print("Simulation completed: outputs in ./outputs")
