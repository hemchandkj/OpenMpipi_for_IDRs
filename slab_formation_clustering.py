from OpenMpipi import IDP, get_mpipi_system, PLATFORM, PROPERTIES
import numpy as np
import math
import openmm as mm
import openmm.unit as unit
from openmm import app
import os
import scipy.constants as c
from scipy.spatial.distance import cdist


"""
Protein-only version of the two-stage OpenMM simulation using OpenMpipi.
 - Builds compact starting structures for wt_a1 protein.
 - Tiles multiple chains on a 3D grid with spacing.
 - Compresses the system via NPT (barostat) until a target density, then removes the barostat.
 - Rescales the box along x to reach ~0.1 g/cm³ while checking y–z dimensions against chain Rg to avoid self-interaction.
 - Continues with an NVT production run in the elongated box.
 - Writes trajectories, state logs, intermediate/final PDBs, and serialized system.xml/state.xml.
"""

# ------------------------------------------
# PARAMETERS
# ------------------------------------------

min_separation = 2.0             # nm; minimum spacing between protein chains
sim_time_compress = 500          # ns; NPT compression phase
sim_time_relax = 1500            # ns; NVT relaxation phase
sim_temp = 273                   # K
sim_C = 150                      # mM NaCl
compact_initial = True           # use compact initial chains
compact_SimTime_prot = 30        # ns; compacting protein chains
barostat_pressure = 1.0          # atm; initial barostat pressure
threshold_density = 0.6 * unit.gram / unit.centimeter**3  # target density for barostat run
max_cycles = 0                   # additional barostat cycles
cycle_length = 50                # ns per additional barostat cycle
barostat_increase = 1.0          # atm per cycle
box_scaling = 6                  # fallback x scaling for box if compression fails

# --------- Protein parameters -----------
seq = 'MGDEDWEAEINPHMSSYVPIFEKDRYSGENGDNFNRTPASSSEMDDGPSRRDHFMKSGFASGRNFGNRDAGECNKRDNTSTMGGFGVGKSFGNRGFSNSRFEDGDSSGFWRESSNDCEDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFGSRRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEEVITGSGKNSWKSEAEGGES'
chainID = 'DDX4'
Nprot = 60                       # number of protein chains to place

# ------------------------------------------
# HELPER FUNCTIONS (Protein Only)
# ------------------------------------------
def calc_biomolecule_dimensions(biomolecule, compact_initial_prot):
    coords = biomolecule.min_rg_coords if compact_initial_prot else biomolecule.initial_coords
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    extent = maxs - mins
    return extent[0], extent[1], extent[2]  # x, y, z

def calc_rg(simulation):
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    prot_rgs = []
    for biomol in simulation.topology.chains():
        idx = [atom.index for atom in biomol.atoms()]
        coords = pos[idx]
        com = coords.mean(axis=0)
        sq = np.sum((coords - com)**2, axis=1)
        rg = np.sqrt(np.mean(sq))
        prot_rgs.append(rg)
    def rg_stats(arr):
        if not arr:
            return None, None
        return float(np.array(arr).max()), float(np.array(arr).std())
    prot_stats = rg_stats(prot_rgs)
    return prot_stats

def calc_target_box(prot, current_box, Nprot, target_density=0.1*unit.gram/unit.centimeter**3):
    total_mass_g = (prot.chain_mass.value_in_unit(unit.dalton)/c.Avogadro*unit.gram * Nprot)
    x_side = current_box[0][0]
    y_side = current_box[1][1]
    z_side = current_box[2][2]
    current_density = total_mass_g / (x_side * y_side * z_side)
    target_volume = total_mass_g / target_density
    target_x_long_side = target_volume / (y_side * z_side)
    x_scale_factor = target_x_long_side / x_side
    target_box = [mm.Vec3(current_box[0][0].value_in_unit(unit.nanometer) * x_scale_factor,
                    current_box[0][1].value_in_unit(unit.nanometer),
                    current_box[0][2].value_in_unit(unit.nanometer)),
                mm.Vec3(current_box[1][0].value_in_unit(unit.nanometer),
                    current_box[1][1].value_in_unit(unit.nanometer),
                    current_box[1][2].value_in_unit(unit.nanometer)),
                mm.Vec3(current_box[2][0].value_in_unit(unit.nanometer),
                    current_box[2][1].value_in_unit(unit.nanometer),
                    current_box[2][2].value_in_unit(unit.nanometer))] * unit.nanometer
    return x_scale_factor, current_density

def check_box_dimensions(prot_stats, current_box_vectors):
    min_side_length = 2.6 * prot_stats[0]  # 2.6 * Rg for protein only
    if current_box_vectors[1][1] < min_side_length or current_box_vectors[2][2] < min_side_length:
        print(f"!!WARNING!! Box dimensions {current_box_vectors} are too small. Minimum required dimensions: {min_side_length:.2f} nm in y-z direction.")
    else:
        print(f" -  Minimum required dimensions: {min_side_length:.2f} nm in y-z direction. Box dimensions {current_box_vectors[0][0]} x {current_box_vectors[1][1]} x {current_box_vectors[2][2]} nm are sufficient to avoid self-interaction.\n", flush=True)

# ------------------------------------------
# Initialize the protein chain
# ------------------------------------------
os.makedirs('OUTPUTS_pdbmodels', exist_ok=True)
prot = IDP(chainID, seq)
prot.get_compact_model(simulation_time = compact_SimTime_prot * unit.nanosecond, T=sim_temp*unit.kelvin, csx=sim_C)
prot_coord = prot.min_rg_coords if compact_initial else prot.initial_coords
prot_coord -= prot_coord.mean(axis=0)

# Save the pdb structure files
pmodel = app.Modeller(prot.topology, prot.initial_coords * unit.nanometer)
pmodel_relaxed = app.Modeller(prot.topology, prot.min_rg_coords * unit.nanometer)
app.PDBFile.writeFile(pmodel.topology, pmodel.positions, open(os.path.join('OUTPUTS_pdbmodels', prot.chain_id + '_model.pdb'), 'w'))
app.PDBFile.writeFile(pmodel_relaxed.topology, pmodel_relaxed.positions, open(os.path.join('OUTPUTS_pdbmodels', prot.chain_id + '_compact_model.pdb'), 'w'))

print(f"\nSIMULATION PARAMETERS:", flush=True)
print(f" - {Nprot} {chainID} protein chains\n", flush=True)

# -------------------------------------------
# Build protein-only system
# -------------------------------------------
print("Setting up the initial model with protein chains...", flush=True)
total_biomols = Nprot
grid_dim = math.ceil(total_biomols ** (1/3))
print(f" - Number of grid dimensions: {grid_dim} x {grid_dim} x {grid_dim} ({grid_dim**3} total grids)\n - Total number of protein chains: {total_biomols}", flush=True)

# Assign all grid positions to protein
biomol_types = ['prot'] * Nprot
np.random.seed(42)
np.random.shuffle(biomol_types)
grid_biomol = np.empty((grid_dim, grid_dim, grid_dim), dtype=object)
count = 0
for i in range(grid_dim):
    for j in range(grid_dim):
        for k in range(grid_dim):
            if count < len(biomol_types):
                grid_biomol[i, j, k] = biomol_types[count]
                count += 1
            else:
                grid_biomol[i, j, k] = None

print(f" - Assigned the grids with protein chains.", flush=True)

# Initialize modeller and place first protein at origin
print(" - Initializing the modeller and placing the first protein at the origin...", flush=True)
model = app.Modeller(prot.topology, prot_coord * unit.nanometer)
print("   First protein chain is placed at the origin.\n", flush=True)

# Generate grid positions with offsets
prot_length = np.array(calc_biomolecule_dimensions(prot, compact_initial))
if compact_initial == True:
    biomolecule_spacing = np.max(prot_length + min_separation)
    x_offsets = np.arange(grid_dim) * biomolecule_spacing
    y_offsets = np.arange(grid_dim) * biomolecule_spacing
    z_offsets = np.arange(grid_dim) * biomolecule_spacing
else:
    x_offsets = np.arange(grid_dim) * (prot_length[0] + min_separation)
    y_offsets = np.arange(grid_dim) * (prot_length[1] + min_separation)
    z_offsets = np.arange(grid_dim) * (prot_length[2] + min_separation)
X, Y, Z = np.meshgrid(x_offsets, y_offsets, z_offsets, indexing='ij')
grid_coords = np.stack((X, Y, Z), axis=-1)

print("   Grid spacings are computed.", flush=True)

# Place remaining protein chains in the grid
count = 1
print(f" - Placing the rest of the protein chains {total_biomols -1 } in the grid...", flush=True)
for i in range(grid_dim):
    for j in range(grid_dim):
        for k in range(grid_dim):
            if count >= total_biomols:
                break
            if (i, j, k) == (0, 0, 0):
                continue
            biomol = grid_biomol[i, j, k]
            if biomol is None:
                continue
            grid_coord = grid_coords[i, j, k]
            model.add(prot.topology, (prot_coord + grid_coord) * unit.nanometer)
            count += 1
            if count % 100 == 0:
                model = app.Modeller(model.topology, model.positions)

print("   All protein chains are placed in the grid.\n", flush=True)

app.PDBFile.writeFile(model.topology, model.positions, open(os.path.join('OUTPUTS_pdbmodels', 'start_model.pdb'), 'w'))
print(f'The initial model is set up.', flush=True)
assert model.topology.getNumAtoms() == len(model.positions), " - !!WARNING!! Mismatch between topology and coordinate count."
print(f" - Total number of atoms in the system: {model.topology.getNumAtoms()}\n", flush=True)

# ----------------------------------------------
# Set periodic box for initial simulation
# ----------------------------------------------
print("Setting periodic box and initializing the simulation.")
model_positions = np.array(model.positions.value_in_unit(unit.nanometer))
box_length = np.max(np.ptp(model_positions, axis=0)) + biomolecule_spacing / 2
box_vecs = [mm.Vec3(box_length, 0.0,        0.0),
            mm.Vec3(0.0,        box_length, 0.0),
            mm.Vec3(0.0,        0.0,        box_length)] * unit.nanometer
model.topology.setPeriodicBoxVectors(box_vecs)
print("  Periodic box vectors:")
for i, vec in enumerate(box_vecs):
    print(f"v{i+1} =", vec)
print("  Periodic box is set.\n", flush=True)

# ----------- Set up system  -------------
print(" - Initializing simulation system...", flush=True)
globular_indices_dict = {chain.id: [] for chain in model.topology.chains()}
system = get_mpipi_system(model.positions, model.topology, globular_indices_dict, sim_temp, sim_C, CM_remover=True, periodic=True)
print(" - Simulation system is initialized.\n", flush=True)

# -----------------------------------
# Simulation run 1: NPT with barostat to compress the box over 200 ns
# -----------------------------------
print("Starting simulation run 1: NPT with barostat to compress the box.", flush=True)
barostat = mm.MonteCarloBarostat(barostat_pressure * unit.atmosphere, sim_temp * unit.kelvin, 100)
system.addForce(barostat)
integrator = mm.LangevinMiddleIntegrator(sim_temp * unit.kelvin, 0.2 / unit.picosecond, 10 * unit.femtosecond)
simulation = app.Simulation(model.topology, system, integrator, platform=PLATFORM, platformProperties=PROPERTIES)
simulation.context.setPositions(model.positions)
simulation.context.setPeriodicBoxVectors(*model.topology.getPeriodicBoxVectors())

with open(os.path.join('OUTPUTS_pdbmodels', "initial_simulation_setup.pdb"), "w") as f:
    app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f, keepIds=True)
print(f" - Simulation box is set, saved initial setup as OUTPUTS_pdbmodels/initial_simulation_setup.pdb.\n", flush=True)

print(" - Minimizing energy...", flush=True)
simulation.minimizeEnergy()
print("  Energy is minimized.", flush=True)

simulation.reporters.append(app.XTCReporter('traj_barostat.xtc', 100000))
simulation.reporters.append(app.StateDataReporter('state_data_barostat.out', reportInterval=10000, step=True,
                                                   potentialEnergy=True, temperature=True, elapsedTime=True, density=True, volume=True))

print(f" - Running isotropic NPT simulation for {sim_time_compress} ns, with pressure {barostat_pressure}...", flush=True)
Nsteps_compress = int(sim_time_compress * unit.nanosecond / (10 * unit.femtosecond))
simulation.step(Nsteps_compress)
print("   Simulation run 1 completed.\n", flush=True)

#cluster based method 

def get_largest_cluster_fraction(positions, cutoff=1.5):
    # positions: (N_atoms, 3) in nm
    # Returns: fraction of atoms in largest cluster
    # Simple single-linkage clustering by distance
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    dists = cdist(positions, positions)
    adjacency = (dists < cutoff).astype(int)
    np.fill_diagonal(adjacency, 0)
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    # Find largest cluster
    largest = np.bincount(labels).max()
    return largest / len(labels)

# ... after NPT run
converged = False
max_npt_cycles = 5  # safety
for cycle in range(max_npt_cycles):
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    frac = get_largest_cluster_fraction(pos, cutoff=1.5)  # 1.5 nm is typical for CG beads
    print(f"Cycle {cycle}: Largest cluster fraction = {frac:.2f}")
    if frac > 0.90:  # >90% of atoms in one cluster
        print("Single dense cluster formed. Proceeding to slab setup.")
        converged = True
        break
    else:
        print("Cluster not yet formed. Continuing NPT...")
        simulation.step(int(50 * unit.nanosecond / (10 * unit.femtosecond)))  # 50 ns more
if not converged:
    print("Warning: Did not reach single dense cluster after max cycles.")


state = simulation.context.getState(getPositions=True, getVelocities=True)

# -----------------------------
# Remove the barostat and extend the box in x-direction and center the droplet
# -----------------------------
print(" - Removing barostat, extending box in x-direction, and centering the droplet...\n", flush=True)
for idx in range(system.getNumForces()):
    force = system.getForce(idx)
    if isinstance(force, mm.MonteCarloBarostat):
        system.removeForce(idx)
        print("   Barostat removed.\n", flush=True)
        break
simulation.context.reinitialize()
simulation.context.setState(state)


current_box = state.getPeriodicBoxVectors()
current_box_vectors = current_box.value_in_unit(unit.nanometer)
x_scale_factor, current_density = calc_target_box(prot, current_box, Nprot, target_density=0.1*unit.gram/unit.centimeter**3)
print(f" - Current condensate density: {current_density.value_in_unit(unit.gram/unit.centimeter**3)} g/cm^3", flush=True)
print(f" - X scale factor for box extension: {x_scale_factor}", flush=True)
prot_stats = calc_rg(simulation)
print(f" - Protein sizes: largest Rg = {prot_stats[0]:.2f} nm, std Rg = {prot_stats[1]:.2f} nm", flush=True)
check_box_dimensions(prot_stats, current_box_vectors)

# Calculate the new box dimensions
old_box_x = current_box[0][0]

new_box_x = x_scale_factor * old_box_x

# Calculate the center of mass of the protein droplet
positions = state.getPositions(asNumpy=True)

center_of_mass_x = np.mean(positions.value_in_unit(unit.nanometer)[:, 0]) * unit.nanometer
new_box_center_x = new_box_x / 2.0
shift_x = new_box_center_x - center_of_mass_x


# Create the new box vectors
new_box = [mm.Vec3(new_box_x, 0.0, 0.0),
           current_box[1],
           current_box[2]]

# Apply the shift to the atomic positions
shifted_positions = positions + mm.Vec3(shift_x.value_in_unit(unit.nanometer), 0.0, 0.0) * unit.nanometer
simulation.context.setPositions(shifted_positions)

# Set the new box vectors
simulation.context.setPeriodicBoxVectors(*new_box)

print(f" - Box has been extended in the x-direction with a scale factor {x_scale_factor:.2f} and centered.\n", flush=True)

simulation.context.setVelocitiesToTemperature(sim_temp * unit.kelvin)

integrator.setFriction(0.2 / unit.picosecond)

# -----------------------------
# Simulation run 2: NVT with extended box for relaxation
# -----------------------------
print("Starting simulation run 2: NVT with extended box.", flush=True)
simulation.reporters = []
simulation.reporters.append(app.XTCReporter('traj_NVT.xtc', 100000))
simulation.reporters.append(app.StateDataReporter('state_data_NVT.out', reportInterval=10000, step=True, potentialEnergy=True, temperature=True, elapsedTime=True, density=True, volume=True))
with open('initial_NVT_model.pdb', 'w') as f:
    state = simulation.context.getState(getPositions=True)
    box_vectors = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(box_vectors)
    app.PDBFile.writeFile(simulation.topology, state.getPositions(), f, keepIds=False)

print(f" - Running NVT simulation for {sim_time_relax} ns...", flush=True)
Nsteps_relax = int(sim_time_relax * unit.nanosecond / (10 * unit.femtosecond))
simulation.step(Nsteps_relax)
print("   Simulation run 2 complete.", flush=True)

print(" - Saving final coordinates to PDB...", flush=True)
with open('final_model.pdb', 'w') as f:
    state = simulation.context.getState(getPositions=True)
    box_vectors = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(box_vectors)
    app.PDBFile.writeFile(simulation.topology, state.getPositions(), f, keepIds=False)
print("   Final coordinates saved as final_model.pdb.\n", flush=True)
final_box = simulation.context.getState().getPeriodicBoxVectors()

with open('system.xml', 'w') as f:
    f.write(mm.XmlSerializer.serialize(system))
with open('state.xml', 'w') as outfile:
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    state_xml = mm.XmlSerializer.serialize(state)
    outfile.write(state_xml)
print("   Last state has been saved as state.xml and system.xml.")

print("Simulation has finished.", flush=True)
