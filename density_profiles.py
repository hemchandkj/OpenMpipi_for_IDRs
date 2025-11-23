import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Paths to your trajectory and PDB files
trajectory_files = [file paths]
temperatures = [300, 310, 315, 320, 325]  # in Kelvin
pdb_file = '/content/drive/MyDrive/Colab Files/wtA1/equi_model.pdb'

def wrap_boundary(coords, box_length):
    """Wrap coordinates into [0, box_length] using periodic boundaries."""
    return np.remainder(coords, box_length)

def process_trajectory(traj_file, topology):
    """Load trajectory, center and wrap coordinates, return flattened x-coords and box length."""
    traj = md.load_xtc(traj_file, top=topology)[-1000:]  # Use last 1000 frames
    box_length = traj.unitcell_lengths[0][0]  # X box length
    traj.center_coordinates()
    coords = traj.xyz.copy()
    coords[:, :, 0] += box_length / 2
    coords[:, :, 0] = wrap_boundary(coords[:, :, 0], box_length)
    coords[:, :, 0] -= (box_length / 2)
    return coords[:, :, 0].flatten(), box_length, traj.n_frames

def compute_density(x_coords, box_length, num_frames, num_bins=50):
    """Compute density histogram for x-coordinates."""
    x_min, x_max = -box_length/2, box_length/2
    bins = np.linspace(x_min, x_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    hist, _ = np.histogram(x_coords, bins=bins)
    density = hist / (bin_width * num_frames)
    return bin_centers, density

def main():
    print('Loading topology...')
    topology = md.load(pdb_file).topology

    all_density = []
    max_density = 0

    # --- Process trajectories ---
    for traj_file, temp in zip(trajectory_files, temperatures):
        print(f"Processing: {traj_file} (Temperature {temp} K)")
        x_coords, box_length, n_frames = process_trajectory(traj_file, topology)
        bin_centers, density = compute_density(x_coords, box_length, n_frames)
        all_density.append((bin_centers, density, temp))
        max_density = max(max_density, np.max(density))
        # Save density data to CSV (Optional: keep this if you still want the data saved)
        df = pd.DataFrame({'Position (nm)': bin_centers, 'Density (atoms/nm)': density})
        df.to_csv(f'density_{temp}K.csv', index=False)

    y_max = max_density * 1.2  # 20% headroom for the plot

    # --- Plot each profile ---
    for bin_centers, density, temp in all_density:
        plt.figure(figsize=(15, 5))
        plt.plot(bin_centers, density, color=(0.353, 0.204, 0.655))
        plt.title(f'Temperature: {temp} K, 150mM NaCl')
        plt.xlabel('X-axis Position (nm)')
        plt.ylabel('Number Density (atoms / nm)')
        plt.ylim(0, y_max)
        # REMOVE: plt.savefig(f'density_profile_{temp}K.png', dpi=600, bbox_inches='tight')
        plt.show()  # ADD THIS LINE to display the plot
        plt.close() # Keep this to close the figure after displaying
        print(f"Displayed plot for {temp}K") # Optional: change print message

    print("All trajectories processed successfully!")

if __name__ == "__main__":
    main()
