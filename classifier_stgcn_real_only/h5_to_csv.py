import h5py
import os
import numpy as np

# Path to my .h5 file
h5_path = '/fs/nexus-scratch/eloirghi/STEP/pie_affective_features.h5'

# i want to save the CSVs in my STEP folder
save_dir = '/fs/nexus-scratch/eloirghi/STEP/'
os.makedirs(save_dir, exist_ok=True)

# load the HDF5 file
with h5py.File(h5_path, 'r') as f:
    for key in f.keys():
        print(f"Exporting {key}...")
        data = np.array(f[key])
        filename = key.strip("/") + ".csv"
        save_path = os.path.join(save_dir, filename)
        np.savetxt(save_path, data, delimiter=',')

print(f"\n✅ All CSVs saved to: {save_dir}")