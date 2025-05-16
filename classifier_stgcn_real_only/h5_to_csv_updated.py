import h5py
import os
import numpy as np

# paths
base_dir = '/fs/nexus-scratch/eloirghi/STEP/data'

# looping over all .h5 files in the data folder
for filename in os.listdir(base_dir):
    if filename.endswith('.h5'):
        h5_path = os.path.join(base_dir, filename)
        print(f"📂 Processing: {h5_path}")
        
        # directory named after the h5 file without .h5 extension
        subfolder_name = filename.replace('.h5', '')
        subfolder_path = os.path.join(base_dir, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # open and extract all h5 datasets under data/
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                print(f"  ↳ Extracting dataset: {key}")
                data = np.array(f[key])
                csv_filename = key.strip("/") + ".csv"
                save_path = os.path.join(subfolder_path, csv_filename)
                np.savetxt(save_path, data, delimiter=',')
                
        print(f"✅ Saved to: {subfolder_path}\n")

print("🎉 Done extracting all .h5 files.")
