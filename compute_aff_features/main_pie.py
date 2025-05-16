import h5py
import os

from compute_aff_features.compute_features import compute_features
from compute_aff_features.normalize_features import normalize_features
from compute_aff_features.cross_validate import cross_validate

# Hardcoded paths
input_path = '/fs/nexus-scratch/eloirghi/STEP/pie.h5'
output_path = '/fs/nexus-scratch/eloirghi/STEP/pie_affective_features.h5'
labels_path = '/fs/nexus-scratch/eloirghi/STEP/data/labels.h5'  # adjust this if needed

# Read pose sequences
positions = h5py.File(input_path, 'r')
keys = positions.keys()
time_step = 1.0 / 30.0

# Compute features
print('Computing Features ... ', end='')
features = []
for key in keys:
    frames = positions[key]
    feature = [key]
    if frames.ndim == 2:
        feature += compute_features(frames, time_step)
    else:
        feature += frames
    features.append(feature)
print('done.')

# Normalize features
print('Normalizing Features ... ', end='')
normalized_features = []
normalize_features(features, normalized_features)
print('done.')

# Save features to HDF5
print('Saving features ... ', end='')
with h5py.File(output_path, 'w') as aff:
    for feature in normalized_features:
        aff.create_dataset(feature[0], data=feature[1:])
print('done.')

# Uncomment the following lines if you want to load labels and run cross-validation
# for our case, we obviously dont have labels for pie data so this just creates problems
# Load labels
#labels = h5py.File(labels_path, 'r')

# Run cross-validation
#cross_validate(normalized_features, labels)
