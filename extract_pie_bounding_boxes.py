import os
import sys
sys.path.append('/fs/nexus-scratch/pasthana/838M/Ours')
sys.path.append('/fs/nexus-scratch/pasthana/838M/PIE')
#sys.path.append('/fs/nexus-scratch/pasthana/838M/PIE/PIE_dataset')

import torch
from torchvision.utils import save_image
from train import CustomDataset2
from extract_human_images import PIE_data

# setup paths
pie_root = '/fs/nexus-scratch/pasthana/838M/PIE/PIE_dataset'
output_root = '/fs/nexus-scratch/eloirghi/STEP/PIE_valid_imgs'
T = 40  # sequence length expected by STEP
os.makedirs(output_root, exist_ok=True)

# use Pranav's CustomDataset class to load the dataset
pie_data = PIE_data(pie_root)
dataset = CustomDataset2(pie_data, T=T, stride=6)

# extract and save the bounding box images
for idx in range(len(dataset)):
    psid, frame_nums, images = dataset[idx]

    # create output dir for this pedestrian sequence
    ped_dir = os.path.join(output_root, psid)
    os.makedirs(ped_dir, exist_ok=True)

    # split and save each frame
    # frames = torch.chunk(bbox_image, T, dim=2)
    for t, frame in zip(frame_nums, images):
        save_path = os.path.join(ped_dir, f"frame_{t:02d}.jpg")
        save_image(frame[torch.newaxis], save_path)

    if idx % 10 == 0:
        print(f"Saved sequence {idx:04d}/{len(dataset)}")

print("✅ All bounding box sequences saved.")
