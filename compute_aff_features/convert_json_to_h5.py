import os
import json
import h5py
import numpy as np

# Parameters
json_root = "/fs/nexus-scratch/eloirghi/STEP/openpose_output"
#json_root = "/fs/clip-scratch/eloirghi/STEP/openpose_output"
output_h5 = "/fs/nexus-scratch/eloirghi/STEP/pie.h5"
sequence_length = 40  # Number of frames per sequence
num_joints_total = 25
num_joints_used = 16
num_coords = 3

# Example: COCO joints indices or adjust to match classifier joints mapping
selected_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

with h5py.File(output_h5, "w") as hf:
    ped_folders = sorted([d for d in os.listdir(json_root) if os.path.isdir(os.path.join(json_root, d))])

    for ped_folder in ped_folders:
        json_dir = os.path.join(json_root, ped_folder, "json")
        if not os.path.exists(json_dir):
            print(f"⚠️ Skipping {ped_folder}: no json directory found.")
            continue

        json_files = sorted([f for f in os.listdir(json_dir) if f.endswith("_keypoints.json")])
        valid_frames = []

        for file in json_files:
            json_path = os.path.join(json_dir, file)

            if os.path.getsize(json_path) == 0:
                print(f"⚠️ Skipping empty file: {json_path}")
                continue

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping corrupt JSON file: {json_path}")
                continue

            if not data["people"]:
                continue
            keypoints = data["people"][0]["pose_keypoints_2d"]
            if len(keypoints) != num_joints_total * num_coords:
                continue
            keypoints_arr = np.array(keypoints).reshape(num_joints_total, num_coords)
            selected_keypoints = keypoints_arr[selected_joints, :]
            valid_frames.append(selected_keypoints)


        if len(valid_frames) < sequence_length:
            print(f"⚠️ Skipping {ped_folder}: only {len(valid_frames)} valid frames (needs at least {sequence_length}).")
            continue

        seq = np.stack(valid_frames[:sequence_length])  # shape: [T, 16, 3]
        hf.create_dataset(ped_folder, data=seq)  # ✅ use original folder name as dataset name
        print(f"✅ Saved sequence for {ped_folder} as {ped_folder}")

print(f"🎉 Done. Saved sequences to {output_h5}")
