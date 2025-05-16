import os
import h5py
import csv
import numpy as np
import argparse
import torch
from utils import processor

# === CONFIG ===
coords = 3
joints = 16
cycles = 1
device = 'cuda:0'
emotions = ['Angry', 'Neutral', 'Happy', 'Sad']

# PIE file and model paths
pie_path = '/fs/nexus-scratch/eloirghi/STEP/'
#h5_file = os.path.join(pie_path, 'pie_affective_features.h5')
h5_file = '/fs/nexus-scratch/eloirghi/STEP/pie.h5'  # raw pose sequences
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_classifier_stgcn/features')  # or features4DCVAEGCN

# === ARGS ===
parser = argparse.ArgumentParser(description='ST-GCN for emotion classification')
parser.add_argument('--work-dir', type=str, default=model_path)
parser.add_argument('--model_saved_name', default='', help='Name of the model for saving')
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--test-batch-size', type=int, default=6)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--save-score', type=bool, default=False)
parser.add_argument('--print-log', type=bool, default=True)
parser.add_argument('--save-log', type=bool, default=False)
parser.add_argument('--smap', action='store_true', default=False)
parser.add_argument('--save-features', action='store_true', default=False)
parser.add_argument('--step', nargs='+', type=float, default=[0.5, 0.75])
parser.add_argument('--base-lr', type=float, default=0.01)
parser.add_argument('--lr-decay-rate', type=float, default=0.1)
parser.add_argument('--lr-decay-step', type=int, default=20)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--weight-decay', type=float, default=0.0001)
parser.add_argument('--nesterov-momentum', type=float, default=0.9)
parser.add_argument('--topk', nargs='+', type=int, default=[1])
args = parser.parse_args()

# === LOAD PIE FEATURES ===
print("Loading PIE affective features from:", h5_file)
with h5py.File(h5_file, 'r') as f:
    keys = list(f.keys())
    print(f"Number of sequences: {len(keys)}")
    num_samples = len(keys)
    T = 40  # fixed sequence length expected by ST-GCN
    data = np.zeros((num_samples, T, joints, coords, 1))
    for i, k in enumerate(keys):
        d = np.array(f[k])
        print(f"Sample {i}: {k} → shape {d.shape}, min={d.min():.4f}, max={d.max():.4f}, mean={d.mean():.4f}, std={d.std():.4f}")
        if d.shape != (T, joints, coords):
            raise ValueError(f"Unexpected shape {d.shape} for key {k}")
        data[i, :, :, :, 0] = d  # ✅ directly assign
    labels = np.zeros(num_samples)  # dummy labels for inference
print(f"Loading done. Data shape: {data.shape}")

# === CLASSIFICATION ===
num_classes = len(emotions)
graph_dict = {'strategy': 'spatial'}

pr = processor.Processor(args, None, coords, num_classes, graph_dict, device=device, verbose=False)

# === LOAD TRAINED MODEL ===
print(f"Loading best model from: {args.work_dir}")
pr.load_best_model()

# === GENERATE PREDICTIONS ===
labels_pred, vecs_pred = pr.generate_predictions(data, num_classes, joints, coords)

print(f"vecs_pred.shape: {vecs_pred.shape}")

# === PRINT + SAVE RESULTS ===
print("\nPredicted Emotions with Probabilities:")
csv_file = os.path.join(os.path.dirname(h5_file), "predictions.csv")
with open(csv_file, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    header = ["ped_id", "predicted_emotion"] + [f"prob_{e.lower()}" for e in emotions]
    writer.writerow(header)
    for idx in range(labels_pred.shape[0]):
        probs = vecs_pred[idx]
        probs_str = ", ".join([f"{emotions[i]}: {probs[i]:.4f}" for i in range(len(emotions))])
        print(f"{idx:2d}.\tPredicted: {emotions[int(labels_pred[idx])]:<8}\tProbabilities: {probs_str}")
        row = [keys[idx], emotions[int(labels_pred[idx])]] + [f"{probs[i]:.4f}" for i in range(len(emotions))]
        writer.writerow(row)

print(f"\n✅ Predictions saved to: {csv_file}")

if args.smap:
    pr.smap()
if args.save_features:
    pr.save_best_feature("PIE", data, joints, coords)

print("✅ Done.")
