#!/bin/bash
#SBATCH --job-name=openpose_single
#SBATCH --output=/fs/nexus-scratch/eloirghi/STEP/openpose_%j.out
#SBATCH --error=/fs/nexus-scratch/eloirghi/STEP/openpose_%j.err
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=5G
#SBATCH --time=00:20:00

apptainer shell --nv \
-B /fs/nexus-scratch/pasthana:/fs/nexus-scratch/pasthana \
-B /fs/nexus-scratch/eloirghi:/fs/nexus-scratch/eloirghi \
-B /fs/nexus-scratch/eloirghi/openpose:/openpose \
-B /fs/nexus-scratch/eloirghi/openpose_build_dir:/openpose/build \
/fs/nexus-scratch/eloirghi/openpose_build/openpose_ampere.sif <<EOF

cd /openpose

ped_dir="$PED_DIR"
ped_name=\$(basename "\$ped_dir")
out_dir="/fs/nexus-scratch/eloirghi/STEP/openpose_output/\$ped_name" 

mkdir -p "\$out_dir/json" "\$out_dir/images"

echo "Running OpenPose for \$ped_name..."

./build/examples/openpose/openpose.bin \
    --image_dir "\$ped_dir" \
    --write_json "\$out_dir/json" \
    --write_images "\$out_dir/images" \
    --display 0 \
    --render_pose 1

if [ -d "\$out_dir/json" ] && ls "\$out_dir/json"/*.json 1>/dev/null 2>&1; then
    echo "✅ Output JSONs detected → deleting source folder \$ped_dir"
    rm -rf "\$ped_dir"
else
    echo "⚠️ No output JSONs found → keeping source folder \$ped_dir"
fi

EOF
