#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --job-name=train_e
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:4
#SBATCH --mem=500GB
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

bash train_e.sh

