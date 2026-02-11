#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --job-name=train_e
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:2
#SBATCH --mem=200GB
#SBATCH --time=6:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
module load python/3.11.5 gcc opencv/4.10.0 arrow/21.0.0 cuda/12.6
source ~/scratch/venvs/ui-r1-redo/bin/activate
bash ~/scratch/UI-R1/evaluation/test_ss_pro.sh

