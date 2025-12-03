#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --job-name=testing_screenspot_pro_Full_Gaussian
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:4
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j_base_dast.out
#SBATCH --error=%x-%j_base_dast.err

bash test_ss_pro.sh