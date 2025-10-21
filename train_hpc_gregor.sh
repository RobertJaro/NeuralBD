#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100.80gb:1
#SBATCH --output=/cl_tmp/schirnin/logs/log%j.log     # Standard output and error log
#SBATCH --mem=128G

cd /usr/people/EDVZ/schirnin
python -m nbd.train_nbd --config '/usr/people/EDVZ/schirnin/configs/gregor.yaml'
# python -m nbd.train_nbd --config '/usr/people/EDVZ/schirnin/configs/muram.yaml'