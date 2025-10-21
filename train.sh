#!/bin/bash
#
#
#SBATCH -J neuralbd                           #use -N only if you use both GPUs on the nodes, otherwise leave this line out
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --ntasks-per-node=64
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1                   #or --gres=gpu:1 if you only want to use half a node
#SBATCH --output=/gpfs/data/fs71254/schirni/logs/log%j.log     # Standard output and error log

cd /home/fs71254/schirni
python -m nbd.train_nbd --config '/home/fs71254/schirni/configs/gregor.yaml'
# python -m nbd.train_nbd --config '/home/fs71254/schirni/configs/muram.yaml'
# python -m nbd.train_nbd --config '/home/fs71254/schirni/configs/dkist.yaml'