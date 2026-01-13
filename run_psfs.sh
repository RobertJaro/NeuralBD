#!/bin/bash
#
#
#SBATCH -J psfs                           #use -N only if you use both GPUs on the nodes, otherwise leave this line out
#SBATCH --partition zen3_1024
#SBATCH --qos zen3_1024
#SBATCH --time=72:00:00
#SBATCH --mem 64G
#SBATCH --output=/gpfs/data/fs71254/schirni/logs/log%j.log     # Standard output and error log

cd /home/fs71254/schirni
python -m nbd.data.spatially_varying_psfs --out_file "/gpfs/data/fs71254/schirni/nstack/data/convolved_image_000.npy"