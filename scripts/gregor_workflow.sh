#!/usr/bin/env bash
set -euo pipefail

nbd-prepare-gregor --input /glade/work/cschirninger/data/hifi_20220602_095015_sd.fts --output /glade/work/rjarolim/neuralbd/data/gregor_hifi/processed_burst.npz --axis-order auto --hdu-start 0 --subframe-size 512 --subframe-x 512 --subframe-y 512

nbd-train --config /glade/u/home/rjarolim/projects/NeuralBD/examples/configs/gregor_hifi.yaml

nbd-evaluate --run /glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi
