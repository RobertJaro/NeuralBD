import argparse

import numpy as np

from neuralbd.evaluation import NeuralBDOutput


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained NeuralBD image model on stored coordinates.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args(argv)

    output = NeuralBDOutput(args.checkpoint, device="cpu")
    np.save(args.out, output.reconstruct(batch_size=args.batch_size))


if __name__ == "__main__":
    main()
