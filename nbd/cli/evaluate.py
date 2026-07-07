import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize NeuralBD run outputs.")
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    outputs = sorted((run_dir / "outputs").glob("*.npy"))
    print(f"Run: {run_dir}")
    print(f"Output arrays: {len(outputs)}")
    for path in outputs:
        print(path.name)


if __name__ == "__main__":
    main()
