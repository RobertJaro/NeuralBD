from pathlib import Path

import numpy as np


def main():
    out_path = Path("examples/data/synthetic_burst.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size = 32
    n_frames = 4
    y, x = np.mgrid[:size, :size].astype("float32")
    base = (
        np.exp(-((x - 11) ** 2 + (y - 15) ** 2) / 45)
        + 0.7 * np.exp(-((x - 22) ** 2 + (y - 18) ** 2) / 30)
        + 0.15 * np.sin(x / 2)
    )
    base = (base - base.min()) / (base.max() - base.min())

    rng = np.random.default_rng(0)
    frames = []
    for idx in range(n_frames):
        shifted = np.roll(base, shift=idx - 1, axis=0)
        shifted = np.roll(shifted, shift=1 - idx, axis=1)
        frames.append(shifted + rng.normal(0, 0.01, shifted.shape))
    burst = np.stack(frames, axis=-1).astype("float32")
    np.save(out_path, burst)
    print(out_path)


if __name__ == "__main__":
    main()
