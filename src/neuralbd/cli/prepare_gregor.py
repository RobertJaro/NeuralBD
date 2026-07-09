import argparse
import json
import logging
from pathlib import Path

import numpy as np

from neuralbd.data import load_burst_from_config
from neuralbd.processing import align_stack


DEFAULT_INPUT = "/glade/work/cschirninger/data/hifi_20220602_095015_sd.fts"
LOGGER = logging.getLogger("neuralbd.prepare_gregor")


def _parse_int_list(value):
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_range(value):
    if value is None:
        return None
    start, stop = value.split(":", 1)
    return [int(start), int(stop)]


def _subframe_from_args(args):
    subframe = {}
    if args.subframe_size is not None:
        subframe["size"] = args.subframe_size
    if args.subframe_x is not None:
        subframe["x"] = args.subframe_x
    if args.subframe_y is not None:
        subframe["y"] = args.subframe_y
    if args.x_range is not None:
        subframe["x_range"] = _parse_range(args.x_range)
    if args.y_range is not None:
        subframe["y_range"] = _parse_range(args.y_range)
    return subframe or None


def _frame_slice_from_args(args):
    if args.frame_start is None and args.frame_stop is None and args.frame_step is None:
        return None
    return {"start": args.frame_start, "stop": args.frame_stop, "step": args.frame_step}


def build_loader_config(args):
    return {
        "type": "gregor",
        "path": str(args.input),
        "axis_order": args.axis_order,
        "hdu_indices": _parse_int_list(args.hdu_indices),
        "hdu_start": args.hdu_start,
        "hdu_stop": args.hdu_stop,
        "source_channels": args.source_channels,
        "source_channel": args.source_channel,
        "frame_indices": _parse_int_list(args.frame_indices),
        "frame_slice": _frame_slice_from_args(args),
        "n_images": args.n_images,
        "channels": _parse_int_list(args.channels),
        "subframe": _subframe_from_args(args),
    }


def _buffer_subframe_for_alignment(subframe_config, max_shift):
    max_shift = int(max_shift or 0)
    if max_shift < 0:
        raise ValueError("alignment max shift must be non-negative")
    if not subframe_config or max_shift == 0:
        return subframe_config, (0, 0)

    buffered = dict(subframe_config)
    if "size" in buffered:
        buffered["size"] = int(buffered["size"]) + 2 * max_shift
        return buffered, (max_shift, max_shift)

    trim_x = 0
    trim_y = 0
    if "x_range" in buffered:
        x0, x1 = buffered["x_range"]
        buffered["x_range"] = [int(x0) - max_shift, int(x1) + max_shift]
        trim_x = max_shift
    if "y_range" in buffered:
        y0, y1 = buffered["y_range"]
        buffered["y_range"] = [int(y0) - max_shift, int(y1) + max_shift]
        trim_y = max_shift
    return buffered, (trim_x, trim_y)


def build_alignment_loader_config(loader_config, max_shift):
    alignment_config = dict(loader_config)
    subframe_config, trim_buffer = _buffer_subframe_for_alignment(loader_config.get("subframe"), max_shift)
    alignment_config["subframe"] = subframe_config
    return alignment_config, trim_buffer


def trim_alignment_buffer(images, trim_buffer):
    trim_x, trim_y = [int(value) for value in trim_buffer]
    if trim_x < 0 or trim_y < 0:
        raise ValueError("alignment trim buffer must be non-negative")
    if trim_x == 0 and trim_y == 0:
        return images
    if images.shape[0] <= 2 * trim_x or images.shape[1] <= 2 * trim_y:
        raise ValueError("Alignment buffer is too large for the prepared burst shape")
    x_slice = slice(trim_x, -trim_x if trim_x else None)
    y_slice = slice(trim_y, -trim_y if trim_y else None)
    return images[x_slice, y_slice, ...]


def _clean_config(config):
    return {key: value for key, value in config.items() if value is not None}


def build_metadata(args, loader_config, images, shifts):
    max_shift = int(getattr(args, "alignment_max_shift", 0) or 0)
    alignment = {
        "method": "bounded-normalized-cross-correlation",
        "reference_frame": 0,
        "max_shift": max_shift,
        "shifts": shifts,
    }
    if max_shift:
        alignment["max_shift_buffer"] = max_shift
    return {
        "format": "neuralbd-prepared-burst",
        "format_version": 1,
        "instrument": "GREGOR/HIFI",
        "source_path": str(args.input),
        "layout": "height,width,frames,channels",
        "array_key": "images",
        "metadata_key": "metadata",
        "shape": [int(value) for value in images.shape],
        "dtype": str(images.dtype),
        "preparation": _clean_config(loader_config),
        "alignment": alignment,
    }


def save_prepared_burst(path, images, metadata):
    np.savez_compressed(
        path,
        images=np.asarray(images, dtype="float32"),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def configure_logging(verbose=False, quiet=False):
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def _should_log_progress(current, total):
    if current is None or total is None or total <= 20:
        return True
    interval = max(1, total // 20)
    return current == 1 or current == total or current % interval == 0


def _log_prepare_progress(stage, current, total, detail):
    if stage == "load_start":
        LOGGER.info("Loading burst from %s", detail)
    elif stage == "load_hdu":
        if _should_log_progress(current, total):
            LOGGER.info("Loading HDU %s/%s (index %s)", current, total, detail)
    elif stage == "load_done":
        LOGGER.info("Loaded burst with shape %s", detail)
    elif stage == "select_frames_done":
        LOGGER.info("After frame selection: shape %s", detail)
    elif stage == "select_channels_done":
        LOGGER.info("After channel selection: shape %s", detail)
    elif stage == "subframe_done":
        LOGGER.info("After subframe crop: shape %s", detail)
    elif stage == "align_start":
        LOGGER.info(
            "Alignment search: %s frames, %s candidate shifts per frame, workers=%s",
            total,
            detail["candidate_count"],
            detail["num_workers"],
        )
    elif stage == "align_frame_start":
        if _should_log_progress(current, total):
            LOGGER.info("Searching alignment for frame %s/%s (index %s)", current, total, detail["frame_index"])
    elif stage == "align_frame_done":
        if _should_log_progress(current, total):
            LOGGER.info(
                "Aligned frame %s/%s (index %s): shift=%s, score=%.5f",
                current,
                total,
                detail["frame_index"],
                detail["shift"],
                detail["score"],
            )
    elif stage == "align_done":
        LOGGER.info("Finished alignment for %s frames", total)
    elif stage == "alignment_buffer_done":
        LOGGER.info("Removed alignment buffer: shape %s", detail)
    else:
        LOGGER.debug("Progress stage %s: current=%s total=%s detail=%s", stage, current, total, detail)


def prepare_gregor(args):
    LOGGER.info("Preparing GREGOR/HIFI burst")
    loader_config = build_loader_config(args)
    alignment_loader_config, trim_buffer = build_alignment_loader_config(loader_config, args.alignment_max_shift)
    if trim_buffer != (0, 0):
        LOGGER.info("Using alignment buffer: x=%s px, y=%s px", trim_buffer[0], trim_buffer[1])
    images = load_burst_from_config(alignment_loader_config, progress_callback=_log_prepare_progress)
    LOGGER.info("Converting burst to float32")
    images = np.asarray(images, dtype="float32")
    LOGGER.info(
        "Aligning %s frames with bounded cross-correlation (max shift: %s px, workers: %s)",
        images.shape[2],
        args.alignment_max_shift,
        args.alignment_workers or "auto",
    )
    images, shifts = align_stack(
        images,
        max_shift=args.alignment_max_shift,
        num_workers=args.alignment_workers,
        progress_callback=_log_prepare_progress,
    )
    images = trim_alignment_buffer(images, trim_buffer)
    if trim_buffer != (0, 0):
        _log_prepare_progress("alignment_buffer_done", None, None, tuple(images.shape))
    shift_limits = np.asarray(shifts, dtype=int)
    metadata = build_metadata(args, loader_config, images, shifts)
    LOGGER.info("Creating output directory: %s", args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing compressed prepared burst: %s", args.output)
    save_prepared_burst(args.output, images, metadata)
    LOGGER.info("Finished data preparation")

    print(f"Wrote processed burst: {args.output}")
    print(f"Shape: {tuple(images.shape)}")
    print(
        "Aligned stack with bounded cross-correlation shifts: "
        f"min={shift_limits.min(axis=0).tolist()}, max={shift_limits.max(axis=0).tolist()}"
    )
    print("Keys: images, metadata")
    print("Use this path in a NeuralBD training config:")
    print(f"  data.path: {args.output}")
    print("  data.type: npz")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a GREGOR/HIFI FITS burst for NeuralBD training.")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT), help="GREGOR/HIFI .fits or .fts file.")
    parser.add_argument("--output", type=Path, default=Path("runs/gregor_hifi/work/data/processed_burst.npz"))

    parser.add_argument("--axis-order", default="auto", help="Input FITS cube layout, e.g. auto, fyx, yxf, fyxc, yxfc.")
    parser.add_argument("--hdu-indices", help="Comma-separated HDU indices. Overrides hdu-start/hdu-stop.")
    parser.add_argument("--hdu-start", type=int, default=0)
    parser.add_argument("--hdu-stop", type=int)
    parser.add_argument(
        "--source-channels",
        type=int,
        default=2,
        help="Number of interleaved source channels in a GREGOR 2D HDU stack.",
    )
    parser.add_argument("--source-channel", type=int, help="Select one channel before frame/channel filtering.")
    parser.add_argument("--frame-indices", help="Comma-separated frame indices.")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--frame-step", type=int)
    parser.add_argument("--n-images", type=int)
    parser.add_argument("--channels", help="Comma-separated channel indices.")
    parser.add_argument("--subframe-x", type=int)
    parser.add_argument("--subframe-y", type=int)
    parser.add_argument("--subframe-size", type=int)
    parser.add_argument("--x-range", help="Subframe x range as start:stop.")
    parser.add_argument("--y-range", help="Subframe y range as start:stop.")
    parser.add_argument(
        "--alignment-max-shift",
        "--max-shift",
        type=int,
        default=20,
        help="Maximum expected alignment shift in pixels; expands the subframe before alignment and trims it after.",
    )
    parser.add_argument(
        "--alignment-workers",
        type=int,
        help="Number of worker processes for frame alignment. Defaults to the available CPU count.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors.")

    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    prepare_gregor(args)


if __name__ == "__main__":
    main()
