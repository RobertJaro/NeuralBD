import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def integer_shift(image, shift_x, shift_y):
    shifted = np.zeros_like(image)
    height, width = image.shape[:2]

    dst_x0, dst_x1 = max(0, shift_x), min(height, height + shift_x)
    dst_y0, dst_y1 = max(0, shift_y), min(width, width + shift_y)
    src_x0, src_x1 = max(0, -shift_x), min(height, height - shift_x)
    src_y0, src_y1 = max(0, -shift_y), min(width, width - shift_y)
    shifted[dst_x0:dst_x1, dst_y0:dst_y1, ...] = image[src_x0:src_x1, src_y0:src_y1, ...]
    return shifted


def _registration_image(image):
    image = np.asarray(image, dtype="float32")
    if image.ndim == 3:
        image = np.nanmean(image, axis=-1)
    image = np.nan_to_num(image - np.nanmean(image))
    return image


def _overlap_slices(shape, shift_x, shift_y):
    height, width = shape[:2]
    dst_x0, dst_x1 = max(0, shift_x), min(height, height + shift_x)
    dst_y0, dst_y1 = max(0, shift_y), min(width, width + shift_y)
    src_x0, src_x1 = max(0, -shift_x), min(height, height - shift_x)
    src_y0, src_y1 = max(0, -shift_y), min(width, width - shift_y)
    return (
        slice(dst_x0, dst_x1),
        slice(dst_y0, dst_y1),
        slice(src_x0, src_x1),
        slice(src_y0, src_y1),
    )


def _normalized_cross_correlation(reference, image, shift_x, shift_y):
    dst_x, dst_y, src_x, src_y = _overlap_slices(reference.shape, shift_x, shift_y)
    reference_overlap = reference[dst_x, dst_y]
    image_overlap = image[src_x, src_y]
    if reference_overlap.size == 0 or image_overlap.size == 0:
        return -np.inf

    valid = np.isfinite(reference_overlap) & np.isfinite(image_overlap)
    if not np.any(valid):
        return -np.inf
    reference_values = reference_overlap[valid]
    image_values = image_overlap[valid]
    reference_values = reference_values - np.mean(reference_values)
    image_values = image_values - np.mean(image_values)
    denominator = np.linalg.norm(reference_values) * np.linalg.norm(image_values)
    if denominator <= 1e-12:
        return -np.inf
    return float(np.sum(reference_values * image_values) / denominator)


def bounded_cross_correlation_shift(reference, image, max_shift):
    reference = _registration_image(reference)
    image = _registration_image(image)
    max_shift = int(max_shift)
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    best_shift = (0, 0)
    best_score = -np.inf
    best_distance = 0
    for shift_x in range(-max_shift, max_shift + 1):
        for shift_y in range(-max_shift, max_shift + 1):
            score = _normalized_cross_correlation(reference, image, shift_x, shift_y)
            distance = abs(shift_x) + abs(shift_y)
            if score > best_score or (score == best_score and distance < best_distance):
                best_score = score
                best_shift = (shift_x, shift_y)
                best_distance = distance
    return int(best_shift[0]), int(best_shift[1]), float(best_score)


def phase_shift(reference, image, max_shift=20):
    shift_x, shift_y, _score = bounded_cross_correlation_shift(reference, image, max_shift=max_shift)
    return shift_x, shift_y


def _align_frame_job(reference, frame, frame_idx, max_shift):
    shift_x, shift_y, score = bounded_cross_correlation_shift(reference, frame, max_shift=max_shift)
    return frame_idx, integer_shift(frame, shift_x, shift_y), [shift_x, shift_y], score


def align_stack(images, reference_index=0, max_shift=0, num_workers=1, progress_callback=None):
    images = np.asarray(images, dtype="float32")
    aligned = np.empty_like(images)
    reference = images[:, :, int(reference_index), ...]
    max_shift = int(max_shift or 0)
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, int(num_workers))

    n_frames = images.shape[2]
    candidate_count = (2 * max_shift + 1) ** 2
    if progress_callback is not None:
        progress_callback(
            "align_start",
            0,
            n_frames,
            {"max_shift": max_shift, "candidate_count": candidate_count, "num_workers": min(num_workers, n_frames)},
        )
    shifts = [None] * n_frames
    if num_workers == 1 or n_frames <= 1:
        for frame_idx in range(n_frames):
            if progress_callback is not None:
                progress_callback("align_frame_start", frame_idx + 1, n_frames, {"frame_index": frame_idx})
            result = _align_frame_job(reference, images[:, :, frame_idx, ...], frame_idx, max_shift)
            result_idx, aligned_frame, shift, _score = result
            aligned[:, :, result_idx, ...] = aligned_frame
            shifts[result_idx] = shift
            if progress_callback is not None:
                progress_callback(
                    "align_frame_done",
                    frame_idx + 1,
                    n_frames,
                    {"frame_index": result_idx, "shift": shift, "score": _score},
                )
    else:
        with ProcessPoolExecutor(max_workers=min(num_workers, n_frames)) as executor:
            futures = [
                executor.submit(_align_frame_job, reference, images[:, :, frame_idx, ...], frame_idx, max_shift)
                for frame_idx in range(n_frames)
            ]
            for done_count, future in enumerate(as_completed(futures), start=1):
                result_idx, aligned_frame, shift, _score = future.result()
                aligned[:, :, result_idx, ...] = aligned_frame
                shifts[result_idx] = shift
                if progress_callback is not None:
                    progress_callback(
                        "align_frame_done",
                        done_count,
                        n_frames,
                        {"frame_index": result_idx, "shift": shift, "score": _score},
                    )
    if progress_callback is not None:
        progress_callback("align_done", n_frames, n_frames, {"shifts": shifts})
    return aligned, shifts
