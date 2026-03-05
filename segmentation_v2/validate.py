"""Dataset validation for SKW segmentation v2.

Checks image dimensions, readability, and label format.
Results are cached to .dataset_validation.json — skipped if dataset unchanged.
"""

import json
import hashlib
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from shared.data import load_labels
from data import _collect_items


def print_dataset_summary(data_dir: Path):
    """Print split counts for a dataset (auto-detects format)."""
    items = _collect_items(data_dir)
    counts: Counter[str] = Counter()
    for _, _, is_val in items:
        counts["val" if is_val else "train"] += 1
    print(
        f"Dataset: {len(items)} images — "
        f"Train: {counts.get('train', 0)}, Val: {counts.get('val', 0)}"
    )
    return items


def validate_dataset(
    data_dir: Path,
    canvas_size: int = 600,
    cache_path: Path = Path(".dataset_validation.json"),
) -> set[str]:
    """Validate all images and return stems to exclude.

    Checks for unreadable, non-square, oversized images and OOB bboxes.
    Results are cached based on a fingerprint of the image list.
    """
    items = _collect_items(data_dir)
    all_items = [(img, lbl) for img, lbl, _ in items]

    # Fingerprint: hash of sorted image paths + count
    fp_str = (
        str(len(all_items))
        + "|"
        + str(sorted(p.name for p, _ in all_items[:100] + all_items[-100:]))
    )
    fingerprint = hashlib.md5(fp_str.encode()).hexdigest()

    bad_images = []
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("fingerprint") == fingerprint:
            bad_images = [(Path(e[0]), Path(e[1]), e[2]) for e in cache["bad_images"]]
            print(
                f"Dataset unchanged \u2014 loaded {len(bad_images)} bad images from cache"
            )
        else:
            cache = None
    else:
        cache = None

    if cache is None:

        def _check_item(img_path, lbl_path):
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                return (str(img_path), str(lbl_path), f"unreadable: {e}")
            if h != w:
                return (str(img_path), str(lbl_path), f"non-square: {w}x{h}")
            if h > 2 * canvas_size:
                return (str(img_path), str(lbl_path), f"oversized: {w}x{h}")
            if lbl_path.exists():
                try:
                    _, bboxes = load_labels(lbl_path)
                    if len(bboxes) > 0 and ((bboxes < 0).any() or (bboxes > 1.5).any()):
                        return (
                            str(img_path),
                            str(lbl_path),
                            f"bbox OOB: [{bboxes.min():.2f}, {bboxes.max():.2f}]",
                        )
                except Exception as e:
                    return (str(img_path), str(lbl_path), f"bad labels: {e}")
            return None

        t0 = time.time()
        print(f"Validating {len(all_items)} images (threaded)...")
        bad_raw = []
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {
                pool.submit(_check_item, ip, lp): (ip, lp) for ip, lp in all_items
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    bad_raw.append(result)

        cache_path.write_text(
            json.dumps({"fingerprint": fingerprint, "bad_images": bad_raw})
        )
        bad_images = [(Path(e[0]), Path(e[1]), e[2]) for e in bad_raw]
        print(f"Done in {time.time() - t0:.1f}s \u2014 cached to {cache_path}")

    print(f"{len(bad_images)} bad images found")
    for img, _, reason in bad_images:
        print(f"  {reason}: {img.name}")

    exclude_stems = {img.stem for img, _, _ in bad_images}
    print(f"exclude_stems: {len(exclude_stems)} entries")
    return exclude_stems
