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

from shared.data import parse_splits, load_labels
from data import collect_v2_items


def print_dataset_summary(data_dir: Path, data_v2_dir: Path | None = None):
    """Print split counts for original + v2 datasets."""
    split_map = parse_splits(data_dir)
    counts = Counter(split_map.values())
    seen_stems = set(split_map.keys())

    v2_items = []
    if data_v2_dir is not None:
        v2_items = collect_v2_items(data_v2_dir)
        for img_path, _, split in v2_items:
            if img_path.stem not in seen_stems:
                counts[split] += 1
                seen_stems.add(img_path.stem)

    print(f"Original:  {sum(Counter(parse_splits(data_dir).values()).values())} images")
    if data_v2_dir is not None:
        print(
            f"+ data_v2: {len(v2_items)} images"
            f" ({len(v2_items) - len(parse_splits(data_dir)):+d} new)"
        )
    print(
        f"Combined:  Train: {counts.get('train', 0)},"
        f" Val: {counts.get('val', 0)}, Test: {counts.get('test', 0)}"
    )
    print(f"Total: {sum(counts.values())}")
    return v2_items


def validate_dataset(
    data_dir: Path,
    data_v2_dir: Path | None = None,
    canvas_size: int = 600,
    cache_path: Path = Path(".dataset_validation.json"),
) -> set[str]:
    """Validate all images and return stems to exclude.

    Checks for unreadable, non-square, oversized images and OOB bboxes.
    Results are cached based on a fingerprint of the image list.
    """
    split_map = parse_splits(data_dir)
    seen = set(split_map.keys())
    all_items = []

    img_dir = data_dir / "images"
    if img_dir.exists():
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                all_items.append((f, data_dir / "labels" / f"{f.stem}.txt"))
                seen.add(f.stem)

    if data_v2_dir is not None:
        v2_items = collect_v2_items(data_v2_dir)
        for img_path, lbl_path, _ in v2_items:
            if img_path.stem not in seen:
                all_items.append((img_path, lbl_path))
                seen.add(img_path.stem)

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
