import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from preprocessing import (
    gaussian_kernel,
    divisive_dog,
    normalize_standard_deviation
)

BASE = Path("data/paper_van_hateren")
RAW = BASE / "raw"
PROCESSED = BASE / "processed"
METADATA = BASE / "metadata"

MANIFEST = METADATA / "selection_manifest.json"
LOG = METADATA / "preprocessing_records.jsonl"

SHAPE = (1024, 1536)
RAW_SIZE = 1024 * 1536 * 2

CENTER = gaussian_kernel(1.0, 4)
SURROUND = gaussian_kernel(1.5, 6)
DIVISIVE = gaussian_kernel(1.5, 6)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def verify_processed(path):
    image = np.load(path, mmap_mode="r", allow_pickle=False)

    if image.shape != SHAPE:
        raise ValueError(f"Incorrect shape: {path.name}")

    if image.dtype != np.float32:
        raise ValueError(f"Incorrect dtype: {path.name}")

    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite values: {path.name}")

    std = float(image.std(dtype=np.float64))

    if not np.isclose(std, 1.0, atol=1e-5):
        raise ValueError(f"Incorrect SD: {path.name}: {std}")

    return {
        "output_file": path.name,
        "mean": float(image.mean(dtype=np.float64)),
        "std": std,
        "minimum": float(image.min()),
        "maximum": float(image.max()),
        "processed_sha256": sha256_file(path)
    }


def process_one(filename):
    source = RAW / filename
    destination = PROCESSED / filename.replace(".iml", ".npy")

    if not source.is_file():
        raise FileNotFoundError(source)

    if source.stat().st_size != RAW_SIZE:
        raise ValueError(f"Invalid raw file size: {filename}")

    raw_hash = sha256_file(source)

    if destination.exists():
        result = verify_processed(destination)
        status = "VERIFIED EXISTING"

    else:
        image = np.fromfile(source, dtype=">u2")
        image = image.reshape(SHAPE).astype(np.float64)

        filtered = divisive_dog(
            image, CENTER, SURROUND, DIVISIVE
        )

        normalized = normalize_standard_deviation(filtered)

        if not np.isfinite(normalized).all():
            raise ValueError(f"Non-finite result: {filename}")

        if not np.isclose(normalized.std(), 1.0, atol=1e-10):
            raise ValueError(f"Invalid normalized SD: {filename}")

        output = normalized.astype(np.float32)

        temporary = destination.with_suffix(".npy.part")

        try:
            with temporary.open("wb") as f:
                np.save(f, output)

            result = verify_processed(temporary)

            os.replace(temporary, destination)

        finally:
            temporary.unlink(missing_ok=True)

        result["output_file"] = destination.name
        status = "PROCESSED"

    return {
        "source_file": filename,
        "raw_sha256": raw_hash,
        **result
    }, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    selected = manifest["filenames"]

    if len(selected) != 1000 or len(set(selected)) != 1000:
        raise ValueError("Invalid selection manifest")

    if not 1 <= args.limit <= len(selected):
        raise ValueError("Invalid limit")

    PROCESSED.mkdir(parents=True, exist_ok=True)

    for index, filename in enumerate(selected[:args.limit], 1):
        record, status = process_one(filename)

        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"[{index}/{args.limit}] {status} "
            f"{filename} SD={record['std']:.6f}",
            flush=True
        )

    print("\nPREPROCESSING RUN COMPLETED")


if __name__ == "__main__":
    main()
