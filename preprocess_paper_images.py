import argparse
import json
import hashlib
import numpy as np
from pathlib import Path

from preprocessing import (
    gaussian_kernel,
    divisive_dog,
    normalize_standard_deviation
)

BASE = Path("data/paper_van_hateren")
RAW = BASE / "raw"
PROCESSED = BASE / "processed"
METADATA = BASE / "metadata"

CENTER = gaussian_kernel(1.0, 4)
SURROUND = gaussian_kernel(1.5, 6)
DIVISIVE = gaussian_kernel(1.5, 6)


def process_image(filename):
    source = RAW / filename
    destination = PROCESSED / filename.replace(".iml", ".npy")

    if destination.exists():
        raise FileExistsError(
            f"Processed file already exists: {destination}"
        )

    image = np.fromfile(
        source, dtype=">u2"
    ).reshape(1024, 1536).astype(np.float64)

    filtered = divisive_dog(
        image, CENTER, SURROUND, DIVISIVE
    )

    normalized = normalize_standard_deviation(filtered)

    assert normalized.shape == (1024, 1536)
    assert np.isfinite(normalized).all()
    assert np.isclose(normalized.std(), 1.0, atol=1e-10)

    output = normalized.astype(np.float32)

    assert np.isfinite(output).all()
    assert np.isclose(output.std(), 1.0, atol=1e-5)

    np.save(destination, output)

    digest = hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()

    return {
        "filename": filename,
        "output_file": destination.name,
        "mean": float(output.mean(dtype=np.float64)),
        "std": float(output.std(dtype=np.float64)),
        "minimum": float(output.min()),
        "maximum": float(output.max()),
        "sha256": digest
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    manifest = json.loads(
        (METADATA / "selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    filenames = manifest["filenames"]

    if not 1 <= args.limit <= len(filenames):
        raise ValueError("Invalid limit")

    PROCESSED.mkdir(parents=True, exist_ok=True)

    records = []

    for filename in filenames[:args.limit]:
        result = process_image(filename)
        records.append(result)

        print(
            filename,
            "SD:", round(result["std"], 6),
            "Mean:", round(result["mean"], 6),
            "PASSED",
            flush=True
        )

    output = METADATA / "preprocessing_pilot.json"

    if output.exists():
        raise FileExistsError(
            f"Pilot record already exists: {output}"
        )

    output.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8"
    )

    print("\nPREPROCESSING PILOT: PASSED")
    print("Images processed:", len(records))


if __name__ == "__main__":
    main()
