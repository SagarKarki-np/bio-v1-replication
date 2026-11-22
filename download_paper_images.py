import argparse
import hashlib
import json
import time
import zlib
from pathlib import Path

from remotezip import RemoteZip

BASE = Path("data/paper_van_hateren")
RAW_DIR = BASE / "raw"
META_DIR = BASE / "metadata"
MANIFEST = META_DIR / "selection_manifest.json"
HASH_LOG = META_DIR / "download_hashes.jsonl"

EXPECTED_SIZE = 1024 * 1536 * 2


def verify_file(path, expected_crc):
    sha = hashlib.sha256()
    crc = 0
    size = 0

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)

    if size != EXPECTED_SIZE:
        raise ValueError(f"Invalid size: {path.name}")

    if crc != expected_crc:
        raise ValueError(f"CRC mismatch: {path.name}")

    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = manifest["filenames"]

    assert len(selected) == 1000
    assert len(set(selected)) == 1000

    if args.limit is not None:
        if not 1 <= args.limit <= len(selected):
            raise ValueError("Invalid --limit")
        selected = selected[:args.limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    with RemoteZip(manifest["archive_url"]) as archive:
        for index, filename in enumerate(selected, 1):
            destination = RAW_DIR / filename
            temporary = destination.with_suffix(".iml.part")
            expected_crc = archive.getinfo(filename).CRC

            if destination.exists():
                digest = verify_file(destination, expected_crc)
                print(f"[{index}/{len(selected)}] VERIFIED EXISTING {filename}", flush=True)
            else:
                for attempt in range(1, 4):
                    try:
                        temporary.unlink(missing_ok=True)

                        with archive.open(filename) as source, temporary.open("wb") as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)

                        digest = verify_file(temporary, expected_crc)
                        temporary.replace(destination)

                        print(f"[{index}/{len(selected)}] DOWNLOADED {filename}", flush=True)
                        break

                    except Exception as exc:
                        temporary.unlink(missing_ok=True)
                        print(f"Attempt {attempt} failed: {filename}: {exc}", flush=True)
                        if attempt == 3:
                            raise
                        time.sleep(2 * attempt)

            record = {
                "filename": filename,
                "sha256": digest,
                "size_bytes": EXPECTED_SIZE,
                "zip_crc32": f"{expected_crc:08x}"
            }

            with HASH_LOG.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")

    print("\nDOWNLOAD AND VERIFICATION COMPLETED")


if __name__ == "__main__":
    main()
