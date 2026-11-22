import json
from pathlib import Path

import numpy as np


class PaperImageDataset:
    """
    Read-only adapter for the locked 1000-image dataset.

    Presents the same indexing interface as an array
    with shape (height, width, n_images).

    Loads one image on demand instead of holding the
    complete image collection in RAM.
    """

    def __init__(self, base_dir="data/paper_van_hateren"):

        self.base = Path(base_dir)
        self.processed = self.base / "processed"

        manifest_path = (
            self.base
            / "metadata"
            / "processed_dataset_manifest.json"
        )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        selection_path = (
            self.base
            / "metadata"
            / "selection_manifest.json"
        )

        selection = json.loads(
            selection_path.read_text(encoding="utf-8")
        )

        if manifest["selection_sha256"] != selection["selection_sha256"]:
            raise ValueError("Dataset selection mismatch.")

        self.filenames = [
            Path(name).with_suffix(".npy").name
            for name in selection["filenames"]
        ]

        if len(self.filenames) != 1000:
            raise ValueError("Expected exactly 1000 images.")

        if len(set(self.filenames)) != 1000:
            raise ValueError("Duplicate filenames.")

        self.shape = (1024, 1536, len(self.filenames))

    def __getitem__(self, key):

        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError(
                "Use dataset[:, :, image_index]"
            )

        rows, columns, image_index = key

        if rows != slice(None) or columns != slice(None):
            raise IndexError(
                "Only full-image access is supported."
            )

        if not isinstance(image_index, (int, np.integer)):
            raise TypeError("Image index must be an integer.")

        if not 0 <= image_index < len(self.filenames):
            raise IndexError("Image index out of range.")

        path = self.processed / self.filenames[image_index]

        image = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False
        )

        if image.shape != self.shape[:2]:
            raise ValueError(
                f"Invalid image shape: {path.name}"
            )

        if image.dtype != np.float32:
            raise ValueError(
                f"Invalid image dtype: {path.name}"
            )

        return image
