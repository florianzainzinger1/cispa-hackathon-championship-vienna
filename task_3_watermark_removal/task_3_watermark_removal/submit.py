#!/usr/bin/env python3
"""
submit.py
---------
Uploads submission.npz to the Hackathon watermark-removal endpoint.

This version is robust:
- Validates the NPZ structure (images + names)
- Uploads via multipart/form-data using the expected field name: "file"
- Prints status code + content-type
- Safely handles JSON *and* non-JSON / empty responses (no crash)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import requests


# =========================
# CONFIGURATION
# =========================

API_KEY: str = "5091102ba147d8bece3d901377dbb6d1"
SUBMIT_URL: str = "http://35.192.205.84:80/submit/09-watermark-removal"
SUBMISSION_FILE: Path = Path(__file__).parent / "submission.npz"

# Network safety (avoid hanging forever)
CONNECT_TIMEOUT_SECONDS: int = 10
READ_TIMEOUT_SECONDS: int = 180


def _load_and_validate_submission(npz_path: Path) -> None:
    """
    Loads submission.npz and performs lightweight sanity checks.

    Why these checks?
    - 'images' / 'names' key naming is common for leaderboard submissions
    - Ensures 100 images and matching 100 names (as per your output)
    - Ensures image data is 4D and ends with 3 channels (HWC)
    """
    print("Loading submission file...")

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        raise RuntimeError(f"Could not read NPZ file: {npz_path} ({exc})") from exc

    keys = set(data.files)

    images_key = "images" if "images" in keys else None
    names_key = "names" if "names" in keys else None

    if images_key is None or names_key is None:
        raise ValueError(
            f"NPZ keys must include 'images' and 'names'. Found keys: {sorted(keys)}"
        )

    images = data[images_key]
    names = data[names_key]

    print(f"  Images shape: {getattr(images, 'shape', None)}")
    print(f"  Names count: {len(names) if hasattr(names, '__len__') else None}")

    try:
        sample = [str(x) for x in list(names)[:5]]
    except Exception:
        sample = ["<unavailable>"]
    print(f"  Sample names: {sample}")

    if not hasattr(images, "shape") or len(images.shape) != 4:
        raise ValueError(
            f"'images' must be a 4D array (N, H, W, C). Got: {getattr(images, 'shape', None)}"
        )

    n, h, w, c = images.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB). Got C={c}")

    if n != 100:
        print(f"[WARN] Expected 100 images, got {n}. (May still be accepted depending on server rules.)")

    if hasattr(names, "__len__") and len(names) != n:
        raise ValueError(f"'names' count ({len(names)}) must match number of images ({n}).")


def _print_server_response(response: requests.Response) -> None:
    """

    """
    print("\nResponse received")
    print(f"  Status: {response.status_code}")
    print(f"  Content-Type: {response.headers.get('Content-Type')}")

    try:
        payload: Any = response.json()
        print("  JSON:")
        print(payload)
    except ValueError:
        text = (response.text or "").strip()
        print("  Non-JSON:")
        print(text if text else "<empty response>")

    if not response.ok:
        raise RuntimeError(f"Submission failed with HTTP {response.status_code}")


def main() -> None:
    """
    Entry point.
    """
    if not SUBMISSION_FILE.exists():
        print(f"[ERROR] File not found: {SUBMISSION_FILE}")
        print("Hint: put submission.npz next to submit.py or adjust SUBMISSION_FILE.")
        sys.exit(1)

    try:
        _load_and_validate_submission(SUBMISSION_FILE)
    except Exception as exc:
        print(f"[ERROR] Submission file validation failed: {exc}")
        sys.exit(1)

    print("\nSubmitting to leaderboard server...")
    print(f"  URL: {SUBMIT_URL}")

    headers = {"X-API-Key": API_KEY}

    # Multipart/form-data upload: field name MUST be "file"
    # Use a context manager so the file handle is always closed properly.
    try:
        with SUBMISSION_FILE.open("rb") as fh:
            files = {
                "file": ("submission.npz", fh, "application/octet-stream")
            }
            response = requests.post(
                SUBMIT_URL,
                headers=headers,
                files=files,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out (server may be busy). Try again in a few minutes.")
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"[ERROR] Network/HTTP error while submitting: {exc}")
        sys.exit(1)

    try:
        _print_server_response(response)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
