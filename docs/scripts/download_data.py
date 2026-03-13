#!/usr/bin/env python3
"""
Download the full Willett et al. (Nature 2023) speech neuroprosthesis dataset
from Dryad. Includes competition data and diagnostic blocks.

Usage:
    python download_data.py                # download everything
    python download_data.py --list         # just list available files
    python download_data.py --file NAME    # download specific file
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from config import DATA_DIR

DRYAD_DOI = "10.5061/dryad.x69p8czpq"
DRYAD_API = "https://datadryad.org/api/v2"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def get_dataset_info():
    encoded_doi = DRYAD_DOI.replace("/", "%2F")
    url = f"{DRYAD_API}/datasets/doi%3A{encoded_doi}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_file_list(dataset_info):
    files_href = dataset_info.get("_links", {}).get("stash:files", {}).get("href", "")
    if not files_href:
        print("No files link found in dataset metadata")
        return []
    if not files_href.startswith("http"):
        files_href = "https://datadryad.org" + files_href

    req = urllib.request.Request(files_href, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        files_data = json.loads(resp.read())

    return files_data.get("_embedded", {}).get("stash:files", [])


def download_file(file_info, dest_dir):
    """Download a single file with progress bar."""
    path = file_info.get("path", "unknown")
    size = file_info.get("size", 0)
    dl_href = file_info.get("_links", {}).get("stash:download", {}).get("href", "")

    if not dl_href:
        print(f"  No download link for {path}")
        return False

    if not dl_href.startswith("http"):
        dl_href = "https://datadryad.org" + dl_href

    dest = dest_dir / path
    if dest.exists() and dest.stat().st_size == size:
        print(f"  {path}: already downloaded ({size / 1e6:.1f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    size_mb = size / 1e6

    print(f"  {path}: downloading {size_mb:.1f} MB ...")
    try:
        req = urllib.request.Request(dl_href)
        with urllib.request.urlopen(req, timeout=300) as resp:
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded / size * 100 if size > 0 else 0
                    print(f"\r    {downloaded / 1e6:.1f}/{size_mb:.1f} MB ({pct:.0f}%)",
                          end="", flush=True)
            print()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"\n    FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Willett et al. ECoG data from Dryad")
    parser.add_argument("--list", action="store_true", help="List files without downloading")
    parser.add_argument("--file", type=str, help="Download specific file by name")
    parser.add_argument("--dest", type=str, default=str(DATA_DIR / "dryad"),
                        help="Destination directory")
    args = parser.parse_args()

    dest_dir = Path(args.dest)

    print(f"Dataset DOI: {DRYAD_DOI}")
    print(f"Destination: {dest_dir}\n")

    # Get dataset metadata
    print("Querying Dryad API...")
    try:
        info = get_dataset_info()
    except Exception as e:
        print(f"Failed to query Dryad: {e}")
        print("\nManual download:")
        print(f"  https://datadryad.org/stash/dataset/doi:{DRYAD_DOI}")
        sys.exit(1)

    title = info.get("title", "Unknown")
    print(f"Title: {title}")
    print(f"Version: {info.get('versionNumber', '?')}\n")

    # Get file listing
    try:
        files = get_file_list(info)
    except Exception as e:
        print(f"Failed to get file list: {e}")
        sys.exit(1)

    if not files:
        print("No files found. The dataset may require authentication.")
        print(f"Visit: https://datadryad.org/stash/dataset/doi:{DRYAD_DOI}")
        sys.exit(1)

    total_size = sum(f.get("size", 0) for f in files)
    print(f"Available files ({len(files)} files, {total_size / 1e9:.2f} GB total):")
    for f in files:
        sz = f.get("size", 0) / 1e6
        print(f"  {f.get('path', '?'):<50} {sz:>8.1f} MB")

    if args.list:
        return

    # Download
    if args.file:
        targets = [f for f in files if args.file in f.get("path", "")]
        if not targets:
            print(f"\nNo file matching '{args.file}'")
            sys.exit(1)
    else:
        targets = files

    print(f"\nDownloading {len(targets)} files to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for f in targets:
        if download_file(f, dest_dir):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed")
    if fail > 0:
        print("Some downloads failed. The dataset may require authentication.")
        print(f"Try: https://datadryad.org/stash/dataset/doi:{DRYAD_DOI}")


if __name__ == "__main__":
    main()
