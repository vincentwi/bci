"""Download the full Angrick et al. (2024) dataset from OSF.

OSF node: https://osf.io/49rt7/
Uses the OSF REST API to enumerate and download all files.
"""

import json
from pathlib import Path

import requests
from tqdm import tqdm

from . import config

# Top-level OSF folder IDs (from the OSF storage structure)
# These are the folder_id values for each top-level directory
OSF_FILES_URL = f"{config.OSF_API_BASE}/nodes/{config.OSF_NODE_ID}/files/osfstorage/"


def _get_json(url: str) -> dict:
    """Fetch JSON from OSF API with basic error handling."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_osf_folder(folder_url: str) -> list[dict]:
    """Recursively list all files under an OSF folder.

    Handles pagination (OSF returns 10 items per page).

    Returns
    -------
    files : list of {name, path, download_url, size, kind}
    """
    files = []
    url = folder_url

    while url:
        data = _get_json(url)
        for item in data.get("data", []):
            attrs = item.get("attributes", {})
            links = item.get("links", {})
            name = attrs.get("name", "")
            kind = attrs.get("kind", "")
            size = attrs.get("size", 0)
            materialized_path = attrs.get("materialized_path", "")

            if kind == "folder":
                # Recurse into subfolders
                subfolder_url = links.get("related", {})
                if isinstance(subfolder_url, dict):
                    subfolder_url = subfolder_url.get("href")
                if not subfolder_url:
                    rel = item.get("relationships", {})
                    subfolder_url = rel.get("files", {}).get("links", {}).get("related", {}).get("href")
                if subfolder_url:
                    files.extend(list_osf_folder(subfolder_url))
            else:
                download_url = links.get("download")
                if download_url:
                    files.append({
                        "name": name,
                        "path": materialized_path,
                        "download_url": download_url,
                        "size": size,
                        "kind": kind,
                    })

        # Handle pagination
        next_url = data.get("links", {}).get("next")
        url = next_url

    return files


def download_file(url: str, dest: Path, expected_size: int | None = None,
                  chunk_size: int = 8192) -> Path:
    """Download a single file with progress bar and optional size verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) or expected_size or 0
    desc = dest.name[:40]

    with open(dest, "wb") as f, \
         tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    if expected_size and dest.stat().st_size != expected_size:
        print(f"  WARNING: size mismatch for {dest.name}: "
              f"expected {expected_size}, got {dest.stat().st_size}")

    return dest


def _map_osf_path_to_local(osf_path: str, base_dir: Path) -> Path:
    """Map an OSF materialized path to local filesystem path.

    OSF paths look like: /KeywordReading/train/2022_09_22/KeywordReading_Overt_R01.mat
    or /SyllableRepetition/2022_09_22/SyllableRepetition_Overt.mat

    We map:
      /KeywordReading/train/...     → base_dir/train/...
      /KeywordReading/validation/...→ base_dir/validation/...
      /KeywordReading/test/...      → base_dir/test/...
      /KeywordReading/online_sessions/... → base_dir/online_sessions/...
      /SyllableRepetition/...       → base_dir/syllable/...
    """
    parts = osf_path.strip("/").split("/")

    if parts[0] == "SyllableRepetition":
        # /SyllableRepetition/2022_09_22/file.mat → syllable/2022_09_22/file.mat
        return base_dir / "syllable" / "/".join(parts[1:])
    elif parts[0] == "KeywordReading":
        # /KeywordReading/train/2022_09_22/file.mat → train/2022_09_22/file.mat
        return base_dir / "/".join(parts[1:])
    else:
        # Unknown top-level folder — preserve structure
        return base_dir / "/".join(parts)


def download_all(base_dir: Path | None = None,
                 skip_existing: bool = True,
                 dry_run: bool = False) -> dict:
    """Download the complete dataset from OSF.

    Parameters
    ----------
    base_dir : local directory for downloaded data (default: config.DATA_DIR)
    skip_existing : skip files that already exist locally
    dry_run : if True, list files without downloading

    Returns
    -------
    manifest : {
        "total_files": int,
        "downloaded": int,
        "skipped": int,
        "failed": list[str],
        "total_bytes": int,
    }
    """
    base_dir = base_dir or config.DATA_DIR

    print("Enumerating files on OSF (this may take a minute)...")
    all_files = list_osf_folder(OSF_FILES_URL)
    print(f"Found {len(all_files)} files on OSF")

    if dry_run:
        total_size = sum(f["size"] for f in all_files)
        print(f"\nTotal download size: {total_size / 1e9:.2f} GB")
        print("\nFiles:")
        for f in sorted(all_files, key=lambda x: x["path"]):
            print(f"  {f['path']} ({f['size'] / 1e6:.1f} MB)")
        return {"total_files": len(all_files), "total_bytes": total_size}

    manifest = {
        "total_files": len(all_files),
        "downloaded": 0,
        "skipped": 0,
        "failed": [],
        "total_bytes": 0,
    }

    for file_info in tqdm(all_files, desc="Downloading dataset"):
        local_path = _map_osf_path_to_local(file_info["path"], base_dir)

        if skip_existing and local_path.exists():
            # Check size matches
            if file_info["size"] and local_path.stat().st_size == file_info["size"]:
                manifest["skipped"] += 1
                continue

        try:
            download_file(
                file_info["download_url"],
                local_path,
                expected_size=file_info.get("size"),
            )
            manifest["downloaded"] += 1
            manifest["total_bytes"] += file_info.get("size", 0)
        except Exception as e:
            print(f"\n  FAILED: {file_info['path']}: {e}")
            manifest["failed"].append(file_info["path"])

    print(f"\nDone: {manifest['downloaded']} downloaded, "
          f"{manifest['skipped']} skipped, "
          f"{len(manifest['failed'])} failed")

    return manifest


def verify_download(base_dir: Path | None = None) -> dict:
    """Verify that all expected files are present.

    Returns
    -------
    report : {split: {day: {run: [missing_files]}}}
    """
    base_dir = base_dir or config.DATA_DIR
    report = {}

    splits = {
        "train": config.TRAIN_DAYS,
        "validation": [config.VAL_DAY],
        "test": [config.TEST_DAY],
        "online_sessions": config.ONLINE_DAYS,
    }

    for split_name, days in splits.items():
        split_dir = base_dir / split_name
        for day_id in days:
            day_dir = split_dir / day_id
            missing = []
            for run in config.RUNS:
                prefix = f"KeywordReading_Overt_{run}"
                for ext in [".mat", ".wav", "_trials.lab"]:
                    fpath = day_dir / f"{prefix}{ext}"
                    if not fpath.exists():
                        missing.append(fpath.name)
            if missing:
                report.setdefault(split_name, {})[day_id] = missing

    # Check syllable repetition
    for day_id in config.ALL_DAYS:
        syll_path = base_dir / "syllable" / day_id / "SyllableRepetition_Overt.mat"
        if not syll_path.exists():
            report.setdefault("syllable", {})[day_id] = ["SyllableRepetition_Overt.mat"]

    if not report:
        print("All expected files present!")
    else:
        for split, days in report.items():
            for day, files in days.items():
                print(f"  MISSING {split}/{day}: {', '.join(files)}")

    return report


if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv:
        download_all(dry_run=True)
    else:
        download_all()
        verify_download()
