#!/usr/bin/env python3
"""Download the full Angrick et al. (2024) dataset from OSF (~4 GB).

Usage:
    python scripts/01_download.py                # download everything
    python scripts/01_download.py --dry-run      # list files only
    python scripts/01_download.py --verify       # check for missing files
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from speech_bci.download import download_all, verify_download
from speech_bci import config


def main():
    parser = argparse.ArgumentParser(description="Download Speech BCI dataset from OSF")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded files")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="Download directory")
    args = parser.parse_args()

    if args.verify:
        report = verify_download(args.data_dir)
        sys.exit(1 if report else 0)

    manifest = download_all(args.data_dir, skip_existing=True, dry_run=args.dry_run)

    if not args.dry_run:
        print("\n--- Verification ---")
        verify_download(args.data_dir)

    return manifest


if __name__ == "__main__":
    main()
