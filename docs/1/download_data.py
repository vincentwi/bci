#!/usr/bin/env python3
"""
Download & Verify Data for Speech BCI Project
==============================================

Downloads the BCI competition dataset (ECoG keyword reading + syllable
repetition) and verifies file integrity.

Supports multiple data sources:
  1. Local path (copy/symlink existing data)
  2. Direct URL (e.g. pre-signed S3/GCS links)
  3. API endpoint with bearer token authentication

Usage:
    # From a direct URL
    python download_data.py --source url \\
        --url https://example.com/data.tar.gz \\
        --output-dir data

    # From an API with auth token
    python download_data.py --source api \\
        --api-url https://api.example.com/files \\
        --token YOUR_BEARER_TOKEN \\
        --output-dir data

    # Just verify existing data
    python download_data.py --verify-only --data-dir data

    # Show expected data structure
    python download_data.py --show-structure
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


# ============================================================
# Expected data structure
# ============================================================
EXPECTED_STRUCTURE = {
    'train': {
        'sessions': ['2022_09_22'],
        'per_session': {
            'runs': ['R01', 'R02', 'R03', 'R04'],
            'files_per_run': [
                'KeywordReading_Overt_{run}.mat',
                'KeywordReading_Overt_{run}.wav',
                'KeywordReading_Overt_{run}_trials.lab',
            ],
            'optional_per_run': [
                'KeywordReading_Overt_{run}_CI.xlsx',
            ],
        },
    },
    'syllable': {
        'sessions': ['2022_09_22'],
        'per_session': {
            'files': ['SyllableRepetition_Overt.mat'],
        },
    },
}

# Minimum expected file sizes (bytes) for quick integrity checks
MIN_FILE_SIZES = {
    '.mat': 10_000_000,   # ~60 MB typical
    '.wav': 5_000_000,    # ~15 MB typical
    '.lab': 500,          # ~1 KB typical
}


def show_structure():
    """Print the expected data directory layout."""
    print("""
Expected data directory structure:
==================================

data/
├── train/
│   └── 2022_09_22/
│       ├── KeywordReading_Overt_R01.mat      (~61 MB, ECoG signal)
│       ├── KeywordReading_Overt_R01.wav      (~15 MB, audio)
│       ├── KeywordReading_Overt_R01_trials.lab  (~1 KB, trial timing)
│       ├── KeywordReading_Overt_R01_CI.xlsx  (optional, channel info)
│       ├── KeywordReading_Overt_R02.mat
│       ├── KeywordReading_Overt_R02.wav
│       ├── KeywordReading_Overt_R02_trials.lab
│       ├── KeywordReading_Overt_R03.mat
│       ├── KeywordReading_Overt_R03.wav
│       ├── KeywordReading_Overt_R03_trials.lab
│       ├── KeywordReading_Overt_R04.mat
│       ├── KeywordReading_Overt_R04.wav
│       └── KeywordReading_Overt_R04_trials.lab
└── syllable/
    └── 2022_09_22/
        └── SyllableRepetition_Overt.mat      (~66 MB)

Data format:
  .mat  — BCI2000 format: signal(n_samples, 131), states, parameters
          signal[:, :128] = ECoG (int16, gain=0.25 µV/LSB, fs=1000 Hz)
          states.StimulusCode = word labels per sample
  .wav  — audio recording (float32, 16 kHz)
  .lab  — tab-separated: start_sec  end_sec  word_label
          6 words: Back, Down, Enter, Left, Right, Up
""")


# ============================================================
# Verification
# ============================================================
def verify_data(data_dir, verbose=True):
    """Verify that all expected data files exist and have reasonable sizes.

    Returns:
        (bool, list[str]): (all_ok, list of issues found)
    """
    data_dir = Path(data_dir)
    issues = []
    found = 0

    # Check train data
    for session in EXPECTED_STRUCTURE['train']['sessions']:
        session_dir = data_dir / 'train' / session
        if not session_dir.exists():
            issues.append(f'Missing directory: {session_dir}')
            continue

        for run in EXPECTED_STRUCTURE['train']['per_session']['runs']:
            for pattern in EXPECTED_STRUCTURE['train']['per_session']['files_per_run']:
                fname = pattern.format(run=run)
                fpath = session_dir / fname
                if not fpath.exists():
                    issues.append(f'Missing: {fpath}')
                else:
                    size = fpath.stat().st_size
                    ext = fpath.suffix
                    min_size = MIN_FILE_SIZES.get(ext, 0)
                    if size < min_size:
                        issues.append(
                            f'Too small: {fpath} ({size:,} bytes, '
                            f'expected >{min_size:,})')
                    else:
                        found += 1
                        if verbose:
                            print(f'  OK  {fpath.relative_to(data_dir)} '
                                  f'({size / 1e6:.1f} MB)')

    # Check syllable data
    for session in EXPECTED_STRUCTURE['syllable']['sessions']:
        session_dir = data_dir / 'syllable' / session
        if not session_dir.exists():
            issues.append(f'Missing directory: {session_dir}')
            continue

        for fname in EXPECTED_STRUCTURE['syllable']['per_session']['files']:
            fpath = session_dir / fname
            if not fpath.exists():
                issues.append(f'Missing: {fpath}')
            else:
                size = fpath.stat().st_size
                if size < MIN_FILE_SIZES.get('.mat', 0):
                    issues.append(f'Too small: {fpath} ({size:,} bytes)')
                else:
                    found += 1
                    if verbose:
                        print(f'  OK  {fpath.relative_to(data_dir)} '
                              f'({size / 1e6:.1f} MB)')

    all_ok = len(issues) == 0
    if verbose:
        print(f'\nVerification: {found} files OK', end='')
        if issues:
            print(f', {len(issues)} issues:')
            for issue in issues:
                print(f'  !! {issue}')
        else:
            print(' — all good!')

    return all_ok, issues


def deep_verify(data_dir, verbose=True):
    """Deep verification: load .mat files and check internal structure."""
    import scipy.io as sio

    data_dir = Path(data_dir)
    issues = []

    for session in EXPECTED_STRUCTURE['train']['sessions']:
        for run in EXPECTED_STRUCTURE['train']['per_session']['runs']:
            mat_path = (data_dir / 'train' / session /
                        f'KeywordReading_Overt_{run}.mat')
            if not mat_path.exists():
                continue

            if verbose:
                print(f'  Checking {run}...', end=' ')
            try:
                mat = sio.loadmat(str(mat_path))

                # Check signal
                if 'signal' not in mat:
                    issues.append(f'{run}: missing "signal" key')
                else:
                    sig = mat['signal']
                    if sig.shape[1] < 128:
                        issues.append(
                            f'{run}: signal has {sig.shape[1]} channels '
                            f'(expected >=128)')
                    if sig.shape[0] < 100000:
                        issues.append(
                            f'{run}: signal has {sig.shape[0]} samples '
                            f'(expected >100k)')

                # Check states
                if 'states' not in mat:
                    issues.append(f'{run}: missing "states" key')
                else:
                    states = mat['states'][0, 0]
                    if 'StimulusCode' not in states.dtype.names:
                        issues.append(f'{run}: missing StimulusCode in states')

                if verbose:
                    print(f'signal={mat["signal"].shape}, OK')

            except Exception as e:
                issues.append(f'{run}: load error: {e}')
                if verbose:
                    print(f'ERROR: {e}')

    # Check lab files match
    for session in EXPECTED_STRUCTURE['train']['sessions']:
        for run in EXPECTED_STRUCTURE['train']['per_session']['runs']:
            lab_path = (data_dir / 'train' / session /
                        f'KeywordReading_Overt_{run}_trials.lab')
            if not lab_path.exists():
                continue
            lines = lab_path.read_text().strip().split('\n')
            n_trials = len(lines)
            if n_trials < 50 or n_trials > 100:
                issues.append(
                    f'{run}: {n_trials} trials in .lab '
                    f'(expected 50-100)')
            if verbose:
                words = set()
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        words.add(parts[2])
                print(f'  {run} trials: {n_trials}, words: {sorted(words)}')

    all_ok = len(issues) == 0
    if verbose:
        if all_ok:
            print('  Deep verification: all checks passed')
        else:
            print(f'  Deep verification: {len(issues)} issues')
            for issue in issues:
                print(f'    !! {issue}')

    return all_ok, issues


# ============================================================
# Download functions
# ============================================================
def download_file(url, dest_path, token=None, chunk_size=8192):
    """Download a file from URL with optional auth token."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            total = resp.headers.get('Content-Length')
            total = int(total) if total else None
            downloaded = 0

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f'\r  Downloading {dest_path.name}: '
                              f'{downloaded/1e6:.1f}/{total/1e6:.1f} MB '
                              f'({pct:.0f}%)', end='', flush=True)
                    else:
                        print(f'\r  Downloading {dest_path.name}: '
                              f'{downloaded/1e6:.1f} MB', end='', flush=True)
            print()
            return True

    except HTTPError as e:
        print(f'\n  HTTP Error {e.code}: {e.reason}')
        if e.code == 401:
            print('  → Check your bearer token')
        elif e.code == 403:
            print('  → Access denied. You may need authentication.')
        return False
    except URLError as e:
        print(f'\n  Connection error: {e.reason}')
        return False


def download_from_url(url, output_dir, token=None):
    """Download data archive from a URL and extract it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename from URL
    fname = url.split('/')[-1].split('?')[0]
    archive_path = output_dir / fname

    print(f'Downloading from: {url}')
    if not download_file(url, archive_path, token=token):
        return False

    # Extract if archive
    print(f'Extracting {fname}...')
    if fname.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(output_dir)
    elif fname.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(output_dir)
    elif fname.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(output_dir)
    else:
        print(f'  Not an archive — saved as-is to {archive_path}')
        return True

    print(f'  Extracted to {output_dir}')
    return True


def download_from_api(api_url, output_dir, token, file_list=None):
    """Download individual files from an API endpoint.

    The API is expected to serve files at: {api_url}/{filename}
    """
    output_dir = Path(output_dir)

    if file_list is None:
        # Default file list
        file_list = []
        for session in EXPECTED_STRUCTURE['train']['sessions']:
            for run in EXPECTED_STRUCTURE['train']['per_session']['runs']:
                for pattern in EXPECTED_STRUCTURE['train']['per_session']['files_per_run']:
                    fname = pattern.format(run=run)
                    file_list.append(
                        (f'train/{session}/{fname}',
                         output_dir / 'train' / session / fname))
            for fname in EXPECTED_STRUCTURE['syllable']['per_session']['files']:
                file_list.append(
                    (f'syllable/{session}/{fname}',
                     output_dir / 'syllable' / session / fname))

    ok = 0
    fail = 0
    for remote_path, local_path in file_list:
        url = f'{api_url.rstrip("/")}/{remote_path}'
        if download_file(url, local_path, token=token):
            ok += 1
        else:
            fail += 1

    print(f'\nDownloaded: {ok} files, Failed: {fail} files')
    return fail == 0


def copy_local(source_dir, output_dir):
    """Copy or symlink data from a local path."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        print(f'Source directory not found: {source_dir}')
        return False

    for subdir in ['train', 'syllable']:
        src = source_dir / subdir
        dst = output_dir / subdir
        if src.exists():
            if dst.exists():
                print(f'  {dst} already exists, skipping')
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst)
                print(f'  Copied {src} → {dst}')
        else:
            print(f'  Warning: {src} not found')

    return True


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Download and verify BCI speech data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show expected data structure
  python download_data.py --show-structure

  # Verify existing data
  python download_data.py --verify-only --data-dir data

  # Deep verify (loads .mat files)
  python download_data.py --verify-only --deep --data-dir data

  # Download from URL
  python download_data.py --source url --url https://example.com/data.tar.gz

  # Download from API with token
  python download_data.py --source api --api-url https://api.example.com \\
      --token YOUR_TOKEN

  # Copy from local directory
  python download_data.py --source local --local-path /path/to/existing/data
""")

    parser.add_argument('--source', choices=['url', 'api', 'local'],
                        help='Download source type')
    parser.add_argument('--url', type=str,
                        help='Direct download URL for archive')
    parser.add_argument('--api-url', type=str,
                        help='API base URL for file downloads')
    parser.add_argument('--token', type=str,
                        help='Bearer token for authentication')
    parser.add_argument('--local-path', type=str,
                        help='Local path to copy data from')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Output directory for data (default: data)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing data, no download')
    parser.add_argument('--deep', action='store_true',
                        help='Deep verification (loads .mat files)')
    parser.add_argument('--show-structure', action='store_true',
                        help='Print expected data structure and exit')
    args = parser.parse_args()

    if args.show_structure:
        show_structure()
        return

    data_dir = Path(args.data_dir)

    if args.verify_only:
        print(f'Verifying data in: {data_dir.resolve()}\n')
        ok, issues = verify_data(data_dir)
        if args.deep and ok:
            print('\nRunning deep verification...')
            deep_verify(data_dir)
        sys.exit(0 if ok else 1)

    if not args.source:
        parser.print_help()
        print('\nError: --source is required for downloading')
        sys.exit(1)

    # Download
    print(f'Output directory: {data_dir.resolve()}\n')
    success = False

    if args.source == 'url':
        if not args.url:
            print('Error: --url required for url source')
            sys.exit(1)
        success = download_from_url(args.url, data_dir, token=args.token)

    elif args.source == 'api':
        if not args.api_url or not args.token:
            print('Error: --api-url and --token required for api source')
            sys.exit(1)
        success = download_from_api(args.api_url, data_dir, args.token)

    elif args.source == 'local':
        if not args.local_path:
            print('Error: --local-path required for local source')
            sys.exit(1)
        success = copy_local(args.local_path, data_dir)

    if success:
        print('\nVerifying downloaded data...')
        ok, issues = verify_data(data_dir)
        if not ok:
            print('\nSome files are missing or incomplete.')
            print('Check the issues above and re-download if needed.')
    else:
        print('\nDownload failed. Check the errors above.')
        sys.exit(1)


if __name__ == '__main__':
    main()
