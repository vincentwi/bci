"""Channel selection using CI analysis from .xlsx files."""

from pathlib import Path

import numpy as np
import openpyxl

from . import config


def parse_ci_xlsx(xlsx_path: Path) -> dict[int, dict]:
    """Parse the CI (confidence interval) analysis spreadsheet.

    The CI xlsx files contain per-channel statistics indicating which
    channels carry significant speech-related activity.

    Parameters
    ----------
    xlsx_path : path to KeywordReading_Overt_R01_CI.xlsx

    Returns
    -------
    results : {channel_idx: {name, ci_lower, ci_upper, significant, ...}}
    """
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        wb.close()
        return {}

    header = [str(h).strip().lower() if h else f"col_{i}"
              for i, h in enumerate(rows[0])]

    results = {}
    for row_idx, row in enumerate(rows[1:]):
        row_dict = {header[i]: row[i] for i in range(min(len(header), len(row)))}

        # Try to extract channel index from various possible column names
        ch_idx = None
        for key in ["channel", "chan", "electrode", "index", "col_0"]:
            if key in row_dict and row_dict[key] is not None:
                try:
                    ch_idx = int(row_dict[key])
                except (ValueError, TypeError):
                    # Might be a name like "chan1" — extract number
                    val = str(row_dict[key])
                    digits = "".join(c for c in val if c.isdigit())
                    if digits:
                        ch_idx = int(digits) - 1  # 0-indexed
                break

        if ch_idx is None:
            ch_idx = row_idx

        # Extract CI bounds and significance
        ci_lower = None
        ci_upper = None
        significant = False

        for key in header:
            val = row_dict.get(key)
            if val is None:
                continue
            if "lower" in key or "ci_lo" in key:
                ci_lower = float(val) if val else None
            elif "upper" in key or "ci_hi" in key:
                ci_upper = float(val) if val else None
            elif "sig" in key or "significant" in key or "p_val" in key or "pval" in key:
                if isinstance(val, bool):
                    significant = val
                elif isinstance(val, (int, float)):
                    significant = float(val) < 0.05

        # If CI bounds exist but no explicit significance, check if CI excludes 0
        if ci_lower is not None and ci_upper is not None and not significant:
            significant = (ci_lower > 0) or (ci_upper < 0)

        results[ch_idx] = {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": significant,
            "raw": row_dict,
        }

    wb.close()
    return results


def select_speech_channels(ci_data: dict[int, dict],
                            threshold: float = 0.05) -> np.ndarray:
    """Return sorted array of channel indices that pass significance threshold.

    Parameters
    ----------
    ci_data : output of parse_ci_xlsx
    threshold : p-value threshold for significance

    Returns
    -------
    channels : sorted array of significant channel indices
    """
    significant = [ch for ch, info in ci_data.items() if info.get("significant")]
    return np.sort(significant)


def get_default_speech_channels() -> np.ndarray:
    """Return the paper's exact speech-relevant electrode set.

    Matches SelectElectrodesOverSpeechAreas from the paper's GitHub:
    https://github.com/cronelab/delayed-speech-synthesis/blob/main/local/common.py

    64 channels: 60 from the speech grid + 4 dorsal laryngeal electrodes.
    Channels 18, 37, 47, 51 are excluded (bad/non-speech channels).
    """
    # Exact mapping from paper's code (1-indexed)
    speech_grid_mapping = np.array([
        1, 2, 3, 0, 4, 11, 5, 6, 7, 10, 12, 9, 19, 8, 15, 20, 13, 14, 17, 22,
        18, 21, 29, 16, 23, 28, 35, 36, 27, 25, 26, 55, 45, 46, 44, 24, 37, 40,
        33, 34, 32, 51, 47, 39, 31, 54, 53, 30, 48, 38, 43, 41, 52, 61, 59, 62,
        49, 66, 60, 63, 58, 50, 42, 56, 67, 57, 81, 68
    ]) + 1  # Convert to 1-indexed

    # Remove 4 excluded channels (1-indexed: 19, 38, 48, 52)
    speech_grid_mapping = np.array([v for v in speech_grid_mapping if v not in [19, 38, 48, 52]])

    # Convert back to 0-indexed and sort
    speech_grid_mapping -= 1
    return np.sort(speech_grid_mapping)


def get_channel_mask(data_dir: Path | None = None,
                     day_ids: list[str] | None = None,
                     method: str = "ci") -> np.ndarray:
    """Get the channel selection mask.

    Parameters
    ----------
    data_dir : directory containing session data with CI xlsx files
    day_ids : which days to consider
    method : "ci" (from xlsx analysis) or "default" (paper's hardcoded set)

    Returns
    -------
    channels : sorted array of selected channel indices
    """
    if method == "default" or data_dir is None:
        return get_default_speech_channels()

    day_ids = day_ids or config.TRAIN_DAYS
    all_significant = set()

    for day_id in day_ids:
        day_dir = data_dir / day_id
        xlsx_files = list(day_dir.glob("*_CI.xlsx"))
        for xlsx_path in xlsx_files:
            ci_data = parse_ci_xlsx(xlsx_path)
            significant = select_speech_channels(ci_data)
            all_significant.update(significant)

    if not all_significant:
        return get_default_speech_channels()

    return np.sort(list(all_significant))
