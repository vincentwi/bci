"""
Exact replication of Angrick et al. 2023 "Online speech synthesis using a chronically
implanted brain-computer interface" training pipeline.

Targets: LPC coefficients (20 dims) decoded by LPCNet vocoder
Model: Bidirectional LSTM, 2 layers, 100 hidden units, dropout 0.5
Optimizer: RMSprop, lr=0.0001
Batch size: 1
Features: 64 high-gamma channels over speech areas
Preprocessing: CAR (excl ch 19,38,48,52), bad channel correction, z-score per day

Uses GPU for training. Preprocessing (HG extraction + LPC feature computation) runs
on CPU as it depends on the paper's Cython extensions (hga_optimized, LPCNet).
"""
import argparse
import logging
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torchinfo
import tqdm
import h5py
import mne
import math
from pathlib import Path
from collections import defaultdict
from functools import reduce
from typing import Optional, List, Callable, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from scipy.signal import sosfilt, sosfilt_zi
from scipy.io import loadmat
from scipy.io.wavfile import read as wavread, write as wavwrite
from pydub import AudioSegment, effects
from hga_optimized import compute_log_power_features, WarmStartFrameBuffer
from LPCNet import LPCFeatureEncoder, LPCNet

import matplotlib
matplotlib.use('Agg')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
    datefmt='%d.%m.%y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("train_paper_replication")


# ============================================================================
# Channel selection and preprocessing (from paper's local/common.py)
# ============================================================================

class SelectElectrodesFromBothGrids:
    def __init__(self):
        self.grid_mapping = [125, 123, 121, 119, 122, 111, 118, 124, 120, 126, 127, 116, 114, 113, 115, 117, 98, 97, 96,
                             104, 100, 102, 101, 99, 105, 112, 107, 106, 108, 103, 109, 110, 17, 21, 9, 28, 26, 31, 13,
                             27, 25, 22, 30, 11, 29, 23, 19, 15, 1, 2, 4, 0, 24, 12, 14, 7, 5, 18, 6, 10, 3, 8, 20, 16,
                             50, 33, 44, 51, 63, 40, 38, 46, 42, 48, 56, 37, 35, 41, 47, 58, 61, 60, 59, 43, 49, 45, 54,
                             62, 32, 53, 55, 52, 57, 39, 34, 36, 85, 84, 83, 87, 80, 86, 90, 78, 75, 92, 76, 88, 82, 94,
                             70, 74, 69, 66, 79, 71, 73, 77, 68, 67, 64, 65, 95, 93, 81, 72, 91, 89]
    def __len__(self):
        return len(self.grid_mapping)
    def __call__(self, data):
        return data[:, self.grid_mapping]


class SelectElectrodesOverSpeechAreas:
    def __init__(self):
        self.speech_grid_mapping = np.array([1, 2, 3, 0, 4, 11, 5, 6, 7, 10, 12, 9, 19, 8, 15, 20, 13, 14, 17, 22,
                                             18, 21, 29, 16, 23, 28, 35, 36, 27, 25, 26, 55, 45, 46, 44, 24, 37, 40,
                                             33, 34, 32, 51, 47, 39, 31, 54, 53, 30, 48, 38, 43, 41, 52, 61, 59, 62,
                                             49, 66, 60, 63, 58, 50, 42, 56, 67, 57, 81, 68]) + 1
        self.speech_grid_mapping = np.array([val for val in self.speech_grid_mapping if val not in [19, 38, 48, 52]])
        self.speech_grid_mapping -= 1
        self.speech_grid_mapping = np.array(sorted(self.speech_grid_mapping))
    def __len__(self):
        return len(self.speech_grid_mapping)
    def __call__(self, data):
        return data[:, self.speech_grid_mapping]


class CommonAverageReferencing:
    def __init__(self, exclude_channels, grids, layout):
        self.grids = grids
        self.layout = layout
        self.selection_masks_application = [np.isin(layout, grid) for grid in grids]
        self.selection_masks_computation = []
        for grid, mask_appl in zip(self.grids, self.selection_masks_application):
            mask_comp = mask_appl.copy()
            for excluded_channel in exclude_channels:
                if excluded_channel in grid:
                    mask_comp[np.argmax(layout == excluded_channel)] = False
            self.selection_masks_computation.append(mask_comp)

    def __call__(self, data):
        result = data.copy()
        for mask_comp, mask_appl in zip(self.selection_masks_computation, self.selection_masks_application):
            means = np.mean(data[:, mask_comp], axis=1).reshape((-1, 1))
            means = np.tile(means, reps=(1, np.count_nonzero(mask_appl)))
            result[:, mask_appl] = result[:, mask_appl] - means
        return result


class BadChannelCorrection:
    def __init__(self, bad_channels, grids, layout):
        from scipy.ndimage import binary_dilation
        self.grids = grids
        self.layout = layout
        self.masks = [np.ones(grid.shape, dtype=bool) for grid in grids]
        for bc in bad_channels:
            for i, grid in enumerate(grids):
                if bc in grid:
                    row, col = np.where(grid == bc)
                    self.masks[i][row, col] = False
        footprint = np.ones(9, dtype=bool).reshape((3, 3))
        footprint[1, 1] = False
        self.patches = []
        for bc in bad_channels:
            for i, grid in enumerate(grids):
                if bc in grid:
                    row, col = np.where(grid == bc)
                    mask = np.zeros(grid.shape, dtype=bool)
                    mask[row, col] = True
                    mask = binary_dilation(mask, structure=footprint)
                    mask = mask & self.masks[i]
                    neighbors = grid[mask]
                    bc_idx = np.where(self.layout == bc)[0]
                    neighbor_idx = np.concatenate([np.where(self.layout == n)[0] for n in neighbors])
                    self.patches.append((bc_idx, neighbor_idx))

    def __call__(self, data):
        result = data.copy()
        for bc_loc, neighbors in self.patches:
            result[:, bc_loc] = np.mean(data[:, neighbors], axis=1).reshape((len(data), -1))
        return result


# ============================================================================
# High-gamma feature extractor (from paper's local/units.py)
# ============================================================================

class HighGammaExtractor:
    def __init__(self, fs, nb_electrodes, window_length=0.05, window_shift=0.01,
                 l_freq=70, h_freq=170, pre_transforms=None, post_transforms=None):
        self.fs = fs
        self.nb_electrodes = nb_electrodes
        self.window_length = window_length
        self.window_shift = window_shift
        self.pre_transform = None
        self.post_transform = None
        if pre_transforms is not None:
            self.pre_transform = reduce(lambda f, g: lambda x: g(f(x)), pre_transforms, lambda x: x)
        if post_transforms is not None:
            self.post_transform = reduce(lambda f, g: lambda x: g(f(x)), post_transforms, lambda x: x)

        self.framebuffer = WarmStartFrameBuffer(frame_length=window_length, frame_shift=window_shift,
                                                fs=fs, nb_channels=nb_electrodes)
        iir_params = {'order': 8, 'ftype': 'butter'}
        self.hg_filter = mne.filter.create_filter(None, fs, l_freq, h_freq, 'auto', 'auto',
                                                   'auto', 'iir', iir_params, 'zero', 'hamming', 'firwin', verbose=False)["sos"]
        self.fh_filter = mne.filter.create_filter(None, fs, 122, 118, 'auto', 'auto',
                                                   'auto', 'iir', iir_params, 'zero', 'hamming', 'firwin', verbose=False)["sos"]
        hg_state = sosfilt_zi(self.hg_filter)
        fh_state = sosfilt_zi(self.fh_filter)
        self.hg_state = np.repeat(hg_state, nb_electrodes, axis=-1).reshape([hg_state.shape[0], hg_state.shape[1], -1])
        self.fh_state = np.repeat(fh_state, nb_electrodes, axis=-1).reshape([fh_state.shape[0], fh_state.shape[1], -1])

    def extract_features(self, data):
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        data, self.hg_state = sosfilt(self.hg_filter, data, axis=0, zi=self.hg_state)
        data, self.fh_state = sosfilt(self.fh_filter, data, axis=0, zi=self.fh_state)
        data = self.framebuffer.insert(data)
        data = compute_log_power_features(data, self.fs, self.window_length, self.window_shift)
        if self.post_transform is not None:
            data = self.post_transform(data)
        return data


# ============================================================================
# BCI2000 mat file wrapper (from paper's local/common.py)
# ============================================================================

class BCI2000MatFile:
    def __init__(self, mat_filename):
        self.mat_filename = mat_filename
        self.mat = loadmat(mat_filename, simplify_cells=True)
        self.fs = self.mat['parameters']['SamplingRate']['NumericValue']

    def bad_channels(self):
        if 'bad_channels' in self.mat.keys():
            bc = self.mat['bad_channels']
            if type(bc) is np.ndarray:
                bc = bc.tolist()
            bc = [bc] if type(bc) is not list else bc
            bc = [int(b[4:]) for b in bc]
            return bc
        return None

    def contaminated_channels(self):
        if "contaminated_electrodes" in self.mat.keys():
            ce = self.mat['contaminated_electrodes']
            if type(ce) is int:
                return [ce]
            return ce.tolist()
        return None

    def signals(self):
        signals = self.mat['signal']
        gain = self.mat['parameters']['SourceChGain']['NumericValue']
        return signals * gain

    def trial_indices(self, min_trial_length=None):
        stimuli = self._extract_stimuli()
        stimulus_code = self._get_stimulus_code()

        # Check if this is a SyllableRepetition or KeywordReading file
        filename = os.path.basename(self.mat_filename)
        if 'SyllableRepetition' in filename:
            trial_indices = self._get_syllable_trials(stimulus_code, stimuli)
        else:
            trial_indices = self._get_keyword_trials(stimulus_code, stimuli)

        if min_trial_length is not None:
            nb_min = min_trial_length * self.fs
            trial_indices = [(l, s, max(e, s + nb_min)) for l, s, e in trial_indices]
        return trial_indices

    def _get_stimulus_code(self):
        states = self.mat['states']
        if isinstance(states, dict):
            return states['StimulusCode']
        elif isinstance(states, np.ndarray):
            # Flat array format (e.g., 2022_10_27) - the array IS the StimulusCode
            return states
        else:
            raise ValueError(f"Unknown states format: {type(states)}")

    def _extract_stimuli(self):
        stimuli = self.mat['parameters']['Stimuli']['Value']
        if stimuli.ndim == 1:
            return [stimuli[0]]
        elif stimuli.ndim == 2:
            return stimuli[0].tolist()
        return stimuli[0].tolist()

    @staticmethod
    def _get_keyword_trials(stimulus_code, stimuli):
        stimuli_dict = {(i + 1): item for i, item in enumerate(stimuli)}
        result = []
        start = None
        label = None
        for i in range(len(stimulus_code)):
            if stimulus_code[i] != 0 and start is None:
                start = i
                label = stimuli_dict[stimulus_code[i]]
            if stimulus_code[i] == 0 and start is not None:
                result.append((label, start, i))
                start = None
                label = None
        return result

    @staticmethod
    def _get_syllable_trials(stimulus_code, stimuli):
        """SyllableRepetition: alternates between auditory presentation and patient speaking.
        We want the patient speaking segments (every other non-zero segment)."""
        stimuli_dict = {(i + 1): item for i, item in enumerate(stimuli)}
        # Find all non-zero segments
        segments = []
        start = None
        code = None
        for i in range(len(stimulus_code)):
            if stimulus_code[i] != 0 and start is None:
                start = i
                code = stimulus_code[i]
            if stimulus_code[i] == 0 and start is not None:
                segments.append((code, start, i))
                start = None
                code = None

        # Auditory presentation segments are odd-indexed (0, 2, 4, ...)
        # Patient speaking segments are even-indexed (1, 3, 5, ...)
        # Swap: assign the label from the presentation to the speaking segment
        result = []
        for k in range(1, len(segments), 2):
            presentation_code = segments[k - 1][0]
            speaking_start = segments[k][1]
            speaking_stop = segments[k][2]
            label = stimuli_dict[presentation_code]
            result.append((label, speaking_start, speaking_stop))
        return result


# ============================================================================
# Feature extraction pipeline (from paper's prepare_corpus.py)
# ============================================================================

def get_feature_extractor(mat_file):
    fs = mat_file.fs
    bad_channels = mat_file.bad_channels()
    contaminated = mat_file.contaminated_channels()

    feature_selection = SelectElectrodesFromBothGrids()
    pre_transforms = [feature_selection]

    speech_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 1, axis=0)
    motor_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 65, axis=0)
    layout = np.arange(128) + 1
    car = CommonAverageReferencing(exclude_channels=[19, 38, 48, 52], grids=[speech_grid, motor_grid], layout=layout)
    pre_transforms.append(car)
    post_transforms = None

    if contaminated is not None:
        corrected = (bad_channels or []) + contaminated
        ch_correction = BadChannelCorrection(bad_channels=corrected, grids=[speech_grid, motor_grid], layout=layout)
        post_transforms = [ch_correction]

    nb_electrodes = len(feature_selection)
    ex = HighGammaExtractor(fs=fs, nb_electrodes=nb_electrodes, pre_transforms=pre_transforms,
                            post_transforms=post_transforms)
    return ex


def normalize_audio(audio, fs, normalization_factor=-3.0):
    segment = AudioSegment(audio.tobytes(), frame_rate=fs, sample_width=audio.dtype.itemsize, channels=1)
    segment = effects.normalize(segment)
    segment = segment.apply_gain(normalization_factor)
    return np.array(segment.get_array_of_samples())


def extract_features_for_file(mat_path, wav_path, min_trial_length=2.5):
    mat = BCI2000MatFile(mat_path)
    fs_audio, wav = wavread(wav_path)
    ecog = mat.signals()

    features_list = []
    lpc_list = []
    trial_ids_list = []
    stimuli = mat._extract_stimuli()

    last_stimuli_code = None
    for label, start, stop in mat.trial_indices(min_trial_length):
        # HG features
        extractor = get_feature_extractor(mat)
        feats = extractor.extract_features(ecog[start:int(stop + (0.04 * mat.fs)), :])
        features_list.append(feats)

        # LPC coefficients
        audio_start = int(start * fs_audio / mat.fs)
        audio_stop = int(stop * fs_audio / mat.fs) + int(0.04 * fs_audio)
        trial_audio = wav[audio_start:audio_stop]
        if label != "SILENCE":
            trial_audio = normalize_audio(trial_audio, fs=fs_audio, normalization_factor=-3.0)
        # Shift audio by 16 ms for filter delay
        filter_delay_pad = np.zeros(int(0.016 * fs_audio), dtype=np.int16)
        trial_audio = np.hstack([filter_delay_pad, trial_audio[:-len(filter_delay_pad)]]).astype(np.int16)
        encoder = LPCFeatureEncoder()
        lpc_feats = encoder.compute_LPC_features(trial_audio)
        lpc_list.append(lpc_feats[3:-1])

        # Trial IDs
        interval = int(stop + (0.04 * mat.fs)) - start
        overlap = 0.04 * mat.fs
        window_shift = 0.01 * mat.fs
        num_windows = int(np.floor((interval - overlap) / window_shift))
        stim_code = stimuli.index(label) + 1
        if last_stimuli_code is None or last_stimuli_code != stim_code:
            trial_ids_list.append(np.ones(num_windows) * stim_code)
            last_stimuli_code = stim_code
        else:
            trial_ids_list.append(np.ones(num_windows) * stim_code * -1)
            last_stimuli_code = stim_code * -1

    return (np.concatenate(features_list), np.concatenate(lpc_list),
            np.hstack(trial_ids_list).astype(np.int16))


# ============================================================================
# Dataset class (from paper's local/training.py)
# ============================================================================

class SequentialSpeechTrials(Dataset):
    def __init__(self, feature_files, transform=None, target_specifier="lpc_coefficients"):
        self.feature_files = feature_files
        self.transform = transform
        self.target_specifier = target_specifier
        self.fhs = [h5py.File(f, 'r') for f in feature_files]
        self.nb_trials = [self._count_trials(fh['trial_ids'][...]) for fh in self.fhs]
        self.trial_labels = []
        self.frame_counter = 0
        for fh in self.fhs:
            self.frame_counter += len(fh['trial_ids'][...])
            trial_stimuli = self._squeeze_trial_ids(fh['trial_ids'][...])
            self.trial_labels.extend(trial_stimuli)

        self.cumulative_length = 0
        self.feature_dict = {}
        for nb_trials, fh in zip(self.nb_trials, self.fhs):
            self.feature_dict[(self.cumulative_length, self.cumulative_length + nb_trials)] = fh
            self.cumulative_length += nb_trials

    def __del__(self):
        for fh in self.fhs:
            fh.close()

    def __len__(self):
        return sum(self.nb_trials)

    @staticmethod
    def _count_trials(trial_ids):
        return len(np.where(trial_ids[:-1] != trial_ids[1:])[0]) + 1

    @staticmethod
    def _squeeze_trial_ids(trial_ids):
        last = trial_ids[0]
        result = [last]
        for i in range(1, len(trial_ids)):
            if trial_ids[i] != last:
                result.append(abs(trial_ids[i]))
                last = trial_ids[i]
        return result

    @staticmethod
    def _find_indices_of_nth_subsequence(n, seq):
        from itertools import pairwise
        from operator import itemgetter
        take_nth = itemgetter(n)
        borders = (np.where(seq[:-1] != seq[1:])[0] + 1).tolist()
        borders = [0] + borders + [len(seq)]
        borders = tuple(pairwise(borders))
        start, stop = take_nth(borders)
        return start, stop

    def __getitem__(self, index):
        for (start, stop) in self.feature_dict.keys():
            if start <= index < stop:
                trial_ids = self.feature_dict[(start, stop)]['trial_ids'][...]
                trial_start, trial_stop = self._find_indices_of_nth_subsequence(index - start, trial_ids)
                hga = self.feature_dict[(start, stop)]['hga_activity'][trial_start:trial_stop]
                lpc = self.feature_dict[(start, stop)][self.target_specifier][trial_start:trial_stop]
                if self.transform:
                    hga = self.transform(hga)
                return hga, lpc


# ============================================================================
# Model (from paper's local/models.py)
# ============================================================================

class BidirectionalSpeechSynthesisModel(nn.Module):
    def __init__(self, nb_layer=2, nb_hidden_units=100, nb_electrodes=128, dropout=0.0):
        super().__init__()
        self.nb_hidden_units = nb_hidden_units
        self.nb_layer = nb_layer
        self.lstm = nn.LSTM(input_size=nb_electrodes, hidden_size=nb_hidden_units, num_layers=nb_layer,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.regressor = nn.Linear(in_features=(2 * nb_hidden_units), out_features=20)

    def create_new_initial_state(self, batch_size, device="cpu", req_grad=False):
        return (torch.zeros(2 * self.nb_layer, batch_size, self.nb_hidden_units, requires_grad=req_grad, device=device),
                torch.zeros(2 * self.nb_layer, batch_size, self.nb_hidden_units, requires_grad=req_grad, device=device))

    def forward(self, x, state=None):
        if state is None:
            state = self.create_new_initial_state(batch_size=x.size(0), device=next(self.parameters()).device)
        x, new_state = self.lstm(x, state)
        out = self.regressor(x)
        return out, new_state


# ============================================================================
# Leave-one-day-out cross-validation (from paper's local/common.py)
# ============================================================================

class LeaveOneDayOut:
    def split(self, X, start_with_day=None):
        ordered_days = sorted(X)
        if start_with_day is not None:
            while ordered_days[0] != start_with_day:
                ordered_days.append(ordered_days.pop(0))
        for i in range(len(ordered_days)):
            test_day = ordered_days[i]
            train_days = [d for j, d in enumerate(ordered_days) if j != i]
            yield train_days, test_day


# ============================================================================
# Z-score normalization from syllable repetitions
# ============================================================================

def compute_zscore_stats(syllable_dir):
    """Compute per-day z-score statistics from syllable repetition recordings."""
    z_scores = {}
    syll_files = list(Path(syllable_dir).glob("**/SyllableRepetition_Overt.mat"))
    for syll_path in tqdm.tqdm(syll_files, desc="Computing z-score stats"):
        day = syll_path.parent.name
        mat = BCI2000MatFile(syll_path.as_posix())
        ecog = mat.signals()
        data = []
        for _, start, stop in mat.trial_indices():
            extractor = get_feature_extractor(mat)
            feats = extractor.extract_features(ecog[start:int(stop + (0.04 * mat.fs)), :])
            data.append(feats)
        norm_data = np.concatenate(data)
        z_scores[day] = (np.mean(norm_data, axis=0), np.std(norm_data, axis=0))
    return z_scores


# ============================================================================
# Main preprocessing: create HDF corpus
# ============================================================================

def prepare_corpus(keyword_dir, syllable_dir, corpus_dir):
    """Run the paper's exact preprocessing pipeline to create HDF files."""
    corpus_dir = Path(corpus_dir)
    keyword_dir = Path(keyword_dir)

    # Compute z-score normalization stats from syllable repetitions
    z_scores = compute_zscore_stats(syllable_dir)
    logger.info(f"Computed z-score stats for {len(z_scores)} days: {sorted(z_scores.keys())}")

    # Process all keyword reading files (follow symlinks)
    import glob as globmod
    mat_files = sorted([Path(p) for p in globmod.glob(str(keyword_dir / "**" / "KeywordReading_Overt_R*.mat"), recursive=True)])
    logger.info(f"Found {len(mat_files)} keyword reading files")

    for mat_file in tqdm.tqdm(mat_files, desc="Extracting features"):
        day = mat_file.parent.name
        wav_file = mat_file.with_suffix(".wav")

        if day not in z_scores:
            logger.warning(f"No normalization data for {day}. Skipping!")
            continue

        if not wav_file.exists():
            logger.warning(f"No wav file for {mat_file}. Skipping!")
            continue

        logger.info(f"Processing {mat_file.name} from {day}")
        hga, lpc, tids = extract_features_for_file(mat_file.as_posix(), wav_file.as_posix(), min_trial_length=2.5)

        # Z-score normalize
        mu, std = z_scores[day]
        hga = (hga - mu) / std

        # Save to HDF
        out_file = corpus_dir / day / mat_file.with_suffix('.hdf').name
        os.makedirs(out_file.parent, exist_ok=True)
        with h5py.File(out_file, 'w') as hf:
            hf.create_dataset('hga_activity', data=hga)
            hf.create_dataset('lpc_coefficients', data=lpc)
            hf.create_dataset('trial_ids', data=tids)
        logger.info(f"Saved {out_file}: HGA {hga.shape}, LPC {lpc.shape}, trials {len(set(abs(tids)))}")


# ============================================================================
# Training (matching paper's train_bidirectional_model.py exactly)
# ============================================================================

def train(corpus_dir, out_dir, test_day, val_day, nb_epochs, device):
    """Train the bidirectional speech synthesis model exactly as in the paper."""
    os.makedirs(out_dir, exist_ok=True)

    # Set up file logging
    fh = logging.FileHandler(os.path.join(out_dir, "training.log"), 'w+')
    fh.setFormatter(logging.Formatter('[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                                       datefmt='%d.%m.%y %H:%M:%S'))
    logging.getLogger().addHandler(fh)

    E = len(SelectElectrodesOverSpeechAreas())  # 64 channels
    logger.info(f"Number of speech channels: {E}")

    # Find all HDF files
    import glob as globmod
    feature_files = sorted([Path(p) for p in globmod.glob(str(Path(corpus_dir) / "**" / "KeywordReading_Overt_R*.hdf"), recursive=True)])
    groups_by_day = defaultdict(list)
    for f in feature_files:
        groups_by_day[f.parent.name].append(f)

    logger.info(f"Found {len(feature_files)} HDF files across {len(groups_by_day)} days: {sorted(groups_by_day.keys())}")

    # Leave-one-day-out split
    kf = LeaveOneDayOut()
    for train_days, test_d in kf.split(X=groups_by_day.keys(), start_with_day=test_day):
        kf_va = LeaveOneDayOut()
        train_days, val_d = next(kf_va.split(train_days, start_with_day=val_day))
        logger.info(f"Test day: {test_d}, Validation day: {val_d}")
        logger.info(f"Training days: {sorted(train_days)}")

        tr_files = [f.as_posix() for f in feature_files if f.parent.name in train_days]
        va_files = [f.as_posix() for f in feature_files if f.parent.name == val_d]
        te_files = sorted([f.as_posix() for f in feature_files if f.parent.name == test_d])

        # Datasets
        speech_sel = SelectElectrodesOverSpeechAreas()
        tr_dataset = SequentialSpeechTrials(feature_files=tr_files, transform=speech_sel)
        va_dataset = SequentialSpeechTrials(feature_files=va_files, transform=speech_sel)
        te_dataset = SequentialSpeechTrials(feature_files=te_files, transform=speech_sel)

        logger.info(f"Train: {len(tr_dataset)} trials, Val: {len(va_dataset)} trials, Test: {len(te_dataset)} trials")

        # DataLoaders - batch_size=1 as in paper
        dl_params = dict(batch_size=1, num_workers=4, pin_memory=True)
        tr_loader = DataLoader(tr_dataset, **dl_params, shuffle=True)
        va_loader = DataLoader(va_dataset, **dl_params, shuffle=True)
        te_loader = DataLoader(te_dataset, **dl_params, shuffle=False)

        # Model - exact paper config
        model = BidirectionalSpeechSynthesisModel(nb_layer=2, nb_hidden_units=100,
                                                   nb_electrodes=E, dropout=0.5)
        model.to(device)
        logger.info(f"Model on {device}")
        nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {nb_params:,}")

        summary = torchinfo.summary(model, input_size=[(1, 100, E)],
                                    col_names=['input_size', 'output_size', 'num_params'], verbose=0)
        with open(os.path.join(out_dir, 'model.network'), 'w+') as f:
            f.write(str(summary))

        # Optimizer - exact paper config
        optim = torch.optim.RMSprop(model.parameters(), lr=0.0001)
        cfunc = nn.MSELoss(reduction='mean')

        best_val_loss = float('inf')
        best_model_path = os.path.join(out_dir, "best_model.pth")
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(nb_epochs):
            # Training
            model.train()
            train_loss = 0.0
            n_train = 0
            pbar = tqdm.tqdm(tr_loader, desc=f"Epoch {epoch+1:03d}")
            for x, y in pbar:
                x = x.to(device).float()
                y = y.to(device).float()
                state = model.create_new_initial_state(batch_size=x.size(0), device=device, req_grad=True)

                for p in model.parameters():
                    p.grad = None

                pred, _ = model(x, state=state)
                loss = cfunc(pred, y)
                loss.backward()
                optim.step()

                train_loss += loss.item()
                n_train += x.size(0)
                pbar.set_postfix(train_loss=f"{train_loss/n_train:.4f}")

            final_train_loss = train_loss / n_train

            # Validation
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for x, y in va_loader:
                    x = x.to(device).float()
                    y = y.to(device).float()
                    state = model.create_new_initial_state(batch_size=x.size(0), device=device)
                    pred, _ = model(x, state)
                    loss = cfunc(pred, y)
                    val_loss += loss.item()
                    n_val += x.size(0)

            final_val_loss = val_loss / n_val
            history['train_loss'].append(final_train_loss)
            history['val_loss'].append(final_val_loss)

            logger.info(f"Epoch {epoch+1:03d}: Train={final_train_loss:.6f}, Val={final_val_loss:.6f}")

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  -> New best model (val_loss={best_val_loss:.6f})")

        # Evaluate on test set with best model
        logger.info("Loading best model for evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in te_loader:
                x = x.to(device).float()
                state = model.create_new_initial_state(batch_size=x.size(0), device=device)
                pred, _ = model(x, state)
                all_preds.append(pred.cpu().numpy().squeeze())
                all_targets.append(y.numpy().squeeze())

        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)

        # Compute per-feature correlation (paper's evaluation metric)
        correlations = []
        for i in range(preds.shape[1]):
            r = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
            correlations.append(r)
        mean_corr = np.mean(correlations)
        logger.info(f"Test set LPC correlation (mean across 20 features): {mean_corr:.4f}")
        logger.info(f"Per-feature correlations: {[f'{c:.3f}' for c in correlations]}")

        # Save predictions for LPCNet synthesis
        os.makedirs(os.path.join(out_dir, "reco"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "orig"), exist_ok=True)
        np.save(os.path.join(out_dir, "reco", "test_predictions.npy"), preds)
        np.save(os.path.join(out_dir, "orig", "test_targets.npy"), targets)

        # Synthesize audio from predicted LPC coefficients
        logger.info("Synthesizing audio from predicted LPC coefficients...")
        try:
            net = LPCNet()
            pred_audio = np.hstack([net.synthesize(frame) for frame in preds.astype(np.float32)])
            wavwrite(os.path.join(out_dir, "reco", "test_synthesized.wav"), 16000, pred_audio)

            net2 = LPCNet()
            orig_audio = np.hstack([net2.synthesize(frame) for frame in targets.astype(np.float32)])
            wavwrite(os.path.join(out_dir, "orig", "test_original.wav"), 16000, orig_audio)
            logger.info("Audio synthesis complete!")
        except Exception as e:
            logger.error(f"Audio synthesis failed: {e}")

        # Save metrics
        metrics = {
            'test_day': test_d,
            'val_day': val_d,
            'train_days': sorted(train_days),
            'nb_train_trials': len(tr_dataset),
            'nb_val_trials': len(va_dataset),
            'nb_test_trials': len(te_dataset),
            'nb_epochs': nb_epochs,
            'best_val_loss': best_val_loss,
            'mean_lpc_correlation': mean_corr,
            'per_feature_correlations': correlations,
            'history': history,
            'model_config': {
                'nb_layer': 2, 'nb_hidden_units': 100, 'nb_electrodes': E,
                'dropout': 0.5, 'optimizer': 'RMSprop', 'lr': 0.0001, 'batch_size': 1
            }
        }
        with open(os.path.join(out_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Results saved to {out_dir}")
        break  # Only train for the first fold (matching paper)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate Angrick et al. 2023 speech synthesis training")
    parser.add_argument("--keyword_dir", required=True, help="Directory with KeywordReading .mat/.wav files")
    parser.add_argument("--syllable_dir", required=True, help="Directory with SyllableRepetition .mat files")
    parser.add_argument("--corpus_dir", default="/tmp/paper_corpus", help="Directory for preprocessed HDF files")
    parser.add_argument("--out_dir", default="/mnt/home/vincent.wilmet/results_paper_replication",
                        help="Output directory for model and results")
    parser.add_argument("--test_day", default="2022_11_03")
    parser.add_argument("--val_day", default="2022_11_04")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (after CUDA_VISIBLE_DEVICES remapping)")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing if corpus already exists")
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Step 1: Preprocessing (create HDF corpus)
    if not args.skip_preprocess:
        logger.info("Step 1: Preprocessing - extracting HG features and LPC coefficients")
        prepare_corpus(args.keyword_dir, args.syllable_dir, args.corpus_dir)
    else:
        logger.info("Skipping preprocessing (--skip_preprocess)")

    # Step 2: Training
    logger.info("Step 2: Training bidirectional speech synthesis model")
    train(args.corpus_dir, args.out_dir, args.test_day, args.val_day, args.epochs, device)
