"""Neural network models for speech BCI.

Models match the paper's GitHub code (corrected from paper text):
  - nVAD: Uni-LSTM-150, 2-layer, frame-level binary output
  - Acoustic decoder: BiLSTM-100/dir, 2-layer, 20-dim LPC output
  - Word classifier: BiLSTM-128, 2-layer (for validation only)
"""

import torch
import torch.nn as nn
from torch import Tensor

from . import config


class UnidirectionalVAD(nn.Module):
    """Unidirectional LSTM for causal voice activity detection.

    Must be causal (unidirectional) because it runs in real-time to detect
    when speech starts and stops. Frame-level output, stateful across chunks.

    Paper's code: 2-layer, 150 hidden, 0.5 dropout, ~311K params.
    """

    def __init__(self, n_electrodes: int = config.GRID_SIZE,
                 hidden_size: int = config.NVAD_HIDDEN,
                 num_layers: int = config.NVAD_LAYERS,
                 dropout: float = config.NVAD_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            n_electrodes, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 2)  # [silence, speech]

    def forward(self, x: Tensor,
                state: tuple[Tensor, Tensor] | None = None
                ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_electrodes)
        state : optional (h, c) hidden state from previous chunk

        Returns
        -------
        logits : (batch, seq_len, 2)
        new_state : (h, c) for next chunk
        """
        out, new_state = self.lstm(x, state)
        logits = self.head(out)
        return logits, new_state


class BidirectionalAcousticDecoder(nn.Module):
    """Bidirectional LSTM for LPC coefficient prediction.

    Bidirectional because it decodes AFTER the full utterance is spoken
    (delayed synthesis). The full utterance is available, so looking
    backward and forward improves accuracy.

    Paper's code: 2-layer BiLSTM, 100 hidden/dir, 0.5 dropout, ~378K params.
    Single linear projection (no ReLU/dropout between LSTM and output).
    """

    def __init__(self, n_electrodes: int = config.GRID_SIZE,
                 hidden_size: int = config.DECODER_HIDDEN,
                 num_layers: int = config.DECODER_LAYERS,
                 dropout: float = config.DECODER_DROPOUT,
                 n_output: int = config.N_LPC):
        super().__init__()
        self.lstm = nn.LSTM(
            n_electrodes, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.proj = nn.Linear(hidden_size * 2, n_output)

    def create_initial_state(self, batch_size: int, device: str = "cpu",
                             req_grad: bool = False):
        """Create fresh initial hidden state (matches paper's create_new_initial_state)."""
        h = torch.zeros(2 * self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                        requires_grad=req_grad, device=device)
        c = torch.zeros(2 * self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                        requires_grad=req_grad, device=device)
        return (h, c)

    def forward(self, x: Tensor,
                state: tuple[Tensor, Tensor] | None = None
                ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_electrodes)
        state : optional LSTM state

        Returns
        -------
        predictions : (batch, seq_len, n_output)
        new_state : LSTM state
        """
        if state is None:
            state = self.create_initial_state(x.size(0), device=x.device)
        out, new_state = self.lstm(x, state)
        predictions = self.proj(out)
        return predictions, new_state


class WordClassifier(nn.Module):
    """BiLSTM word classifier (for validation/sanity checking only).

    This cannot generalize to unseen words — it's only useful for
    verifying that the pipeline works.
    """

    def __init__(self, n_electrodes: int = config.GRID_SIZE,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 n_classes: int = config.N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            n_electrodes, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_electrodes)
        lengths : (batch,) sequence lengths for packing

        Returns
        -------
        logits : (batch, n_classes)
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            _, (h, _) = self.lstm(packed)
        else:
            _, (h, _) = self.lstm(x)

        # Concatenate final hidden states from both directions
        out = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(out)


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_nvad(n_electrodes: int = config.GRID_SIZE) -> UnidirectionalVAD:
    """Build nVAD with paper's configuration."""
    model = UnidirectionalVAD(n_electrodes=n_electrodes)
    print(f"nVAD: {count_params(model):,} params")
    return model


def build_acoustic_decoder(n_electrodes: int = config.GRID_SIZE) -> BidirectionalAcousticDecoder:
    """Build acoustic decoder with paper's configuration."""
    model = BidirectionalAcousticDecoder(n_electrodes=n_electrodes)
    print(f"Acoustic decoder: {count_params(model):,} params")
    return model


def build_word_classifier(n_electrodes: int = config.GRID_SIZE) -> WordClassifier:
    """Build word classifier for validation."""
    model = WordClassifier(n_electrodes=n_electrodes)
    print(f"Word classifier: {count_params(model):,} params")
    return model
