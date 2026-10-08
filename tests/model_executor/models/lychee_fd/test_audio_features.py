# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.lychee_fd.audio_features import (
    N_MELS,
    WINDOW_SAMPLES,
    alternating_model_positions,
    complete_audio_windows,
    feature_token_count,
    log_mel_spectrogram,
    pad_mel_features,
    valid_mel_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_full_window_has_reference_shape_and_tick_count() -> None:
    mel = log_mel_spectrogram(torch.zeros(WINDOW_SAMPLES))

    assert mel.shape == (N_MELS, 42)
    assert valid_mel_frames(mel.shape[1]) == 40
    assert feature_token_count(mel.shape[1]) == 5
    assert alternating_model_positions(mel.shape[1]) == 10


def test_batched_dynamic_range_is_isolated_per_window() -> None:
    generator = torch.Generator().manual_seed(7)
    loud = torch.randn(WINDOW_SAMPLES, generator=generator)
    quiet = torch.randn(WINDOW_SAMPLES, generator=generator) * 1e-3

    batched = log_mel_spectrogram(torch.stack((loud, quiet)))

    torch.testing.assert_close(batched[0], log_mel_spectrogram(loud))
    torch.testing.assert_close(batched[1], log_mel_spectrogram(quiet))


def test_complete_windows_leave_partial_tail_unconsumed() -> None:
    audio = torch.arange(WINDOW_SAMPLES * 2 + 17)
    windows = complete_audio_windows(audio)

    assert windows.shape == (2, WINDOW_SAMPLES)
    torch.testing.assert_close(windows.reshape(-1), audio[: WINDOW_SAMPLES * 2])


def test_padding_preserves_per_window_valid_lengths() -> None:
    first = torch.zeros(N_MELS, 42)
    second = torch.zeros(N_MELS, 30)

    padded, lengths = pad_mel_features([first, second])

    assert padded.shape == (2, N_MELS, 42)
    assert lengths.tolist() == [40, 28]
