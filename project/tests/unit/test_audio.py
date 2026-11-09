"""Buffer circular e reamostragem em fluxo (sem perder nem duplicar amostras)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from ddsd.audio.resampler import StreamResampler
from ddsd.audio.ring_buffer import AudioRingBuffer


def test_ring_buffer_preserves_order_across_wrap() -> None:
    ring = AudioRingBuffer(capacity=100)
    signal = np.arange(1000, dtype=np.float32)
    received = []
    for start in range(0, 1000, 37):
        ring.write(signal[start : start + 37], captured_at=float(start))
        read = ring.read(timeout=0)
        assert read is not None and read.lost_before == 0
        assert read.start == start
        received.append(read.samples)
    assert np.array_equal(np.concatenate(received), signal)
    assert ring.stats().overrun_samples == 0


def test_ring_buffer_counts_overrun_and_exposes_gap() -> None:
    ring = AudioRingBuffer(capacity=100)
    ring.write(np.arange(150, dtype=np.float32), captured_at=1.0)  # leitor parado: 50 perdidas
    read = ring.read(timeout=0)
    assert read is not None
    assert read.lost_before == 50 and read.start == 50
    assert np.array_equal(read.samples, np.arange(50, 150, dtype=np.float32))
    assert ring.stats().overrun_samples == 50


def test_ring_buffer_read_timeout_returns_none() -> None:
    assert AudioRingBuffer(10).read(timeout=0.01) is None


@pytest.mark.parametrize("capacity", [480_000, 4_096])
def test_ring_buffer_concurrent_writer_reader(capacity: int) -> None:
    """Escritor sem ritmo (muito acima do tempo real) contra um leitor concorrente.

    Com capacidade suficiente, nada se perde. Com capacidade pequena, pode haver
    perda, mas ela é sempre contada, e tudo o que chega está na posição absoluta
    correta: recebidas + perdidas = total.
    """
    ring = AudioRingBuffer(capacity=capacity)
    total = 480_000
    signal = np.arange(total, dtype=np.float32)

    def writer() -> None:
        for start in range(0, total, 512):
            ring.write(signal[start : start + 512], captured_at=0.0)

    thread = threading.Thread(target=writer)
    thread.start()
    received = lost = 0
    expected_start = 0
    while received + lost < total:
        read = ring.read(timeout=2.0)
        assert read is not None, "leitor ficou sem dados"
        assert read.start == expected_start + read.lost_before
        assert np.array_equal(read.samples, signal[read.start : read.start + len(read.samples)])
        received += len(read.samples)
        lost += read.lost_before
        expected_start = read.start + len(read.samples)
    thread.join()
    assert received + lost == total
    assert lost == ring.stats().overrun_samples
    if capacity >= total:
        assert lost == 0


@pytest.mark.parametrize(("in_rate", "out_rate"), [(48_000, 16_000), (44_100, 48_000), (16_000, 16_000)])
def test_stream_resampler_matches_one_shot(in_rate: int, out_rate: int) -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(in_rate * 2).astype(np.float32) * 0.1
    one_shot = StreamResampler(in_rate, out_rate)
    expected = np.concatenate([one_shot.process(signal), one_shot.process(np.zeros(0, np.float32), last=True)])
    chunked = StreamResampler(in_rate, out_rate)
    pieces, position = [], 0
    for size in rng.integers(1, 3000, size=10_000):
        if position >= len(signal):
            break
        pieces.append(chunked.process(signal[position : position + size]))
        position += size
    pieces.append(chunked.process(np.zeros(0, np.float32), last=True))
    got = np.concatenate(pieces)
    assert len(got) == len(expected)
    assert np.allclose(got, expected, atol=1e-5)
    assert abs(len(got) - len(signal) * out_rate / in_rate) <= 2
