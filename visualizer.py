from __future__ import annotations

import argparse
import warnings
from io import BytesIO
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class AudioLoadError(ValueError):
    """Raised when an audio file cannot be decoded for visualization."""


def generate_mel_spectrogram(
    input_audio: Path,
    output_image: Path,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    cmap: str = "magma",
    max_duration: float | None = None,
) -> None:
    """Create a mel spectrogram image from an audio file."""
    fig = build_mel_spectrogram_figure(
        input_audio=input_audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        cmap=cmap,
        max_duration=max_duration,
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_image)
    plt.close(fig)


def build_mel_spectrogram_figure(
    input_audio: Path,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    cmap: str = "magma",
    max_duration: float | None = None,
) -> plt.Figure:
    """Build and return a matplotlib figure containing the mel spectrogram."""
    try:
        # Librosa may fallback to audioread for some formats; silence its deprecation warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*librosa\.core\.audio\.__audioread_load.*",
                category=FutureWarning,
            )
            y, sr = librosa.load(input_audio, sr=sample_rate, mono=True, duration=max_duration)
    except Exception as exc:
        raise AudioLoadError(
            "Could not decode this audio file. Try WAV/MP3/FLAC, or re-export the file with a standard codec."
        ) from exc

    if y.size == 0:
        raise AudioLoadError("Audio could not be loaded or is empty.")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    img = librosa.display.specshow(
        mel_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        cmap=cmap,
        ax=ax,
    )
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")
    fig.tight_layout()
    return fig


def generate_mel_spectrogram_bytes(
    input_audio: Path,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    cmap: str = "magma",
    max_duration: float | None = None,
) -> bytes:
    """Return the mel spectrogram image as PNG bytes for UI use."""
    fig = build_mel_spectrogram_figure(
        input_audio=input_audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        cmap=cmap,
        max_duration=max_duration,
    )

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a mel spectrogram visualization from an audio file."
    )
    parser.add_argument("input_audio", type=Path, help="Path to input audio file (.wav, .mp3, etc.).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/mel_spectrogram.png"),
        help="Path to save the output image.",
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate.")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT window size.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length between frames.")
    parser.add_argument("--n-mels", type=int, default=128, help="Number of mel bands.")
    parser.add_argument("--fmin", type=float, default=0.0, help="Minimum frequency in Hz.")
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Maximum frequency in Hz. Defaults to Nyquist frequency.",
    )
    parser.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap.")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional max duration in seconds to visualize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {args.input_audio}")

    generate_mel_spectrogram(
        input_audio=args.input_audio,
        output_image=args.output,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        cmap=args.cmap,
        max_duration=args.max_duration,
    )

    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
