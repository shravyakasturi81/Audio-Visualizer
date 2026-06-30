from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from visualizer import AudioLoadError, generate_mel_spectrogram_bytes


def safe_delete_temp_file(file_path: Path) -> None:
    """Best-effort cleanup for temporary files on Windows.

    Some audio decoders may keep a short-lived file handle, causing unlink to fail.
    """
    try:
        file_path.unlink(missing_ok=True)
    except PermissionError:
        # Avoid crashing the app if the OS still holds a lock on temp files.
        pass
    except OSError:
        # Any other filesystem cleanup issue should not break the UI flow.
        pass


st.set_page_config(page_title="Audio Visualizer", page_icon="🎵", layout="wide")

st.title("Audio Visualizer")
st.write("Upload an audio file and generate a mel spectrogram with custom settings.")

with st.sidebar:
    st.header("Spectrogram Settings")
    sample_rate = st.selectbox("Sample Rate", options=[8000, 16000, 22050, 32000, 44100], index=2)
    n_fft = st.selectbox("FFT Size (n_fft)", options=[512, 1024, 2048, 4096], index=2)
    hop_length = st.selectbox("Hop Length", options=[128, 256, 512, 1024], index=2)
    n_mels = st.slider("Mel Bands", min_value=32, max_value=512, value=128, step=32)
    fmin = st.number_input("Min Frequency (Hz)", min_value=0.0, value=0.0, step=10.0)
    fmax_input = st.number_input("Max Frequency (Hz, 0 for auto)", min_value=0.0, value=0.0, step=100.0)
    cmap = st.selectbox(
        "Colormap",
        options=["magma", "viridis", "plasma", "inferno", "cividis", "turbo"],
        index=0,
    )
    max_duration_input = st.number_input(
        "Max Duration (seconds, 0 for full)", min_value=0.0, value=0.0, step=1.0
    )

uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    col1, col2 = st.columns([1, 3])
    with col1:
        generate_button = st.button("Generate Spectrogram", use_container_width=True)

    if generate_button:
        fmax = None if fmax_input == 0 else float(fmax_input)
        max_duration = None if max_duration_input == 0 else float(max_duration_input)
        image_bytes: bytes | None = None

        with st.spinner("Generating visualization..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_audio_path = Path(tmp.name)

            try:
                image_bytes = generate_mel_spectrogram_bytes(
                    input_audio=temp_audio_path,
                    sample_rate=int(sample_rate),
                    n_fft=int(n_fft),
                    hop_length=int(hop_length),
                    n_mels=int(n_mels),
                    fmin=float(fmin),
                    fmax=fmax,
                    cmap=str(cmap),
                    max_duration=max_duration,
                )
            except AudioLoadError as exc:
                st.error(str(exc))
            except Exception:
                st.error("Unexpected error while generating spectrogram. Try a different file or settings.")
            finally:
                safe_delete_temp_file(temp_audio_path)

        if image_bytes is not None:
            st.success("Spectrogram generated")
            st.image(image_bytes, caption="Mel Spectrogram", use_container_width=True)
            st.download_button(
                label="Download PNG",
                data=image_bytes,
                file_name="mel_spectrogram.png",
                mime="image/png",
                use_container_width=True,
            )
else:
    st.info("Upload a file to begin.")
