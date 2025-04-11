import streamlit as st
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import os
import torch
import time
from pydub import AudioSegment
from dotenv import load_dotenv

# Load .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Chunked Diarization", layout="wide")
st.title("⚡ Chunked Whisper Large-v3 + PyAnnote Diarization")

st.sidebar.markdown("### 🧠 System Info")
st.sidebar.write(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")

@st.cache_resource
def load_whisper():
    return WhisperModel("large-v3", compute_type="float16")

@st.cache_resource
def load_pyannote():
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline

# Upload
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
full_audio = st.sidebar.toggle("Process full audio in 60s chunks?", value=True)

if st.sidebar.button("Transcribe & Diarize"):
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        audio = AudioSegment.from_file(audio_path)
        chunk_duration = 60 * 1000
        total_chunks = len(audio) // chunk_duration + (1 if len(audio) % chunk_duration > 0 else 0)

        whisper_model = load_whisper()
        diarization_pipeline = load_pyannote()

        transcript = []
        output_box = st.empty()

        for i in range(total_chunks if full_audio else 1):
            st.sidebar.write(f"⏳ Processing chunk {i+1}/{total_chunks}")

            start_ms = i * chunk_duration
            end_ms = min((i + 1) * chunk_duration, len(audio))
            chunk = audio[start_ms:end_ms]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                chunk.export(chunk_file.name, format="wav")
                chunk_path = chunk_file.name

            segments, _ = whisper_model.transcribe(chunk_path, beam_size=1)
            segments = list(segments)

            diarization = diarization_pipeline(chunk_path)

            for seg in segments:
                start, end, text = seg.start, seg.end, seg.text.strip()
                speaker = "Unknown"
                for turn, _, label in diarization.itertracks(yield_label=True):
                    if turn.start <= start <= turn.end or turn.start <= end <= turn.end:
                        speaker = label
                        break
                absolute_start = (start_ms / 1000) + start
                absolute_end = (start_ms / 1000) + end
                line = f"[{speaker}] ({absolute_start:.2f}-{absolute_end:.2f}s): {text}"
                transcript.append(line)
                output_box.markdown("\n\n".join(transcript))

            os.remove(chunk_path)

        os.remove(audio_path)
        st.sidebar.success("✅ All chunks processed.")

        st.markdown("---")
        st.download_button("📥 Download Transcript as .txt", "\n".join(transcript), file_name="transcript.txt")
    else:
        st.sidebar.error("Please upload an audio file.")
