import whisper
import streamlit as st
from audiorecorder import audiorecorder

model = whisper.load_model("tiny")
print("load complete")
st.title("Audio Recorder")
audio = audiorecorder("Click to record", "Recording...")
audio_filename = "testing_audio_file.wav"

if len(audio) > 0:
    # To play audio in frontend:
    st.audio(audio)
    
    # To save audio to a file:
    wav_file = open(audio_filename, "wb")
    wav_file.write(audio.tobytes())

    if st.button("Transcribe"):
        transcription = model.transcribe(audio_filename)
        st.text(transcription["text"])

