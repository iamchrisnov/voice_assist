import whisper
import streamlit as st
from audiorecorder import audiorecorder

model = whisper.load_model("tiny")

def transcribe(audio):
    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text

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
        transcription = transcribe(audio_filename)
        st.text(transcription["text"])


# potentially use gdrive as a storage: https://medium.com/@annissouames99/how-to-upload-files-automatically-to-drive-with-python-ee19bb13dda


