from fastrtc import get_tts_model, KokoroTTSOptions, get_stt_model
import sounddevice as sd
import numpy as np

# Initialize TTS and STT models
tts_model = get_tts_model()
stt_model = get_stt_model()

def text_to_speech_and_play(text: str):
    """Converts text to speech and plays it"""
    options = KokoroTTSOptions(
        voice="af_heart",
        speed=1.0,
        lang="en-us"
    )
    
    print(f"Playing: '{text}'")
    
    for chunk in tts_model.stream_tts_sync(text, options):
        sample_rate, audio_data = chunk
        #TODO : for now it just plays the audio with device's default output but we gonna make it interact with a web app (reask about this)
        sd.play(audio_data.T, sample_rate)
        sd.wait()

#TODO : find a better way to handle audio input so that candidate audio is not interrupted (cutted)
#NOTE : this might be a feature can be done in the frontend and returns the audio data
def speech_to_text() -> str:
    """Records audio and converts it to text using FastRTC's STT"""
    print("Listening... Speak now!")
    
    # Record audio using sounddevice
    #TODO: threshhold adding 
    duration = 5  # seconds
    sample_rate = 16000
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    
    # Convert to float32 and flatten
    audio_data = recording.flatten().astype(np.float32)
    
    # Normalize audio if needed
    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Create the audio tuple expected by the STT model
    audio_tuple = (sample_rate, audio_data)
    
    # Use the STT model
    text = stt_model.stt(audio_tuple)
    print(f"Recognized: {text}")
    return text

if __name__ == "__main__":
    # Test TTS
    """ text = "Hello welcome john doe, today we are going to do your interview for a software developer position at techorp, are you ready?"
    text_to_speech_and_play(text) """
    
    # Test STT
    recognized_text = speech_to_text()
    print(f"You said: {recognized_text}")
