import sounddevice as sd
import numpy as np
from fastrtc import get_tts_model, KokoroTTSOptions, get_stt_model
from threading import Event, Thread
import queue

# Initialize TTS and STT models
tts_model = get_tts_model()
stt_model = get_stt_model()

# Global variables for audio processing
is_recording = False
stop_recording = Event()

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

def speech_to_text(max_duration: int = 30, silence_threshold: float = 0.03, silence_duration: float = 1.0) -> str:
    """Records audio with silence detection and converts it to text using FastRTC's STT
    
    Args:
        max_duration (int): Maximum recording duration in seconds
        silence_threshold (float): Audio level below which is considered silence
        silence_duration (float): Duration of silence (seconds) before stopping
        
    Returns:
        str: Transcribed text
    """
    global is_recording, stop_recording
    
    print("Listening... Speak now!")
    
    audio_chunks = []
    silence_start = None
    sample_rate = 16000
    chunk_duration = 0.1  # seconds
    chunk_samples = int(chunk_duration * sample_rate)
    
    # Setup the input stream
    stream = sd.InputStream(samplerate=sample_rate, channels=1, 
                          callback=lambda indata, frames, time, status: audio_chunks.append(indata.copy()),
                          blocksize=chunk_samples)
    
    is_recording = True
    stop_recording.clear()
    
    with stream:
        total_chunks = int(max_duration / chunk_duration)
        for _ in range(total_chunks):
            if stop_recording.is_set():
                break
                
            if len(audio_chunks) > 0:
                current_chunk = audio_chunks[-1]
                amplitude = np.max(np.abs(current_chunk))
                
                if amplitude < silence_threshold:
                    if silence_start is None:
                        silence_start = len(audio_chunks) * chunk_duration
                    elif (len(audio_chunks) * chunk_duration - silence_start) >= silence_duration:
                        break
                else:
                    silence_start = None
            
            sd.sleep(int(chunk_duration * 1000))
    
    is_recording = False
    
    if not audio_chunks:
        return ""
    
    # Combine all chunks and convert to float32
    audio_data = np.concatenate(audio_chunks).flatten().astype(np.float32)
    
    # Normalize audio if needed
    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Create the audio tuple expected by the STT model
    audio_tuple = (sample_rate, audio_data)
    
    # Use the STT model
    text = stt_model.stt(audio_tuple)
    print(f"Recognized: {text}")
    return text

def stop_recording_audio():
    """Stop the current recording session"""
    global is_recording, stop_recording
    if is_recording:
        stop_recording.set()

def is_mic_active() -> bool:
    """Check if the microphone is currently recording"""
    global is_recording
    return is_recording

if __name__ == "__main__":
    # Test TTS
    """ text = "Hello welcome john doe, today we are going to do your interview for a software developer position at techorp, are you ready?"
    text_to_speech_and_play(text) """
    
    # Test STT
    recognized_text = speech_to_text()
    print(f"You said: {recognized_text}")
