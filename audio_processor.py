import time
import threading
import av
import numpy as np
import queue
from typing import List, Dict, Any, Optional
from fastrtc import get_stt_model
from streamlit_webrtc import WebRtcMode, WebRtcStreamerContext, AudioProcessorBase

# Global speech recognition model
stt_model = get_stt_model()

# Constants for transcription
SAMPLE_RATE = 16000
MIN_AUDIO_LEVEL = 0.01  # Minimum audio level to consider as speech
TRANSCRIPT_UPDATE_INTERVAL = 0.5  # Update transcript more frequently (0.5 seconds)
MAX_SILENCE_DURATION = 1.5  # Shorter silence duration to be more responsive
MIN_TRANSCRIPT_LENGTH = 3  # Minimum length of transcript to consider valid

class AudioProcessor(AudioProcessorBase):
    """Audio processor for real-time speech-to-text transcription"""
    
    def __init__(self, 
                 shared_state,
                 enable_auto_send: bool = True,
                 silence_threshold: float = 0.01,
                 update_interval: float = 0.5):
        """
        Initialize the audio processor
        
        Args:
            shared_state: Shared state instance for communication
            enable_auto_send: Whether to automatically send transcripts to chat
            silence_threshold: Threshold for detecting silence
            update_interval: Interval for updating the transcript in seconds
        """
        self.shared_state = shared_state
        self.enable_auto_send = enable_auto_send
        self.silence_threshold = silence_threshold
        self.update_interval = update_interval
        
        # Audio buffer and processing state
        self.audio_buffer = []
        self.current_transcript = ""
        self.final_transcript = ""
        self.previous_transcripts = []
        self.is_speaking = False
        self.last_speech_time = time.time()
        self.last_update_time = time.time()
        self.last_audio_level = 0.0
        self.transcript_lock = threading.Lock()
        self.transcript_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.continuous_silence_duration = 0
        
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_audio_buffer)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frames"""
        try:
            # Convert audio frame to numpy array
            audio_array = frame.to_ndarray()
            
            # Check audio level
            audio_level = np.abs(audio_array).mean()
            self.last_audio_level = audio_level
            
            # Detect speech with improved logic
            if audio_level > self.silence_threshold:
                self.is_speaking = True
                self.last_speech_time = time.time()
                self.continuous_silence_duration = 0
            else:
                # Track continuous silence
                current_time = time.time()
                if current_time - self.last_speech_time > 0.1:  # Check in small increments
                    self.continuous_silence_duration += current_time - self.last_speech_time
                
                # Only mark as not speaking after continuous silence
                if self.continuous_silence_duration > MAX_SILENCE_DURATION:
                    self.is_speaking = False
            
            # Add audio to buffer
            self.audio_buffer.append((SAMPLE_RATE, audio_array.flatten().astype(np.float32)))
            
            # Update transcript more frequently when speaking
            current_time = time.time()
            update_due = current_time - self.last_update_time > self.update_interval
            
            if update_due or (self.is_speaking and len(self.audio_buffer) > 5):
                self.last_update_time = current_time
                self._update_transcript()
            
            # Return the frame unchanged
            return frame
        except Exception as e:
            print(f"Error in recv: {str(e)}")
            return frame
    
    def _update_transcript(self):
        """Update the transcript from the current audio buffer"""
        if not self.audio_buffer:
            return
            
        try:
            # Check if queue is too full - clear oldest item if needed
            if self.transcript_queue.full():
                try:
                    self.transcript_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Queue the audio data for processing in the background thread
            self.transcript_queue.put(self.audio_buffer.copy(), block=False)
            
            # Clear buffer after queueing
            self.audio_buffer = []
        except Exception as e:
            print(f"Error queueing audio for transcription: {str(e)}")
    
    def _process_audio_buffer(self):
        """Background thread that processes audio and updates transcripts"""
        while self.running:
            try:
                # Wait for audio data with a timeout
                try:
                    audio_data = self.transcript_queue.get(timeout=0.5)
                except queue.Empty:
                    # Check if we should finalize transcript due to silence
                    if (not self.is_speaking and 
                        time.time() - self.last_speech_time > MAX_SILENCE_DURATION and 
                        self.current_transcript.strip()):
                        self._finalize_transcript()
                    continue
                
                if not audio_data:
                    continue
                    
                # Combine audio chunks
                combined_audio = None
                for sample_rate, chunk in audio_data:
                    if combined_audio is None:
                        combined_audio = chunk
                    else:
                        combined_audio = np.concatenate((combined_audio, chunk))
                
                if combined_audio is None or len(combined_audio) <= 100:  # Skip very short audio
                    continue
                
                # Normalize audio
                max_abs_value = np.max(np.abs(combined_audio))
                if max_abs_value > 1.0:
                    combined_audio = combined_audio / max_abs_value
                
                # Transcribe audio
                audio_tuple = (SAMPLE_RATE, combined_audio)
                try:
                    transcript = stt_model.stt(audio_tuple)
                    
                    if transcript and transcript.strip():
                        transcript = transcript.strip()
                        
                        # Process the transcript with debouncing and duplicate detection
                        self._update_current_transcript(transcript)
                except Exception as e:
                    print(f"Transcription error: {str(e)}")
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                time.sleep(0.1)
    
    def _update_current_transcript(self, new_transcript):
        """Update the current transcript with smarter text merging"""
        with self.transcript_lock:
            # Skip very short transcripts unless the current transcript is empty
            if len(new_transcript) < MIN_TRANSCRIPT_LENGTH and self.current_transcript:
                return
            
            # Check if this is a duplicate of what we already have
            if self._is_duplicate(new_transcript):
                return
                
            # If current transcript is empty, just use the new one
            if not self.current_transcript:
                self.current_transcript = new_transcript
                print(f"New transcript: {self.current_transcript}")
                return
                
            # Check if new transcript is a longer version of the current one
            if new_transcript.lower() in self.current_transcript.lower():
                # New transcript is contained in current, do nothing
                return
            elif self.current_transcript.lower() in new_transcript.lower():
                # Current transcript is contained in new, replace with new
                self.current_transcript = new_transcript
                print(f"Updated with longer transcript: {self.current_transcript}")
                return
            
            # Check if new transcript is a continuation by looking for overlap
            overlap = self._find_overlap(self.current_transcript, new_transcript)
            if overlap and len(overlap) > 3:
                # There's significant overlap, append only the new part
                self.current_transcript = self.current_transcript + new_transcript[len(overlap):]
                print(f"Appended transcript with overlap: {self.current_transcript}")
            else:
                # No significant overlap, append with space
                self.current_transcript += " " + new_transcript
                print(f"Appended transcript: {self.current_transcript}")
            
            # Check if we should finalize due to silence
            if not self.is_speaking and time.time() - self.last_speech_time > MAX_SILENCE_DURATION:
                self._finalize_transcript()
    
    def _is_duplicate(self, text):
        """Check if text is a duplicate of recent transcripts"""
        # Check against current transcript
        if text.lower() == self.current_transcript.lower():
            return True
            
        # Check against previous transcripts (to prevent sending same thing repeatedly)
        for prev in self.previous_transcripts[-3:]:  # Check last 3
            if text.lower() == prev.lower():
                return True
        
        return False
    
    def _find_overlap(self, text1, text2):
        """Find the largest overlap between end of text1 and start of text2"""
        # Check for overlap of at least 3 characters
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Try different overlap lengths, starting from the longest possible
        max_overlap = min(len(text1), len(text2))
        for i in range(max_overlap, 2, -1):
            if text1_lower[-i:] == text2_lower[:i]:
                return text2[:i]
        
        return ""
    
    def _finalize_transcript(self):
        """Finalize the transcript and send it to the chat"""
        with self.transcript_lock:
            if not self.current_transcript:
                return
                
            # Clean up the transcript
            transcript = self.current_transcript.strip()
            
            # Only process if it's a meaningful transcript
            if len(transcript) >= MIN_TRANSCRIPT_LENGTH:
                self.final_transcript = transcript
                
                # Add to previous transcripts to avoid duplicates
                self.previous_transcripts.append(transcript)
                if len(self.previous_transcripts) > 10:
                    self.previous_transcripts = self.previous_transcripts[-10:]
                
                # Auto-send to chat if enabled
                if self.enable_auto_send:
                    print(f"Auto-sending transcript: '{self.final_transcript}'")
                    self.shared_state.set_user_response(self.final_transcript)
            
            # Reset current transcript
            self.current_transcript = ""
    
    def get_transcript(self) -> str:
        """Get the current transcript"""
        with self.transcript_lock:
            return self.current_transcript
    
    def get_final_transcript(self) -> str:
        """Get the finalized transcript"""
        with self.transcript_lock:
            return self.final_transcript
            
    def reset_transcript(self):
        """Reset the current transcript"""
        with self.transcript_lock:
            self.current_transcript = ""
    
    def send_transcript_to_chat(self):
        """Manually send the current transcript to chat"""
        with self.transcript_lock:
            if self.current_transcript.strip():
                transcript = self.current_transcript.strip()
                
                # Only send if it's a meaningful transcript
                if len(transcript) >= MIN_TRANSCRIPT_LENGTH:
                    # Add to previous to avoid duplicates
                    self.previous_transcripts.append(transcript)
                    if len(self.previous_transcripts) > 10:
                        self.previous_transcripts = self.previous_transcripts[-10:]
                    
                    self.shared_state.set_user_response(transcript)
                    self.final_transcript = transcript
                    self.current_transcript = ""
                    return True
            return False
    
    def is_active(self) -> bool:
        """Check if the audio processor is detecting speech"""
        return self.is_speaking
    
    def get_audio_level(self) -> float:
        """Get the current audio level"""
        return self.last_audio_level
    
    def close(self):
        """Clean up resources"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)


def create_audio_processor(shared_state, enable_auto_send=True):
    """Factory function to create an audio processor"""
    return lambda: AudioProcessor(shared_state, enable_auto_send)