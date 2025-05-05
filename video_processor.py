import os
import av
import cv2
import time
import threading
from PIL import Image
import google.generativeai as genai
from streamlit_webrtc import VideoProcessorBase

# Global variables
llm_response = None
latest_analysis = "No analysis yet"  # Variable to store the latest analysis
processing_lock = threading.Lock()

# Configuration for frame collection and analysis
ANALYSIS_INTERVAL = 10.0  # Analyze a frame every 10 seconds

def analyze_interview_behavior(frame, model):
    """Analyze a single frame for professional interview behavior assessment"""
    global llm_response, latest_analysis

    if frame is None:
        return "No frame to analyze"

    # Convert frame to PIL image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    # Create prompt focused on professional interview analysis
    prompt = (
        "You are an expert recruitment interviewer analyzing a candidate during a job interview. "
        "Analyze this frame of an interview candidate and provide a professional assessment of:"
        "\n1. Facial expressions and emotions (confidence, nervousness, engagement, etc.)"
        "\n2. Body language and posture (professional vs. casual, attentive vs. distracted)"
        "\n3. Eye contact and focus (maintaining appropriate eye contact or looking away)"
        "\n4. Overall professional impression (how they would be perceived in a job interview)"
        "\n\nProvide a concise 2-3 sentence professional assessment that would be useful for a recruiter. "
        "Focus on both positive aspects and areas for improvement. Be specific and constructive."
    )

    # Send to Gemini with the image
    try:
        content = [prompt, pil_image]
        llm_response = model.generate_content(content)
        result_text = llm_response.text

        # Update the latest analysis variable
        latest_analysis = result_text

        # Print the analysis
        print(f"\n--- New Analysis at {time.strftime('%H:%M:%S')} ---")
        print(latest_analysis)
        print("-----------------------------------")

        return result_text
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return error_msg

class VideoProcessor(VideoProcessorBase):
    """Video processor for real-time analysis of interview behavior"""
    def __init__(self, model):
        self.last_analysis_time = time.time()
        self.analysis_count = 0
        self.current_analysis = "Waiting for first analysis..."
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process each frame and analyze interview behavior"""
        global llm_response, processing_lock, latest_analysis

        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        # Analyze a frame every 10 seconds
        if current_time - self.last_analysis_time >= ANALYSIS_INTERVAL and not processing_lock.locked():
            with processing_lock:
                self.last_analysis_time = current_time
                self.analysis_count += 1

                # Process in a separate thread to avoid blocking the video stream
                def process_frame():
                    result = analyze_interview_behavior(img.copy(), self.model)
                    self.current_analysis = result

                threading.Thread(target=process_frame).start()

        # No text or status indicators on the camera feed

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    def get_latest_analysis(self):
        """Return the latest analysis result"""
        return self.current_analysis

def create_video_processor(model):
    """Factory function to create a video processor"""
    return lambda: VideoProcessor(model)