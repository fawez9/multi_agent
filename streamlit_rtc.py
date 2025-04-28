import os
import av
import cv2
import time
import threading
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

load_dotenv()

# --- Set your Gemini API key ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the model (only once)
model = genai.GenerativeModel('gemini-2.0-flash')

# Global variables
llm_response = None
last_process_time = 0
processing_lock = threading.Lock()
latest_analysis = "No analysis yet"  # Variable to store the latest analysis

# Configuration for frame collection and analysis
ANALYSIS_INTERVAL = 10.0  # Analyze a frame every 10 seconds

def analyze_interview_behavior(frame):
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
    def __init__(self):
        self.last_analysis_time = time.time()
        self.analysis_count = 0
        self.current_analysis = "Waiting for first analysis..."

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
                    result = analyze_interview_behavior(img.copy())
                    self.current_analysis = result

                threading.Thread(target=process_frame).start()

        # No text or status indicators on the camera feed

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Set page configuration to wide mode for better layout control
st.set_page_config(layout="wide")

# Add custom CSS to style the page and video component
st.markdown("""
<style>
/* Dark theme for chat interface */
.stApp {
    background-color: #111827;
    color: white;
}

/* Style the chat container */
.main .block-container {
    padding-bottom: 80px; /* Space for the input at bottom */
    max-width: 100% !important;
}

/* Style the chat messages */
[data-testid="stChatMessage"] {
    background-color: #1F2937 !important;
    border-radius: 10px;
    margin-bottom: 10px;
    padding: 10px;
}

/* Style user messages */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #374151 !important;
}

/* Style assistant messages */
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: #1F2937 !important;
}

/* Fix the chat input at the bottom */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #111827;
    padding: 20px;
    z-index: 1000;
    border-top: 1px solid #374151;
}

/* Style the chat input */
[data-testid="stChatInput"] input {
    background-color: #1F2937;
    color: white;
    border-radius: 20px;
}

/* Hide the default Streamlit footer */
footer {display: none !important;}

/* Adjust header styling */
h1, h2, h3 {
    color: white !important;
}

/* Make the video column fixed on the right */
[data-testid="stHorizontalBlock"] > div:nth-child(2) {
    position: fixed !important;
    right: 20px !important;
    top: 80px !important;
    width: 320px !important;
    z-index: 1000 !important;
    background-color: #111827 !important;
    border-radius: 10px !important;
    padding: 10px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

/* Adjust main content to make room for video on the right */
.main .block-container {
    max-width: calc(100% - 350px) !important;
    margin-left: 0 !important;
    padding-right: 0 !important;
}

/* Style the WebRTC component */
.stWebrtcStreamer {
    background-color: transparent !important;
    border: none !important;
}

/* Style the video element */
.stWebrtcStreamer video {
    width: 100% !important;
    max-height: 225px !important;
    border-radius: 8px !important;
}

/* Hide the video container div since we're using the column itself */
.video-container {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Create a two-column layout with chat on the left and video on the right
chat_col, video_col = st.columns([3, 1])

# Video component in the right column
with video_col:
    webrtc_ctx = webrtc_streamer(
        key="behavior-monitor",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    st.markdown(f"""
    <div class="video-container">
        <h3 style="color: white; margin-top: 0; text-align: center; font-size: 16px;">Interview Camera</h3>
    </div>
    """,unsafe_allow_html=True)

# Main chat content
st.title("👁️ Interview Chat")

# Initialize chat messages in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get user input from chat
if user_input := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


