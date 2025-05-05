import os
import time
import threading
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer
# Import the streamlit_autorefresh component
from streamlit_autorefresh import st_autorefresh
from main import interview_flow
from utils.shared_state import shared_state
# Import the video processor
from video_processor import create_video_processor

load_dotenv()

# --- Set your Gemini API key ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY2"))

# Load the model (only once)
model = genai.GenerativeModel('gemini-2.0-flash')

# Global variables
interview_lock = threading.Lock()

# Set page configuration to wide mode for better layout control
st.set_page_config(layout="wide")

# Initialize session state for message tracking if not already done
if "last_message_count" not in st.session_state:
    st.session_state.last_message_count = 0

# Add flag to track if we just submitted a message
if "just_submitted_input" not in st.session_state:
    st.session_state.just_submitted_input = False
if "skip_next_refresh" not in st.session_state:
    st.session_state.skip_next_refresh = False
    
# Track the last time the UI was updated to detect changes
if "last_update_timestamp" not in st.session_state:
    st.session_state.last_update_timestamp = 0

# Add auto-refresh component with more responsive interval
count = st_autorefresh(interval=1000, key="chat_refresh")  # 1 second refresh

# Check for new messages since last refresh
current_message_count = len(shared_state.get_messages())
current_timestamp = shared_state.get_last_update_time()

# Check if we need to force a rerun based on changes
if (current_message_count > st.session_state.last_message_count or 
    current_timestamp > st.session_state.last_update_timestamp):
    
    st.session_state.last_message_count = current_message_count
    st.session_state.last_update_timestamp = current_timestamp
    
    # Only rerun if we didn't just submit input (to avoid race conditions)
    if not st.session_state.just_submitted_input:
        print(f"UI refresh triggered: {current_message_count} messages, last update at {time.strftime('%H:%M:%S', time.localtime(current_timestamp))}")
        # Force Streamlit to rerun immediately to show new messages
        st.rerun()

# Reset the just_submitted_input flag if it was set
if st.session_state.just_submitted_input:
    st.session_state.just_submitted_input = False

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
        video_processor_factory=create_video_processor(model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    # Add code to update shared state with facial analysis
    if webrtc_ctx.video_processor:
        # Set up a thread to periodically update the facial analysis in shared state
        if "facial_analysis_thread" not in st.session_state:
            def update_facial_analysis():
                while True:
                    if webrtc_ctx.video_processor:
                        analysis = webrtc_ctx.video_processor.get_latest_analysis()
                        shared_state.update_facial_analysis(analysis)
                        print(f"Updated facial analysis: {analysis[:50]}...")
                    time.sleep(2)  # Update every 2 seconds
                    
            # Start the update thread
            facial_analysis_thread = threading.Thread(target=update_facial_analysis, daemon=True)
            facial_analysis_thread.start()
            st.session_state.facial_analysis_thread = facial_analysis_thread
    
    st.markdown(f"""
    <div class="video-container">
        <h3 style="color: white; margin-top: 0; text-align: center; font-size: 16px;">Interview Camera</h3>
    </div>
    """,unsafe_allow_html=True)

# Main chat content
st.title("👁️ Interview Chat")

# Initialize session state variables
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "interview_thread" not in st.session_state:
    st.session_state.interview_thread = None

# Function to run interview flow in a separate thread
def run_interview_flow():
    try:
        with interview_lock:
            interview_flow.invoke({"messages": []}, config={"recursion_limit": 100})
    except Exception as e:
        print(f"Error in interview thread: {str(e)}")

# Start the interview thread if not already started
if not st.session_state.interview_started:
    st.session_state.interview_started = True
    # Create and start the interview thread
    st.session_state.interview_thread = threading.Thread(target=run_interview_flow)
    st.session_state.interview_thread.daemon = True  # Make thread exit when main thread exits
    st.session_state.interview_thread.start()
    print("Interview thread started")


# Handle user input
user_input = st.chat_input("Enter your response:")
if user_input:
    # Don't display user message immediately - it will be shown from the shared_state
    
    # Set the user response in shared state
    # Note: set_user_response internally calls add_message so we don't need to call it separately
    shared_state.set_user_response(user_input)
    
    # Set the just_submitted_input flag and skip the next refresh
    st.session_state.just_submitted_input = True
    st.session_state.skip_next_refresh = True
    
    # Update the message count to include this new message
    st.session_state.last_message_count = len(shared_state.get_messages())

# Display chat messages from shared state
messages = shared_state.get_messages()

# Display all messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Update the message count tracker to avoid showing duplicates on refresh
st.session_state.last_message_count = len(messages)