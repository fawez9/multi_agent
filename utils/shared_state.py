"""
Shared state module for communication between Streamlit UI and interview agent.
This module provides a thread-safe way to share state between different components.
"""

import threading
import queue
import time
import json
import uuid
from typing import Any, Optional, List, Dict

# Constants for optimization
MAX_MESSAGES = 50  # Maximum number of messages to store in UI
MAX_CONTEXT_MESSAGES = 10  # Maximum number of messages to include in context
MESSAGE_LOCK_TIMEOUT = 5  # Timeout for message lock in seconds

class SharedState:
    """Thread-safe shared state for communication between UI and agent."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SharedState, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the shared state."""
        self.user_input_queue = queue.Queue()
        self.response_ready = False
        self.current_response = None
        self.messages = []
        self.interview_complete = False
        self.last_message_timestamp = time.time()  # Track when messages were last updated

        # Add more granular locks
        self.message_lock = threading.RLock()  # Reentrant lock for message operations
        self.response_lock = threading.RLock()  # Reentrant lock for response operations

        # Add summarization state
        self.summary = ""
        self.summarized_count = 0

    def add_message(self, role: str, content: str):
        """Add a message to the messages list with pruning if needed."""
        message_id = str(uuid.uuid4())
        message = {
            "role": role, 
            "content": content, 
            "timestamp": time.time(),
            "message_id": message_id,
            "is_refinement": False
        }

        # Use a timeout to prevent deadlocks
        if not self.message_lock.acquire(timeout=MESSAGE_LOCK_TIMEOUT):
            print("Warning: Could not acquire message lock, skipping message addition")
            return None

        try:
            # Check if we need to prune messages
            if len(self.messages) >= MAX_MESSAGES:
                # Summarize older messages before pruning
                self._summarize_oldest_messages(5)  # Summarize 5 oldest messages
                # Remove oldest messages to stay under limit
                self.messages = self.messages[-(MAX_MESSAGES-1):]

            self.messages.append(message)
            self.last_message_timestamp = time.time()
            return message
        finally:
            self.message_lock.release()

    def add_refined_message(self, role: str, content: str, original_content: str = ""):
        """Add a refined version of a message, preserving conversational flow."""
        # Create a unique ID for this refined message
        message_id = str(uuid.uuid4())
        message = {
            "role": role, 
            "content": content, 
            "timestamp": time.time(),
            "message_id": message_id,
            "is_refinement": True,
            "original_content": original_content
        }

        # Use a timeout to prevent deadlocks
        if not self.message_lock.acquire(timeout=MESSAGE_LOCK_TIMEOUT):
            print("Warning: Could not acquire message lock, skipping refinement addition")
            return None

        try:
            # Always add refinements as new messages rather than updating existing ones
            # This preserves conversation flow and ensures UI updates
            self.messages.append(message)
            self.last_message_timestamp = time.time()
            print(f"Added refined message. Original: '{original_content[:30]}...' -> New: '{content[:30]}...'")
            return message
        finally:
            self.message_lock.release()

    def get_messages(self, limit: Optional[int] = None):
        """Get messages with optional limit."""
        with self.message_lock:
            if limit is None:
                return self.messages
            return self.messages[-limit:]
            
    def get_last_update_time(self):
        """Get the timestamp of the last message update."""
        with self.message_lock:
            return self.last_message_timestamp

    def get_context_messages(self):
        """Get messages formatted for model context, including summary of older messages."""
        with self.message_lock:
            result = []

            # Add summary if we have one
            if self.summary and self.summarized_count > 0:
                result.append({
                    "role": "system",
                    "content": f"Summary of {self.summarized_count} previous messages: {self.summary}"
                })

            # Add most recent messages
            recent_messages = self.messages[-MAX_CONTEXT_MESSAGES:] if len(self.messages) > MAX_CONTEXT_MESSAGES else self.messages
            result.extend(recent_messages)

            return result

    def _summarize_oldest_messages(self, count: int):
        """Summarize the oldest messages to reduce context size."""
        if count <= 0 or len(self.messages) <= count:
            return

        to_summarize = self.messages[:count]

        # Only summarize if we have enough messages
        if len(to_summarize) < 3:
            return

        # Update summary count
        self.summarized_count += len(to_summarize)

        # Create a simple summary by concatenating
        if not self.summary:
            self.summary = "Interview started. "

        for msg in to_summarize:
            role = msg["role"]
            content = msg["content"]
            # Add a condensed version to the summary
            if len(content) > 100:
                content = content[:97] + "..."
            self.summary += f"{role}: {content} "

        # Limit summary length
        if len(self.summary) > 1000:
            self.summary = self.summary[:997] + "..."

    def set_user_response(self, response: str):
        """Set the user's response with thread safety."""
        # Use a timeout to prevent deadlocks
        if not self.response_lock.acquire(timeout=MESSAGE_LOCK_TIMEOUT):
            print("Warning: Could not acquire response lock, skipping response update")
            return

        try:
            # Make sure we store the message in messages list first, before setting as response
            # This ensures the message is displayed immediately even if auto-refresh happens
            self.add_message("user", response)
            
            self.current_response = response
            self.response_ready = True
            
            # Also add to queue for backward compatibility
            try:
                # Clear the queue first to avoid any race conditions
                while not self.user_input_queue.empty():
                    try:
                        self.user_input_queue.get_nowait()
                    except queue.Empty:
                        break
                    
                # Now add the new response
                self.user_input_queue.put(response, block=False)
            except queue.Full:
                # Clear the queue if it's full
                try:
                    while True:
                        self.user_input_queue.get_nowait()
                except queue.Empty:
                    pass
                # Now try again
                self.user_input_queue.put(response, block=False)
        finally:
            self.response_lock.release()

    def get_user_response(self, timeout: int = 60) -> Optional[str]:
        """Get the user's response, waiting up to timeout seconds with improved thread safety."""
        # First quick check without locking
        if not self.response_ready:
            # Wait for a response with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Try to acquire lock with short timeout
                if self.response_lock.acquire(timeout=0.5):
                    try:
                        if self.response_ready and self.current_response is not None:
                            response = self.current_response
                            self.response_ready = False
                            self.current_response = None
                            return response
                    finally:
                        self.response_lock.release()
                # Sleep briefly to reduce CPU usage
                time.sleep(0.1)
            return None  # Timeout

        # If response is ready, acquire lock and get it
        if self.response_lock.acquire(timeout=MESSAGE_LOCK_TIMEOUT):
            try:
                if self.response_ready and self.current_response is not None:
                    response = self.current_response
                    self.response_ready = False
                    self.current_response = None
                    return response
            finally:
                self.response_lock.release()

        return None  # Could not acquire lock or response not ready

    def mark_interview_complete(self):
        """Mark the interview as complete."""
        with self._lock:
            self.interview_complete = True
            # Clear any pending responses
            with self.response_lock:
                self.response_ready = False
                self.current_response = None
            # Clear the queue
            try:
                while True:
                    self.user_input_queue.get_nowait()
            except queue.Empty:
                pass

    def update_facial_analysis(self, analysis: str):
        """Update the facial analysis data"""
        with self._lock:
            self._facial_analysis = analysis
            
    def get_facial_analysis(self) -> str:
        """Get the latest facial analysis data"""
        with self._lock:
            return getattr(self, '_facial_analysis', "No analysis yet")

    def is_interview_complete(self) -> bool:
        """Check if the interview is complete."""
        with self._lock:
            return self.interview_complete

    def clear_state(self):
        """Reset the state for a new interview."""
        with self._lock:
            with self.message_lock:
                with self.response_lock:
                    self._initialize()

# Create a global instance
shared_state = SharedState()
