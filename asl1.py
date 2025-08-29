import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import google.generativeai as genai
import os
import streamlit as st
import time
from datetime import datetime
from gtts import gTTS
import tempfile

# Setup
# If using secrets, do: GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_API_KEY = "AIzaSyCxCJoOU1A5JPDAwtmpt5nr-Q97jTqLNzg"
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel('gemini-pro')
chat = chat_model.start_chat(history=[])

# Load the trained ASL model
model_path = './model.p'
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the model is available.")
    st.stop()
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Modified Label dictionary for ASL signs (only alphabets)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 
    26: ' ',  # Keep space as it's useful for word separation
}

def generate_speech(text):
    """Generate speech from text using gTTS and return the audio file path."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def get_gemini_response(question, sentence):
    """Get response from Google Gemini AI."""
    try:
        prompt = f"""The sentence '{sentence}' was signed in ASL.
Question: {question}
Provide a clear, detailed explanation focused on ASL signing."""
        response = chat.send_message(prompt)
        text_response = response.text
        return text_response
    except Exception as e:
        return f"Error: {str(e)}"

def process_hand_landmarks(hand_landmarks):
    """Process hand landmarks into a feature vector for the model."""
    data_aux = []
    x_ = []
    y_ = []
    
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)
    
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))
    
    # Ensure the feature vector has a fixed length
    while len(data_aux) < 42:
        data_aux.append(0)
    return data_aux[:42]

def process_camera():
    """Function to capture video, detect ASL signs, and save detected words/sentences."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access the camera. Please ensure it is connected and try again.")
        return
    
    # Set higher frame dimensions for better resolution
    frame_width = 1920  # Increased width
    frame_height = 1080  # Increased height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    window_name = 'ASL Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Resize the window to desired dimensions (e.g., full HD)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    instructions = [
        "Place hand in green box",
        "Press SPACE to capture each sign (letter)",
        "Press ENTER to add the captured word to the sentence",
        "Press Q when the sentence is complete"
    ]
    
    word_buffer = ""
    sentence_buffer = ""
    last_capture_time = 0
    capture_interval = 1  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera. Exiting.")
            break

        # Add guidance box
        height, width = frame.shape[:2]
        box_size = int(min(width, height) * 0.7)
        center_x = width // 2
        center_y = height // 2
        cv2.rectangle(frame, 
                      (center_x - box_size//2, center_y - box_size//2),
                      (center_x + box_size//2, center_y + box_size//2), 
                      (0, 255, 0), 3)

        # Instructions overlay
        overlay = np.zeros((200, width, 3), dtype=np.uint8)
        frame[0:200, 0:width] = cv2.addWeighted(overlay, 0.5, frame[0:200, 0:width], 0.5, 0)
        
        for idx, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 40 + idx * 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        detected_char = '?'
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            prediction = model.predict([np.asarray(process_hand_landmarks(hand_landmarks))])
            predicted_index = int(prediction[0])
            # Only show characters A-Z (indices 0-25), show '?' for others
            detected_char = labels_dict.get(predicted_index, '?') if predicted_index <= 25 else '?'

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Detected: {detected_char}", (10, 250), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        # Display the accumulated word and sentence
        cv2.putText(frame, f"Word: {word_buffer}", (10, 300), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence_buffer}", (10, 350), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        if key == ord(' '):  # Capture sign and add to word buffer
            if current_time - last_capture_time > capture_interval:
                word_buffer += detected_char
                last_capture_time = current_time
                # Flash effect
                flash = np.ones_like(frame) * 255
                cv2.imshow(window_name, flash)
                cv2.waitKey(50)
        elif key == 13:  # Enter key to add word to sentence
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                word_buffer = ""
        elif key == ord('q'):  # Finish sentence detection
            # Add the last word if any
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                word_buffer = ""
            
            # Save sentence to text file
            with open('detected_text.txt', 'w') as f:
                f.write(sentence_buffer)
            
            # Indicate detection is complete
            st.session_state.completed_detection = True
            break  # Exit the loop to finish detection

    # Clean up
    cap.release()
    cv2.destroyAllWindows() 

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'completed_detection' not in st.session_state:
        st.session_state.completed_detection = False

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="ASL Interpreter")
    
    # Custom CSS for styling
    st.markdown("""<style>
        .stApp { background-color: #2c2f33 }
        .stTitle, .stMarkdown { color: white }
        .stButton > button {
            background-color: #7289da;
            color: white;
            font-size: 20px;
            padding: 10px 20px;
        }
        .stTabs [data-baseweb="tab-list"] > div {
            color: white;
        }
    </style>""", unsafe_allow_html=True)

    init_session_state()
    
    st.title("ASL Interpreter - Word/Sentence Detection")
    st.write("**Instructions:**")
    st.write("""
        1. **Place your hand** within the green box in the OpenCV window.
        2. **Press SPACE** to capture each sign (letter).
        3. **Press ENTER** to add the captured word to the sentence.
        4. **Press Q** to finish the sentence detection and save it to a file.
    """)

    if st.button("Start Detection"):
        # Run camera processing in a separate thread to not block the UI
        camera_thread = threading.Thread(target=process_camera, daemon=True)
        camera_thread.start()
    
    # After detection is completed, read from the file and generate audio
    if os.path.exists('detected_text.txt'):
        with open('detected_text.txt', 'r') as f:
            detected_sentence = f.read().strip()
            print(detected_sentence)
        if detected_sentence:
            st.success(f"Detected Sentence: **{detected_sentence}**")
            # Generate audio from the detected sentence
            audio_path = generate_speech(detected_sentence)
            if audio_path:
                with open(audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
    if st.button("Start a New Sentence"):
        # Reset everything
        if os.path.exists('detected_text.txt'):
            os.remove('detected_text.txt')
        st.session_state.completed_detection = False
        st.rerun()

if __name__ == "__main__":
    main()