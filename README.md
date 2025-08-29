# American-Sign-Language-Detection-
Sign Language to Text Converter
This project converts American Sign Language (ASL) hand gestures into text using computer vision and machine learning. It leverages MediaPipe for hand tracking, OpenCV for image processing, and a trained ML model for gesture classification. The project also includes an optional Streamlit UI for interactive usage.
🚀 Features
•	Real-time hand tracking using MediaPipe
•	Gesture classification with a trained ML model (model.p)
•	Converts ASL signs into text output
•	Supports live webcam input
•	Easy deployment via Streamlit app
📂 Project Structure
├── asl1.py                # Main script for running the application
├── sign-to-text.ipynb     # Jupyter notebook for sign-to-text pipeline
├── ml_model.ipynb         # Notebook for training ML model
├── model.p                # Pre-trained ML model
├── requirements1.txt      # Dependencies
└── README.md              # Project documentation
⚙ Installation
1.	Clone this repository
2.	git clone https://github.com/yourusername/sign-to-text.git
3.	cd sign-to-text
4.	Create virtual environment
5.	python -m venv venv
6.	source venv/bin/activate   # On Linux/Mac
7.	venv\Scripts\activate      # On Windows
8.	Install dependencies
9.	pip install -r requirements1.txt
▶ Usage
Run with Python Script
python asl1.py
Run Jupyter Notebook
Open either sign-to-text.ipynb or ml_model.ipynb in Jupyter and run cells step by step.
Run with Streamlit
streamlit run asl1.py
🛠 Tech Stack
•	Python
•	OpenCV – image processing
•	MediaPipe – hand tracking
•	Scikit-learn / PyTorch – ML model training
•	Streamlit – web app interface
•	NumPy, Pandas, SciPy – data handling
📊 Model Training
•	Training done in ml_model.ipynb
•	The trained model is saved as model.p
•	Uses scikit-learn for classification
📌 Requirements
From requirements1.txt:
•	mediapipe
•	opencv-python & opencv-contrib-python
•	scikit-learn, scipy
•	torch
•	numpy, pandas, matplotlib
•	streamlit
•	PyAudio
•	python-dotenv
•	google-generativeai (optional integration)
🎯 Future Improvements
•	Add support for more ASL gestures
•	Improve accuracy with deep learning models (CNNs, RNNs)
•	Support multi-hand gestures
•	Add text-to-speech output
