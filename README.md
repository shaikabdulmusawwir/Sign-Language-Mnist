# Sign-Language-Mnist
This project builds and deploys a deep learning model capable of recognizing American Sign Language (ASL) alphabets (A–Z) using a Convolutional Neural Network (CNN) trained on the Sign Language MNIST dataset.  The model is trained in Python using TensorFlow/Keras, and deployed as an interactive web app using streamlit.

🚀 Features

📊 Trains a CNN model from scratch on Sign Language MNIST data

🧠 Uses convolutional layers for accurate image classification

💾 Saves trained model as .h5 file (sign_language_mnist_cnn.h5)

🌐 Deployed via Streamlit with a clean and simple UI

📷 Accepts image upload and displays predicted alphabet instantly

🧠 Model Architecture
Layer	Type	Parameters
1	Conv2D (30 filters, 5×5, ReLU)	Input: (28×28×1)
2	MaxPooling2D (2×2)	—
3	Conv2D (15 filters, 3×3, ReLU)	—
4	MaxPooling2D (2×2)	—
5	Dropout (0.2)	Regularization
6	Flatten	—
7	Dense (128, ReLU)	—
8	Dense (50, ReLU)	—
9	Dense (26, Softmax)	Output Layer (A–Z)
🧩 Dataset

The dataset used is Sign Language MNIST, available on Kaggle
.

sign_mnist_train.csv → Training set

sign_mnist_test.csv → Validation/Test set

Each image: 28×28 grayscale pixels

Labels: 0–25 representing letters A–Z (excluding J and Z due to motion)

🧰 Tech Stack
Component	Technology
Programming Language	Python 3.x
Deep Learning	TensorFlow, Keras
Data Processing	NumPy, Pandas
Visualization	Matplotlib
Deployment	Streamlit
Image Handling	OpenCV, Pillow
⚙️ Project Structure
sign_language_mnist_cnn/
│
├── app.py                     # Streamlit web app
├── train_model.py             # CNN training script
├── sign_language_mnist_cnn.h5 # Saved trained model
├── data/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── requirements.txt
└── README.md

📦 Installation and Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/sign-language-mnist-cnn.git
cd sign-language-mnist-cnn

2️⃣ Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate    # For Windows

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Streamlit App
streamlit run app.py


Then open the local URL shown in the terminal (e.g. http://localhost:8501).

📷 Usage Instructions

Upload an image of a hand sign (28×28 grayscale or RGB).

The model will process and classify the image.

The predicted alphabet will be displayed along with confidence levels.

📈 Model Performance
Metric	Train Accuracy	Validation Accuracy
Accuracy	~99%	~95%

(Values may vary depending on number of epochs and dataset split)

🧠 Example Output

Uploaded Image → Predicted: "A"

🗂️ requirements.txt
tensorflow
keras
numpy
pandas
matplotlib
opencv-python
streamlit
pillow

☁️ Deployment Options

🌐 Streamlit Cloud: Deploy directly from GitHub for free.

🤗 Hugging Face Spaces: Host using streamlit runtime easily.

🧑‍💻 Author

Name: Abdul Musawwir
GitHub: @your-username

Email: your.email@example.com

⭐ Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License — free to use, modify, and share.

Would you like me to generate the requirements.txt file automatically (with the exact versions that work well for this project)?
That’ll make your deployment smoother on Hugging Face or Streamlit Cloud.
