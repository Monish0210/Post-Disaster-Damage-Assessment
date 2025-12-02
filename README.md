# 🌍 Post Disaster Damage Assessment (Flask + PyTorch)

<p align="center">
  A deep learning web application to detect changes between pre-disaster and post-disaster satellite images using a Siamese U-Net (SiamUnet) model. The project provides an intuitive interface for image upload and automatic change map generation.
</p>

<p align="center"> 
  <img alt="Flask" src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"> 
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> 
  <img alt="Hugging Face Spaces" src="https://img.shields.io/badge/Hugging%20Face%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black">
</p>


## 🧠 About The Project

This project demonstrates a Siamese U-Net (SiamUnet) model trained to identify changes in disaster-affected regions from satellite imagery.

1.  **Model Training Section:** Conducted on Kaggle using PyTorch to train SiamUnet for change detection. The trained model is saved as siamUnet.pt.
2.  **Web Application Section:** A Flask-based backend that loads the trained model, accepts image uploads, performs inference, and displays results in a simple web interface.



## ✨ Features

- **Siamese U-Net Architecture:** Detects differences between pre- and post-event satellite images.
- **Simple Upload Interface:** Upload two images and instantly view the inference result.
- **Lightweight Deployment:** Designed for easy hosting on Render or any cloud platform.
- **Reusable Model Architecture:** Plug in any compatible .pt file for experimentation.
- **Clean Folder Structure:** Separation of training and deployment environments for clarity.

## 🛠️ Tech Stack

### 🧩 Model Training
- **Platform:** Kaggle
- **Framework:** PyTorch
- **Notebook:** Jupyter (.ipynb)
- **Language:** Python

### 🌐 Web Application
- **Backend:** Flask
- **Language:** Python
- **Styling:** HTML + CSS
- **Deployment:** Render

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### 🔧 Prerequisites
- Python (v3.8+)
- pip
- Git

### ⚙️ Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Monish0210/Post-Disaster-Damage-Assessment.git
    cd Post-Disaster-Damage-Assessment/Web_Application
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Add the Trained Model:**
   Copy your trained file siamUnet.pt from the Kaggle training folder into Web_Application/.

### ▶️ Run the Application Locally

1. **Start the Flask server:**
   ```bash
   python app.py
   ```
2. **Then open your browser at:**
   ```bash
   http://127.0.0.1:5000
   ```

## 🧪 Usage
1. **Upload Images:** On the home page, upload pre-disaster and post-disaster satellite images.
2. **Run Detection:** Click “Run Detection” to process the images.
3. **View Results:** The app displays input images and the resulting change map.

## 📁 Folder Structure
   ```bash
    Post-Disaster-Damage-Assessment/
│
├── Model_Training/               # Kaggle / local training setup
│   ├── Unet-Model.ipynb
│   ├── model.py
│   ├── requirements.txt
│
└── Web_Application/                      # Flask deployment app
    ├── app.py
    ├── model.py
    ├── siamUnet.pt
    ├── requirements.txt
    ├── static/
    │   ├── css/
    │   │   └── style.css
    │   └── uploads/
    │       └── .gitkeep
    └── templates/
        ├── index.html
        └── evaluation.html
   ```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the project, please fork the repository and create a pull request, or open an issue with the "enhancement" tag.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`)
3.  Commit your Changes (`git commit -m 'Add some NewFeature'`)
4.  Push to the Branch (`git push origin feature/NewFeature`)
5.  Open a Pull Request

## 🙌 Credits & Acknowledgment

- [SiamUnet Model Inspiration](https://github.com/microsoft/building-damage-assessment-cnn-siamese) - served as a reference for the Siamese U-Net architecture.
- A heartfelt thanks to **[Priyansh Patel](https://github.com/piyu0506)** for his consistent help, brainstorming ideas, and technical feedback during the development of the SiamUnet model and web deployment.

---

<p align="center"> ⭐ If you found this project useful, consider giving it a star on GitHub! </p> 
