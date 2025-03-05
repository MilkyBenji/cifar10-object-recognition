# **MyObjectRecognition**  

**MyObjectRecognition** is a deep learning-based object recognition system that leverages multiple neural network architectures, including **AlexNet, GoogleNet, InceptionV3, LeNet-5, and VGGNet**. The goal is to train, fine-tune, and evaluate models to achieve optimal accuracy before deploying the best-performing model into an **Android Studio application** using **TensorFlow Lite**.  

---

## **Table of Contents**  

1. [Project Overview](#project-overview)  
2. [Video Demo](#video-demo)  
3. [Motivation and Purpose](#motivation-and-purpose)  
4. [Problem Statement and Objectives](#problem-statement-and-objectives)  
5. [Installation and Usage](#installation-and-usage)  
6. [Contributing](#contributing)  
7. [License](#license)  

---

## **Project Overview**  

### **Introduction**  

**MyObjectRecognition** is a deep learning-powered object recognition system designed to test multiple CNN architectures, fine-tune hyperparameters, and deploy a lightweight, high-accuracy model for mobile applications. The workflow involves:  

- Training models such as **AlexNet, GoogleNet, InceptionV3, LeNet-5, and VGGNet**.  
- Fine-tuning and optimizing models to achieve higher accuracy.  
- Evaluating model performance to select the best one.  
- Converting the best model to **TensorFlow Lite** for mobile deployment.  
- Integrating the model into an **Android Studio application** for real-time object recognition.  

---

## **Video Demo**  

https://github.com/user-attachments/assets/3a63e194-d76f-41c0-8977-5534ca42b66f

https://github.com/user-attachments/assets/ec4e2acd-5c1e-47d2-bbbb-dbcb69dbf7ab

https://github.com/user-attachments/assets/32975747-0717-415e-bc17-d22f261f95b3

---

### **Features**  

- **Multi-Model Training & Comparison:** Evaluates multiple deep learning architectures.  
- **Fine-Tuning & Optimization:** Enhances model accuracy with hyperparameter tuning.  
- **Android Integration:** Deploys the optimized model using **TensorFlow Lite**.  
- **Real-Time Object Recognition:** Detects and classifies objects via a mobile camera.  
- **Lightweight Model Deployment:** Ensures efficient processing on mobile devices.  

---

### **Technologies Used**  

- **Python** – Core programming language.  
- **TensorFlow / TensorFlow Lite** – Deep learning framework for training and deployment.  
- **Keras** – High-level API for neural network development.  
- **CNN Architectures** – AlexNet, GoogleNet, InceptionV3, LeNet-5, VGGNet.  
- **Android Studio** – Mobile application development.  
- **OpenCV** – Preprocessing and image handling.  

---

## **Motivation and Purpose**  

Object recognition is widely used in **computer vision** applications, from self-driving cars to smart assistants. The goal of this project is to **experiment with multiple CNN architectures, optimize performance, and deploy an efficient mobile model using TensorFlow Lite**.  

This project helped me:  

- **Compare deep learning models for real-time object recognition.**  
- **Optimize models for better accuracy and faster inference.**  
- **Deploy machine learning models into mobile applications.**  

---

## **Problem Statement and Objectives**  

### **Problem Statement:**  

Deploying deep learning-based object recognition on mobile devices presents challenges such as **model size, computational efficiency, and accuracy trade-offs**. This project focuses on creating a **lightweight yet high-accuracy recognition model**.  

### **Objectives:**  

- Train and compare deep learning models for object recognition.  
- Optimize models to reduce latency while maintaining accuracy.  
- Convert the best model to **TensorFlow Lite** for mobile use.  
- Integrate the model into an **Android Studio application**.  
- Ensure smooth real-time object detection with minimal computational cost.  

---

## **Installation and Usage**  

### **🔧 Prerequisites**  

- **Python 3.x**  
- **TensorFlow & Keras**  
- **Android Studio**  
- **OpenCV (optional for preprocessing)**  

### **📌 Installation**  

1. **Clone this repository:**  
   ```bash
   git clone https://github.com/MilkyBenji/MyObjectRecognition.git
   cd MyObjectRecognition
   ```  
2. **Install dependencies:**  
   ```bash
   pip install tensorflow keras opencv-python numpy
   ```  
3. **Train and evaluate models:**  
   ```bash
   python train.py
   ```  
4. **Convert the best model to TensorFlow Lite:**  
   ```bash
   python convert_to_tflite.py
   ```  
5. **Deploy in Android Studio (Follow the guide in the `android/` folder).**  

---

## **Contributing**  

If you'd like to contribute to **MyObjectRecognition**, feel free to **fork the repository** and submit a **pull request**. Contributions are always welcome!  

### **Guidelines:**  

- **Code Style:** Follow PEP8 coding standards.  
- **Documentation:** Ensure proper documentation for any new features.  
- **Testing:** Verify that your code works correctly before submitting.  

---

## **License**  

This project is licensed under the **MIT License** – see the `LICENSE` file for details.  

---

This follows the same structure as **MyCalculatorApp**, keeping it **clear, professional, and well-organized**. Let me know if you need modifications! 🚀
