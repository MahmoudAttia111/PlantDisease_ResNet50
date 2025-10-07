# 🌿 Plant Disease Classification using ResNet50

## 🧠 Overview

This project is a **Deep Learning-based Plant Disease Classification System** built using **TensorFlow** and a pre-trained **ResNet50** model.
It can detect **plant diseases** from leaf images and classify them into **38 different categories**, helping farmers and researchers identify issues early and ensure healthier crops.

---

## 🚀 Features

✅ Built with **Transfer Learning (ResNet50)** for high accuracy
✅ **Streamlit Web App** for user-friendly interaction
✅ Detects **38 types of plant diseases** (including healthy leaves)
✅ Trained on the **PlantVillage dataset**
✅ Can be easily deployed using **Streamlit Cloud** 

---

## 🧩 Dataset

* **Dataset used:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
* The dataset contains color images of healthy and diseased plant leaves from various species.


## 🧰 Tech Stack

* **TensorFlow / Keras**
* **ResNet50 (Pre-trained on ImageNet)**
* **Python 3.10+**
* **Streamlit**
* **NumPy**
* **Pillow**

---

## ⚙️ Installation

### 1️⃣ **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/PlantDisease_ResNet50.git
cd PlantDisease_ResNet50
```

### 2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

### 3️⃣ **Download the trained model**
Upload your `.keras` model to **Google Drive**, then update the `DRIVE_URL` in the `app.py` file:

```python
DRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
```

### 4️⃣ **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 🌾 Usage

1. Launch the Streamlit app
2. Upload a leaf image (`.jpg`, `.jpeg`, or `.png`)
3. The model will display:

   * 🌱 **Predicted Disease Name**
   * ⚡ **Prediction Confidence**

---

## 📊 Model Performance

| Metric              | Value                        |
| ------------------- | ---------------------------- |
| Validation Accuracy | ~98%                         |
| Loss Function       | Categorical Cross-Entropy    |
| Optimizer           | Adam (1e-5 fine-tuned)       |
| Epochs              | 25 (5 base + 20 fine-tuning) |

---

## 🖼️ App Preview

*(You can add this once you take a screenshot of your Streamlit app UI)*

```
![App Screenshot](screenshot.png)
```

---

## 🤝 Contributing

Contributions are always welcome!
If you’d like to improve the model or add more features:

1. Fork this repository
2. Create a new branch (`feature/your-feature`)
3. Commit your changes
4. Submit a pull request 🎯

---

### 📜 License

This project is open-source and available under the **MIT License**.

---

### 👨‍💻 Author  

**Developed by:** Mahmoud Ahmed Mahmoud Attia  
📧 Email: [atya90940@gmail.com](mailto:atya90940@gmail.com)  
💼 LinkedIn: [www.linkedin.com/in/mahmoud-ahmed-attiaa](https://www.linkedin.com/in/mahmoud-ahmed-attiaa)  
🌍 GitHub: [https://github.com/MahmoudAttia111](https://github.com/MahmoudAttia111)

 
