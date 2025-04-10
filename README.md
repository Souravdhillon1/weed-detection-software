
# 🌿 AI-Based Weed Detection for Smart Farming

This project implements a deep learning solution using PyTorch and ResNet18 to identify and classify **weeds in crop fields**. Designed for integration with drone sprayers, autonomous robots, or laser weed killers, this tool supports **precision agriculture** by enabling targeted weed control.

---

## 📌 Highlights

- Trained on ~8,000 images of **cotton**, **pigweed**, and **nutgrass**
- Achieved **96.30% validation accuracy** and **95.26% test accuracy**
- Detected weeds accurately in online-sourced images (demonstrating strong generalization)
- Built using **transfer learning** (ResNet18) in **PyTorch**
- Supports **GPU acceleration**
- Easily customizable for other crops and regions

---

## 📁 Dataset

- Format: ImageFolder directory structure
- Classes: `cotton`, `pigweed`, `nutgrass`
- Source: [Kaggle - puneetsaini11/cottonweeds](https://www.kaggle.com/datasets/puneetsaini11/cottonweeds)

> To adapt for other crops, replace the dataset with labeled weed images relevant to the region or crop.

---

## 🛠 Setup & Installation

```bash
git clone https://github.com/yourusername/weed-detection-ai.git
cd weed-detection-ai
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Prepare your dataset in ImageFolder format.
2. Set the dataset path in your training script.
3. Train the model:
   ```python
   weed_model.train(train_dataloader, num_epochs=10)
   ```
4. Evaluate the model:
   ```python
   accuracy = weed_model.evaluate(test_dataloader)
   print(f"Test Accuracy: {accuracy:.2f}%")
   ```

---

## 🛰 Applications

- Smart sprayers for precise herbicide application
- Autonomous weed-killing robots
- Laser-based weed removal systems

---

## 🔁 Generalization

The system is modular and retrainable. You can generalize it for **different crops or geographies** by using local weed datasets and following the same training pipeline.

---

## 👤 Author

**Sourav Singh**  
💼 Connect on [LinkedIn](https://www.linkedin.com/in/your-link)  

---

## 📷 Demo

The trained model accurately recognized images of pigweed, nutgrass, and cotton sourced from the internet — proving its real-world applicability.

---

## 🧠 Future Work

- Weed localization with YOLO/Faster R-CNN
- Real-time deployment on edge devices
- Integration with drone flight controllers and robotic platforms

---
