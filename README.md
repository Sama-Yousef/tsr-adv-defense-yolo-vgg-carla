# 🛡️ Adversarial Defense for Traffic Sign Recognition using YOLO & VGG16 in CARLA

This project proposes a robust adversarial defense system for traffic sign recognition (TSR) in autonomous driving environments. It integrates **YOLO** for segmentation and **VGG16** for classification, with **CARLA simulator** as a testing environment. We explore the impact of various adversarial attacks and evaluate the effectiveness of adversarial training.

---

## 🚗 System Architecture

- **YOLOv8s**: Used for segmenting traffic signs from real-time driving scenes.
- **VGG16**: Used to classify the segmented traffic sign image.
- **CARLA Simulator**: Provides realistic autonomous driving environments to test the perception system.
- **GTSRB Dataset**: Used for training the classifier (VGG16).
  > 🔗 [Download GTSRB Dataset](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

- **Custom YOLO Dataset**: Used to train YOLO on traffic sign detection.  
  > 🔗 [Link to YOLO Dataset on Kaggle](https://kaggle.com/your-dataset-link)

---

## 🧪 Adversarial Attacks Used

We implemented five types of adversarial attacks against the classifier (**VGG16**):

| Attack Type | Description | Adversarial Training Applied? |
|-------------|-------------|-------------------------------|
| **FGSM**    | Fast Gradient Sign Method | ✅ |
| **BIM**     | Basic Iterative Method    | ✅ |
| **PGD**     | Projected Gradient Descent | ✅ |
| **Patch**   | Physical Adversarial Patch | ❌ |
| **UAP**     | Universal Adversarial Perturbation | ❌ |

> Note: All attacks were applied directly on the **input to VGG16 classifier**.

---

## 📉 Impact of Adversarial Attacks (Before Defense)

The following table shows how the accuracy of the original model dropped when exposed to different adversarial attacks:

| Attack | Accuracy After Attack |
|--------|------------------------|
| **FGSM** | 14.00% |
| **BIM**  | 0.00%  |
| **PGD**  | 0.00%  |

---

## 🧠 Adversarial Training Results

We applied adversarial training using adversarial samples for **FGSM**, **BIM**, and **PGD** to improve model robustness. Below are the results:

| Attack Used for Training | Training Accuracy | Test Accuracy |
|--------------------------|-------------------|---------------|
| **FGSM**                 | 95.00%            | 96.42%        |
| **BIM**                  | 82.77%            | 73.21%        |
| **PGD**                  | 90.53%            | 88.74%        |

---

## 🎥 Demo Video

▶️ [**Click here to watch the demo video**](https://drive.google.com/file/d/1kbuMbATJXeZNq74X4tabHxpFCtuqDmif/view?usp=sharing)

🎬 This video demonstrates the model's performance **before and after adversarial training**, highlighting the improvement in robustness against attacks.







---
