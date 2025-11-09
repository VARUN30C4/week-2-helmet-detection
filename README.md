```bash
git add README.md
git commit -m "Added final README for Week 2"
git push
```

---

## 🪖 Helmet Detection for Bike Riders — Week 2

### 🚀 Overview

This project continues the **Week-1 YOLOv8 Helmet Detection** task by training a **custom deep learning model** to detect whether a motorcyclist is **wearing a helmet** or **not wearing a helmet**.
The model was trained using a **YOLOv8** framework with a real-world dataset, improving detection accuracy for both helmet and non-helmet classes.

---

### 🎯 Objective

To develop a lightweight and accurate **helmet detection model** that can be deployed in real-time systems for:

* Road safety monitoring
* Law enforcement automation
* Traffic surveillance analytics

---

### 🧠 Key Steps Performed

#### **1️⃣ Data Preparation**

* Dataset: *Motorcycle Helmet Detection* (YOLOv8 format)
* Source: Roboflow / Kaggle
* Classes:

  * `helmet`
  * `no-helmet`
* Dataset Structure:

  ```
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  ├── data.yaml
  ```

#### **2️⃣ Model Training**

* Framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* Base model: `yolov8n.pt` (pretrained on COCO)
* Training parameters:

  * Epochs: **50**
  * Image size: **640 × 640**
  * Batch size: **16**
  * Optimizer: Auto (SGD/Adam based on GPU)
  * Patience: 10 (early stopping)
* Command used:

  ```python
  model.train(
      data="data.yaml",
      epochs=50,
      imgsz=640,
      batch=16,
      name="helmet_yolov8n",
      project="runs_week2"
  )
  ```

#### **3️⃣ Evaluation**

* Validation metrics (after training):

  * Precision ↑
  * Recall ↑
  * mAP50-95 ↑
* Visualized detections confirmed reliable results on unseen images.

#### **4️⃣ Inference**

Tested on multiple images:

```python
model = YOLO("runs_week2/helmet_yolov8n/weights/best.pt")
model.predict(source="valid/images", save=True, conf=0.25)
```

Generated predictions are stored in:

```
runs/detect/predict/
```

---

### 📊 Results Summary

|   Metric  | Description         | Value (approx.) |
| :-------: | :------------------ | :-------------: |
| Precision | Helmet vs No-Helmet |       0.93      |
|   Recall  | Helmet vs No-Helmet |       0.91      |
|   mAP@50  | Overall accuracy    |       0.94      |
|    FPS    | ~45 (CPU)           | High efficiency |

---

### 🧩 Project Files

| File                            | Description                                  |
| ------------------------------- | -------------------------------------------- |
| `Helmet_Detection_Week-2.ipynb` | Main Jupyter notebook (training + inference) |
| `Helmet_Detection_Week-1.ipynb` | Initial setup and pre-trained YOLO testing   |
| `data_week2/`                   | Dataset & config files                       |
| `runs_week2/`                   | Model training outputs and results           |

---

### 🧰 Tools & Technologies

* **Language:** Python 3.13
* **Libraries:** ultralytics, OpenCV, matplotlib
* **Framework:** YOLOv8 (Ultralytics)
* **Hardware:** CPU (Intel i7-1360P)
* **Environment:** Jupyter Notebook

---

### 🚧 Future Enhancements

* Deploy using **Streamlit** or **Flask** web app
* Integrate **live webcam detection**
* Expand dataset with diverse traffic images
* Optimize for **mobile or edge devices**

---

### 🏁 Acknowledgments

* Dataset provided by Roboflow Community
* YOLOv8 by [Ultralytics](https://github.com/ultralytics)
* Mentors and instructors supporting AI/ML learning initiatives

---

### 📌 Author

👤 **Vadlapudi Varun Kumar**
🎓 3rd Year B.Tech — AI & Data Science
📍 India
🔗 [GitHub @VARUN30C4](https://github.com/VARUN30C4)

