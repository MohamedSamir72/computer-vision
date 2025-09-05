# 📱 Phone Use Detection in Workplace  

This project detects people who use mobile phones during work. It combines **Grounding DINO-tiny** for automatic annotation and **RT-DETR** for object detection, providing a full workflow from dataset preparation to deployment.  

---

## 🚀 Features  
- ✅ Automatic annotation with **Grounding DINO-tiny** (zero-shot labeling).  
- ✅ Training and inference with **RT-DETR**.  
- ✅ Real-time detection from **video files** or **live camera feeds**.  
- ✅ Model optimization: converted trained model to **ONNX (`model.onnx`)** for faster and portable inference. 

---

## 🛠️ Tech Stack  
- **Grounding DINO-tiny** → for automatic annotations.  
- **RT-DETR (Ultralytics)** → for training & detection.  
- **Python 3.10+**, OpenCV, PyTorch, Ultralytics.  

---

## 📸 Example Output  

Here’s an example of detection (`Person+Phone` highlighted in red):  

![Example Detection](images/example_detection.jpg)  

---

## ▶️ How to Run  

### 1️⃣ Clone Repository & Install Requirements
```bash
git clone https://github.com/your-repo/phone-detection.git
cd phone-detection
pip install -r requirements.txt
