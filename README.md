# LeukoShield AI: Offline Medical Dashboard for Leukemia Detection

LeukoShield AI is a smart, user-friendly AI application designed to analyze blood smear images and detect **Acute Lymphoblastic Leukemia (Leukemia / Blood Cancer)**. 

The primary advantage of this project is that it runs **completely offline on local machines**. This ensures total patient data privacy and security, as no medical files need to be uploaded to cloud servers or external APIs.

---

## 🛠️ How It Works (Project Workflow)

The application utilizes a **Dual-Model AI Pipeline** to process images sequentially:

1. **Step 1: Cell Isolation (Segmentation using U-Net)**
   The system scans the complex blood smear slide, identifies the white blood cells, and traces their precise boundaries.

2. **Step 2: Smart Preprocessing (OpenCV Engine)**
   Once the boundaries are detected, the system automatically crops individual cells from the image, removing unnecessary background noise and clusters.

3. **Step 3: Cancer Detection (Classification using ResNet50)**
   The cropped cell image is passed to a deep learning classifier, which instantly determines whether the cell is **Normal** or **Malignant (Cancerous)**.

4. **Step 4: Interactive Display (Streamlit Dashboard)**
   The medical operator simply drags and drops a cell image into a local web dashboard, and the system displays the diagnosis along with a confidence percentage in real-time.

---

## 📊 Model Performance & Results

The AI models were trained and validated using standard clinical datasets, delivering high-reliability scores:

* **Cell Segmentation Accuracy:** The U-Net model tracks and traces cell sizes and boundaries with a **95.28% Validation Dice Score**.
* **Diagnostic Accuracy:** The classification engine identifies leukemia cells with a **98.64% Validation Accuracy**.
* **Clinical Reliability:** During testing, the system successfully flagged **94 out of 96 malignant cases**, demonstrating a very high recall rate necessary for medical scanning tools.

---

## 💻 Technical Stack

* **Python:** Core programming language.
* **PyTorch:** Advanced machine learning framework used to train and run the ResNet50 and U-Net models.
* **OpenCV:** Computer vision library used for automated cell-cropping and image cleanup.
* **Streamlit:** Framework used to deploy the interactive, local user interface.
