
# **MiniFace** 🎭
*A lightweight, customizable, and real-time face recognition system for everybody!*

---

## **🌟 What is MiniFace?**

MiniFace is a project that focuses on building **personalized face recognition models** quickly and efficiently. By using live video feeds to create a dataset, training a custom CNN, and running real-time predictions, MiniFace enables you to create smart applications, all while having fun!

Whether you're working with a **Raspberry Pi**, **Jetson Nano**, or just a standard laptop, this project is made for lightweight deployment and edge computing

---

## **✨ Features**

- **Custom Dataset Creation**: 🎥 Record video and preprocess frames into labeled datasets (`True` and `False` classes)
- **Customizable Training**: 🧑‍🔬 Train a CNN specifically for your data
- **Real-Time Inference**: 🔍 Use a webcam for face detection with dynamic bounding boxes:
  - 🟢 Green: When classified as `True`
  - 🔴 Red: When classified as `False`
- **Lightweight and Scalable**: 🚀 Perfect for low-resource devices like **Raspberry Pi** and **Jetson Nano**
- **Modular Design**: 🏗️ Flexible structure, making it easy to customize for your needs and create more then one models

---

## **📂 Project Structure**

```plaintext
MiniFace/
├── data/
│   ├── True/                 # Labeled images for the "True" class
│   ├── False/                # Labeled images for the "False" class
├── models/
│   └── my_model.h5           # Trained CNN model
├── videos/
│   └── personal_video.mp4          # Personal mp4 videos with faces
├── scripts/
│   ├── extract_true_images.py      # Extract images from the personal video or live feed
|   ├── extract_false_images.py     # Extract neutral images from the FER2013 dataset
│   ├── infer.py                    # Real-time face recognition
│   ├── data_acquisition.py         # Video-based dataset creation
├── README.md                       # This file!
└── requirements.txt                # Python dependencies
````
---

## **📖 How to Use**

### **Step 1: Clone the Repository**
Clone the repository to your local machine:
```bash
git clone https://github.com/tudord14/MiniFace.git
cd MiniFace
```

---

### **Step 2: Install dependencies**
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

---

### **Step 3: Create Your Dataset**

To train the model, you need two datasets:
1. **False Dataset**: Contains neutral face images from the FER2013 dataset.
2. **True Dataset**: Contains images of your face captured from a video or live webcam feed.

Follow these steps to generate the datasets:

---

#### **1. Generate the 🔴False Dataset (Neutral Faces from FER2013)**
Run the `extract_false_images.py` script to extract neutral face images from the FER2013 dataset:
```bash
python scripts/extract_false_images.py
```

What it does:
1. Downloads the FER2013 dataset from Kaggle (free)
2. Extracts all images labeled as "Neutral" from the dataset
3. Saves the processed images in the data/False folder

---

#### **2. Generate the 🟢True Dataset**
Run the `extract_true_images.py` script to extract face images either from a personal video or the program asks you to record one on the spot by choosing the number of seconds for which it will record:

*(IF choosing to record be sure to pick up your laptop and move it around while keeping it centered and focused on your face. It would be amazing if you could even move around a little bit so that the background and lighting changes a bit)*
```bash
python scripts/extract_true_images.py
```
**If you truly want the model to have a strong prediction ability I suggest the recording to be >30 seconds and to have a variety of backgrounds and lighting**

What it does:
1. Gives you a chance to use your own videos that you can store in the videos directory or to record a new video on the spot
2. Opens your webcam and records video for a user-defined duration.
3. Extracts and preprocesses the detected faces into 48x48 grayscale images.
4. Saves the processed images in the data/True folder.
![Figure_1](https://github.com/user-attachments/assets/b8177077-40f0-4e4f-9ec2-07c21a97efae)


---

### **Step 4: Train the model**
*(If training gives you an Accuracy < 50% try again or make the personal video longer, maybe even change light and background environment)*
```bash
python scripts/train_model.py
```

---

### **Step 5: Test the model**
```bash
python scripts/infer.py
```

---

## **🎯 Applications**

MiniFace is versatile and can be used for a variety of practical and fun applications:

- **Access Control Systems**: 🏢 Grant access only to recognized individuals for home or workplace security.
- **Home Automation**: 🏠 Trigger personalized smart home actions when your face is detected.
  - Example: Turn on the lights or unlock the door when you walk in!
- **Educational Projects**: 🎓 Learn the basics of machine learning, dataset creation, and real-time face detection.
- **IoT Devices**: 🤖 Deploy on Raspberry Pi or Jetson Nano to create edge computing applications.

---

## **⚠️ Limitations**

While MiniFace is a powerful and flexible tool, it’s important to note a few limitations:

1. **Lighting Conditions**
2. **Pose Variations**
3. **Small Dataset Size**
5. **Real-World Accuracy**

---




## **👤 Using FER2013 for Neutral Faces**

As part of creating a balanced dataset for the **`False`** class (neutral expressions), **MiniFace** utilizes the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) sourced from Kaggle.

- **Rights and Usage**: FER2013 images are **not** owned by this project. We use them solely under the permissions provided by the Kaggle repository. For more details on usage rights, please refer to the [original dataset license and README](https://www.kaggle.com/datasets/msambare/fer2013).

- **Neutral Extraction**: We focus on **neutral emotion** images to serve as the “negative” or **`False`** class in our custom recognition setup. Only the necessary subset is extracted (neutral faces), saving storage and streamlining training.

> **Disclaimer**: If you plan to redistribute or publish the FER2013 images, review the original dataset’s terms to ensure you comply with the stated license and usage conditions.

---

## **📄 License & Acknowledgements**

- **MiniFace**: This project is distributed under an open-source license; see [LICENSE](LICENSE) for details.
- **FER2013 Dataset**: All credit for the FER2013 dataset belongs to its respective contributors. Please review their [Kaggle page](https://www.kaggle.com/datasets/msambare/fer2013) for license terms and usage guidelines.
