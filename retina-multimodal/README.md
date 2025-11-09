#  Retina Multimodal — Retinal Disease Classification

This project focuses on **Multimodal Retinal Disease Classification** using both **fundus images** and **patient tabular data**.  
The pipeline fuses **EfficientNet-B0 (for images)** and an **MLP (for patient data)** into a single deep learning model capable of classifying multiple retinal diseases.

---

##  Project Overview

Early diagnosis of retinal diseases such as diabetic retinopathy, glaucoma, and macular degeneration is critical.  
Traditional image-only models overlook important clinical data (like age, blood sugar, BP).  
This project integrates both modalities — images and tabular features — to improve diagnostic accuracy.

**Architecture Highlights:**
-  *Image Encoder:* EfficientNet-B0 (pretrained on ImageNet)
-  *Tabular Encoder:* Multi-Layer Perceptron (MLP)
-  *Fusion Layer:* Concatenation of visual + tabular embeddings
-  *Classifier:* Fully connected layers with softmax output
-  *Explainability:* Grad-CAM visualization for feature interpretation

---

##  Repository Structure


