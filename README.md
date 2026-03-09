# Affective Conversational Agent

## 1. Project Description
This project implements an **Emotion-Aware Chatbot** designed to provide empathetic support. It uses a **RoBERTa-based classifier** to detect user intent/emotion and steers a **BlenderBot** generative model to produce contextually sensitive replies.

## 2. Problem Statement
Traditional chatbots often respond purely to keywords, ignoring the emotional weight of a conversation. This "affective blindness" makes them unhelpful in stressful situations (e.g., student burnout). Our solution uses **Emotion-Conditioned Dialogue** to bridge this gap.

## 3. Related Work
- **CARER Framework:** Used for high-accuracy emotion mapping (Saravia et al., 2018).
- **BlenderBot:** A Meta AI model trained specifically on the **EmpatheticDialogues** dataset (Roller et al., 2021).
- **DistilRoBERTa:** A lightweight transformer model used here for real-time emotional supervision.

## 4. Methodology
- **Data:** Utilizing the DAIR-AI Emotion dataset (20,000+ samples).
- **Architecture:** A pipeline where the Classifier acts as a "Supervisor" for the Generator.
- **Evaluation:** The system tracks emotional trends throughout a session to provide a final "Mood Report."
