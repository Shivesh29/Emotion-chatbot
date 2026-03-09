# ================================
# INSTALL LIBRARIES
# ================================
!pip install -q transformers datasets torch

import torch
import random
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Silence the technical warnings
warnings.filterwarnings("ignore")

# ================================
# LOAD EMOTION CLASSIFIER
# ================================
print("Loading emotion classifier...")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    truncation=True
)

# ================================
# LOAD BLENDERBOT (The "Sane" Chatbot)
# ================================
print("Loading conversational model (Blenderbot)...")
# Blenderbot is much better at staying on topic and being empathetic
chat_model_name = "facebook/blenderbot-400M-distill"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
chat_model = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name)

# ================================
# QUICK EVALUATION (100 Samples)
# ================================
print("\nRunning accuracy check...")
dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)
label_names = dataset["train"].features["label"].names

correct = 0
for i in range(100):
    sample = dataset["test"][i]
    prediction = emotion_classifier(sample["text"])[0]["label"]
    if prediction == label_names[sample["label"]]:
        correct += 1

print(f"Evaluation Complete. Accuracy: {correct/100:.2%}")

# ================================
# CHATBOT FUNCTION
# ================================
def emotion_chatbot(user_input):
    # 1. Detect the user's emotion
    result = emotion_classifier(user_input)[0]
    emotion = result["label"].upper()

    # 2. Contextualize the prompt based on emotion
    # We "nudge" Blenderbot to be empathetic by including the emotion in the prompt
    context_prompt = f"I am feeling {emotion.lower()}. {user_input}"

    try:
        # 3. Generate response using Blenderbot's specific logic
        inputs = chat_tokenizer(context_prompt, return_tensors="pt")

        output_ids = chat_model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.3
        )

        reply = chat_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Clean up some common Blenderbot artifacts
        reply = reply.replace("  ", " ").replace(" .", ".").strip()

    except Exception as e:
        reply = "I'm here for you. Tell me more about how you're feeling."

    print(f"\n[Detected Emotion: {emotion}]")
    print(f"Bot: {reply}")

# ================================
# THE CHAT LOOP
# ================================
print("\n" + "="*40)
print("EMOTION CHATBOT IS LIVE")
print("I can feel your vibes now. Type 'quit' to exit.")
print("="*40)

while True:
    user_text = input("\nYou: ")
    if user_text.lower() in ["quit", "exit", "stop"]:
        print("Bot: Goodbye! Take care of yourself.")
        break

    if not user_text.strip():
        continue

    emotion_chatbot(user_text)
