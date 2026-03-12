import os
import pickle
import torch
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


MODEL_FILE = "coach_tk_router_model.pkl"
TOKENIZER_FILE = "coach_tk_router_tokenizer.pkl"


label_map = {
    0: "PERSONAL_COACHING → respond in TK coaching style",
    1: "KNOWLEDGE_QUERY → use RAG retrieval from TK content"
}


def train_and_save():

    context_texts = [
        "I feel stuck in my career growth",
        "I am confused about my next step in leadership",
        "I am losing confidence in my abilities",
        "How do I stay motivated during difficult times",
        "I feel overwhelmed with responsibilities",
        "I don’t know how to make better decisions",
        "I feel stressed about my business future",
        "How can I improve my mindset for success",
        "I am afraid of failure in my career",
        "I feel I am not progressing fast enough",
        "How do I become a better leader",
        "I struggle with self discipline",
        "I feel burnout from continuous work",
        "How do I manage my emotions at work",
        "I feel uncertain about my life direction",
        "How can I build strong daily habits",
        "I feel distracted and unfocused",
        "How do I gain clarity in decision making",
        "I feel pressure from expectations",
        "How can I stay consistent in growth",
        "I don’t feel confident leading a team",
        "How do I overcome self doubt",
        "I feel lost in my professional journey",
        "How do I improve my thinking process",
        "I feel mentally exhausted",
        "How can I develop a success mindset",
        "I feel fear while taking big decisions",
        "How do I stay calm under pressure",
        "I feel confused between multiple opportunities",
        "How do I grow personally and professionally",
        "I feel my productivity is very low",
        "How do I manage time effectively",
        "I feel disconnected from my purpose",
        "How can I improve my discipline",
        "I feel uncertain about starting a business",
        "How do I build leadership presence",
        "I feel anxious about my future",
        "How can I improve focus and clarity",
        "I feel I am not achieving my potential",
        "How do I create a clear vision for life",
        "I feel emotionally drained from work",
        "How can I think more strategically",
        "I feel stuck in comfort zone",
        "How do I push myself to next level",
        "I feel confused about my priorities",
        "How can I improve decision confidence",
        "I feel overwhelmed with too many goals",
        "How do I stay committed to long term success",
        "I feel I am failing despite hard work",
        "How can I transform my mindset for growth"
    ]

    factual_texts = [
        "What is FSNA framework",
        "Explain LYDITJ methodology",
        "What is IT Blueprint strategy",
        "Explain TK decision making model",
        "What are core leadership principles taught by TK",
        "Explain mindset framework used in TK courses",
        "What is strategic thinking in leadership",
        "How does enterprise IT strategy work",
        "Explain business execution framework",
        "What is technology transformation roadmap",
        "Explain leadership maturity model",
        "What is digital transformation strategy",
        "Explain high performance team framework",
        "What is systems thinking in business",
        "Explain coaching methodology used by TK",
        "What is operational excellence framework",
        "Explain long term vision planning",
        "What is risk management in leadership",
        "Explain business scaling strategy",
        "What is organizational alignment model",
        "Explain productivity framework in TK training",
        "What is innovation strategy in enterprises",
        "Explain decision clarity framework",
        "What is value creation in business strategy",
        "Explain change management process",
        "What is execution discipline in leadership",
        "Explain performance measurement framework",
        "What is customer centric strategy",
        "Explain IT governance model",
        "What is enterprise architecture",
        "Explain agile transformation in organizations",
        "What is leadership communication model",
        "Explain problem solving framework in TK coaching",
        "What is business growth roadmap",
        "Explain talent development strategy",
        "What is strategic planning cycle",
        "Explain mindset shift principles in leadership",
        "What is competitive advantage strategy",
        "Explain KPI framework in business",
        "What is continuous improvement model",
        "Explain data driven decision making",
        "What is coaching conversation structure",
        "Explain long term wealth strategy concepts",
        "What is leadership accountability model",
        "Explain business vision creation process",
        "What is scalable technology architecture",
        "Explain cross functional collaboration model",
        "What is execution roadmap in IT programs",
        "Explain enterprise leadership transformation",
        "What is strategic priority setting"
    ]

    texts = context_texts + factual_texts
    labels = [0] * 50 + [1] * 50

    # Train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    # Load DistilBERT
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenization
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length")

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Training config
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        logging_steps=5,
        save_strategy="no",
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    # Save model & tokenizer
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)

    print("Coach-TK router model trained & saved.")

# def load_model():
#     with open(MODEL_FILE, "rb") as f:
#         model = pickle.load(f)

#     with open(TOKENIZER_FILE, "rb") as f:
#         tokenizer = pickle.load(f)

#     return model, tokenizer


def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][pred].item()

    return label_map[pred], round(confidence, 3)


if not os.path.exists(MODEL_FILE):
    print("Training Coach-TK router first time...")
    train_and_save()

print("Loading saved Coach-TK router...")
model, tokenizer = load_model()

test_text = "accoridng my problem pls guide me "

label, conf = predict(test_text, model, tokenizer)

print("\nUser Text:", test_text)
print("Prediction:", label)
print("Confidence:", conf)
