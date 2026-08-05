import pandas as pd
import torch
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from torch.utils.data import Dataset


# Load dataset

data = pd.read_csv(
    "dataset/employee_feedback.csv"
)


# Encode labels

encoder = LabelEncoder()

data["Sentiment"] = encoder.fit_transform(
    data["Sentiment"]
)


# Save label encoder

import joblib

joblib.dump(
    encoder,
    "models/bert_label_encoder.pkl"
)


# Split data

train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["Feedback"].tolist(),
    data["Sentiment"].tolist(),
    test_size=0.2,
    random_state=42
)


# Load BERT tokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# Dataset class

class FeedbackDataset(Dataset):

    def __init__(self, texts, labels):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )

        self.labels = labels


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):

        item = {
            key: torch.tensor(val[index])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index]
        )

        return item



train_dataset = FeedbackDataset(
    train_texts,
    train_labels
)


test_dataset = FeedbackDataset(
    test_texts,
    test_labels
)



# Load BERT model

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)



# Training settings
training_args = TrainingArguments(
    output_dir="models/bert_training",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs"
)
# Trainer

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset

)



# Train

trainer.train()



# Save model

model.save_pretrained(
    "models/bert_sentiment_model"
)


tokenizer.save_pretrained(
    "models/bert_sentiment_model"
)


print("BERT Employee Feedback Sentiment Model Created Successfully")
