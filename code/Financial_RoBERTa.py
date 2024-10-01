import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../data/dataset.csv')

# Load the Financial RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('abhilash1910/financial_roberta')

# Combine all relevant text columns into a single column
df['text'] = df['amount'].astype(str) + " " + df['involvement'].astype(str) + " " + df['payment_method'].astype(str)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Convert labels to integers
label_dict = {label: idx for idx, label in enumerate(df['transaction_type'].unique())}
df['label'] = df['transaction_type'].map(label_dict)
    
# Split the data into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)  # 70% train, 30% temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # 15% val, 15% test

# Tokenize the datasets separately
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)

# Create the dataset
class FinancialDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Create the dataset objects
train_dataset = FinancialDataset(train_encodings, train_df['label'].tolist())
val_dataset = FinancialDataset(val_encodings, val_df['label'].tolist())
test_dataset = FinancialDataset(test_encodings, test_df['label'].tolist())

# Load the Financial RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('abhilash1910/financial_roberta', num_labels=len(label_dict))

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1000,  # You can adjust this depending on your needs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=3e-5,  # Explicitly setting a learning rate can help with training stability
    evaluation_strategy="steps",  # Evaluate during training
    eval_steps=50,  # Evaluate every 50 steps
    save_strategy="steps",  # Save checkpoints every eval_steps
    save_steps=50,
    save_total_limit=3,  # Save the last 3 checkpoints only
    load_best_model_at_end=True,  # Load the best model found during evaluation
    metric_for_best_model="f1",  # Metric to use for choosing the best model
    greater_is_better=True  # Whether higher values of the metric are better
)

# Define a compute_metrics function to calculate metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    logloss = log_loss(labels, p.predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "log_loss": logloss}

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, # Use validation set for evaluation during training
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test results:", test_results)

# Save the trained model and tokenizer
model.save_pretrained('./results/financial_roberta_model')
tokenizer.save_pretrained('./results/financial_roberta_tokenizer')

# # Example prediction
# test_text = "I paid 500 dollars to my friend using a credit card."
# inputs = tokenizer(test_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
# outputs = model(**inputs)
# predictions = torch.argmax(outputs.logits, dim=-1)
# predicted_label = list(label_dict.keys())[list(label_dict.values()).index(predictions.item())]
# print(predicted_label)

# Example prediction
test_text = "I paid 500 dollars to my friend using a credit card."
inputs = tokenizer(test_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
predicted_label = label_dict.get(predictions.item(), "Unknown")
print(predicted_label)

