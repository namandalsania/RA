import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('./data/dataset.csv')

# Load the FinBERT tokenizer
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# Combine all relevant text columns into a single column
df['text'] = df['amount'] + " " + df['involvement'] + " " + df['payment_method']

# Convert labels to integers
label_dict = {label: idx for idx, label in enumerate(df['transaction_type'].unique())}
df['label'] = df['transaction_type'].map(label_dict)

# Split the data into training, validation, and test sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize the datasets separately
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=128)
eval_encodings = tokenizer(eval_df['text'].tolist(), truncation=True, padding=True, max_length=128)

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
eval_dataset = FinancialDataset(eval_encodings, eval_df['label'].tolist())

# Load the FinBERT model
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=len(label_dict), ignore_mismatched_sizes=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="steps",  # Adding evaluation during training
    eval_steps=50,                # Evaluate every 50 steps
    learning_rate=3e-5
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
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the trained model and tokenizer
model.save_pretrained('./results/finbert_model')
tokenizer.save_pretrained('./results/finbert_tokenizer')

# Example prediction
test_text = "I paid 500 dollars to my friend using a credit card."
inputs = tokenizer(test_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
predicted_label = list(label_dict.keys())[list(label_dict.values()).index(predictions.item())]
print(predicted_label)

# New Code for saving outputs

# Function to classify and save results
def classify_and_save_output(sentences, categories, filename_main, filename_detailed):
    classified_sentences_main = []
    classified_sentences_detailed = []

    for sentence in sentences:
        classification = classify_text(sentence, categories)
        
        # Save the main category results
        main_classification = {
            'text': sentence,
            'income': classification['income'],
            'expense': classification['expense'],
            'neither_expense_nor_income': classification['neither_expense_nor_income']
        }
        classified_sentences_main.append(main_classification)
        
        # Save detailed category results
        detailed_classification = classification.copy()
        detailed_classification['text'] = sentence
        classified_sentences_detailed.append(detailed_classification)
        
    # Save to CSV for main categories
    df_main = pd.DataFrame(classified_sentences_main)
    df_main.to_csv(filename_main, index=False)
    
    # Save to CSV for detailed categories
    df_detailed = pd.DataFrame(classified_sentences_detailed)
    df_detailed.to_csv(filename_detailed, index=False)

# Example usage
example_sentences = [
    "I deposited funds into my savings account.",
    "We sold a piece of machinery last week.",
    "A loan was repaid on time.",
    "Today, I withdrew cash from the bank.",
    "We acquired a new computer for the office.",
    "There was a dividend payment received this quarter.",
    "A mortgage payment was made for the new house.",
    "Interest income was generated from the savings account.",
    "We made a purchase of office supplies.",
    "A new copier was bought for the office.",
    "The state tax payment was processed.",
    "Today, I earned rewards from my credit card.",
    "A significant amount of stock was sold.",
    "We settled a personal expense last month.",
    "A business loan was approved and funds were secured.",
    "The rental insurance payment is due next week.",
    "We processed a customer refund yesterday.",
    "Utility bills, including electricity and water, were paid.",
    "A car was rented for the business trip.",
    "Funds were added to the company's bank account."
]

classify_and_save_output(example_sentences, categories, 'main_categories_output.csv', 'detailed_categories_output.csv')
