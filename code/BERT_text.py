import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress specific warning messages
warnings.filterwarnings("ignore", message=r"A parameter name that contains `beta` will be renamed.*")
warnings.filterwarnings("ignore", message=r"A parameter name that contains `gamma` will be renamed.*")

# Set paths
model_dir = './saved_model'
model_file_exists = os.path.exists(model_dir)

# Define voice interaction functionality
recognizer = sr.Recognizer()

# Define text interaction functionality
def ask_question(question):
    print(f"Computer: {question}")
    response = input("You: ").strip().lower()  # Use text input instead of voice input
    return response

def save_to_csv(filename, fieldnames, data):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data)
        
# Collect transaction details from user
def collect_transaction_data():
    transaction_details = {}
    
    response = ask_question("Did you buy something or pay someone?")
    if "yes" in response:
        path = 'expenses.csv'
        transaction_details['amount'] = ask_question("Okay, how much did you pay?")
        transaction_details['involvement'] = ask_question("Ok, whom did you pay?")
        transaction_details['payment_method'] = ask_question("Okay, what was the method of payment?")
        transaction_details['transaction_type'] = ask_question("Great, what was it for?")

    elif "no" in response:
        response = ask_question("Alright. Did you get paid for something?")
        if "yes" in response:
            type = 'income.csv'
            transaction_details['amount'] = ask_question("Okay, how much did you receive?")
            transaction_details['involvement'] = ask_question("Who paid you?")
            transaction_details['payment_method'] = ask_question("Great, how were you paid?")
            transaction_details['transaction_type'] = ask_question("Great. Was it a sale of a product or service or for something else?")

            if "product" in transaction_details['transaction_type']:
                path = 'income_product.csv'
            elif "service" in transaction_details['transaction_type']:
                path = 'income_service.csv'
            else:
                path = 'income_miscellaneous.csv'

        elif "no" in response:
            path = 'other_transactions.csv'
            transaction_details['amount'] = ask_question("Amounts involved?")
            transaction_details['involvement'] = ask_question("Who else was involved?")
            transaction_details['transaction_type'] = ask_question("Okay, then please describe the transaction.")
            
    else:
        print("Sorry, I did not understand your response.")
        
    return path, transaction_details or {}

# Dataset class for BERT
class TransactionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Take all columns as text inputs
        amount = str(self.dataframe.iloc[index]['amount'])
        involvement = str(self.dataframe.iloc[index]['involvement'])
        payment_method = str(self.dataframe.iloc[index]['payment_method'])
        transaction_type = str(self.dataframe.iloc[index]['transaction_type'])
        category = self.dataframe.iloc[index]['category_encoded']

        # Concatenate all columns into a single text string
        text = f"Amount: {amount}. Involvement: {involvement}. Payment Method: {payment_method}. Transaction Type: {transaction_type}"

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(category, dtype=torch.long)
        }

# Function to train the BERT model
def train_bert_model(train_loader, model, optimizer, device):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Function to train the model or load the pretrained one
def train_or_load_model(file_path):
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if a pre-trained model exists
    if os.path.exists(model_dir):
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model = model.to(device)
    else:
        df = pd.read_csv(file_path)

        # Preprocess data: Encode categories and split
        label_encoder = LabelEncoder()
        df['category_encoded'] = label_encoder.fit_transform(df['category'])

        MAX_LEN = 128
        BATCH_SIZE = 16

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset = TransactionDataset(train_df, tokenizer, MAX_LEN)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Load BERT model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['category_encoded'].unique()))
        model = model.to(device)

        # Set up optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # Train the model
        for epoch in range(3):
            print(f'Epoch {epoch + 1}')
            train_bert_model(train_loader, model, optimizer, device)

        # Save the trained model
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    return model, tokenizer, device

# Function to classify new transactions using the trained BERT model
def classify_transaction(text, model, tokenizer, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    _, prediction = torch.max(logits, dim=1)

    return prediction.item()

# Main function that orchestrates the entire process
def main():
    # Step 1: Train the BERT model
    model, tokenizer, device = train_or_load_model('./data/updated_dataset.csv')

    # Step 2: Collect transaction data using voice
    path, transaction_details = collect_transaction_data()
    transaction_text = f"Amount: {transaction_details.get('amount', '')}. Payment method: {transaction_details.get('payment_method', '')}. Transaction type: {transaction_details.get('category', '')}."

    # Step 3: Classify the transaction using the trained BERT model
    predicted_category = classify_transaction(transaction_text, model, tokenizer, device)

    # Step 4: Save the result to an Excel file
    transaction_details['category'] = predicted_category
    
    # Retain only the required fields
    filtered_transaction_details = {
        'amount': transaction_details['amount'],
        'involvement': transaction_details['involvement'],
        'payment_method': transaction_details['payment_method'],
        'transaction_type': transaction_details['transaction_type'],
        'category': transaction_details['category']
    }
    
    fieldnames = ['amount', 'involvement', 'payment_method', 'transaction_type', 'category']  # Required columns
    save_to_csv(path, fieldnames, filtered_transaction_details)

    print("Transaction data saved successfully!")

if __name__ == "__main__":
    main()
