import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertPreTrainedModel, BertModel
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import numpy as np
import warnings

# Suppress specific warning messages
warnings.filterwarnings("ignore", message=r"A parameter name that contains `beta` will be renamed.*")
warnings.filterwarnings("ignore", message=r"A parameter name that contains `gamma` will be renamed.*")

# Define voice interaction functionality
recognizer = sr.Recognizer()

def gtts_speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("temp.mp3")
    sound = AudioSegment.from_mp3("temp.mp3")
    play(sound)
    os.remove("temp.mp3")
    
def get_audio():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Could not request results; check your network connection")
            return None
    
def ask_question(question):
    print(f"Computer: {question}")
    gtts_speak(question)
    response = get_audio()
    
    while response is None:
        print("Please repeat your response.")
        gtts_speak("Please repeat your response.")
        response = get_audio()
    return response

def save_to_csv(filename, fieldnames, data):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data)

# Dataset class for BERT
class TransactionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
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

# Custom BERT Model with Dropout
class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.3)  # Dropout probability 0.3
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token output

        pooled_output = self.dropout(pooled_output)  # Apply dropout

        logits = self.classifier(pooled_output)  # Classification head

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits

# Function for training with early stopping
def train_bert_model_with_early_stopping(train_loader, val_loader, model, optimizer, device, patience=3, n_epochs=10):
    best_loss = np.inf  # Set best loss to infinity initially
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation loss calculation
        val_loss = evaluate_model(val_loader, model, device)

        print(f"Epoch {epoch + 1}/{n_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {val_loss}")

        # Check if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the model
            model.save_pretrained('./best_model')
            print("Model improved and saved.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")

        # Early stopping check
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return model

# Function for model evaluation
def evaluate_model(val_loader, model, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

# Function to train or load model
def train_or_load_model(file_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists('./best_model'):
        model = CustomBertForSequenceClassification.from_pretrained('./best_model')
        model = model.to(device)
    else:
        df = pd.read_csv(file_path)
        label_encoder = LabelEncoder()
        df['category_encoded'] = label_encoder.fit_transform(df['category'])

        MAX_LEN = 128
        BATCH_SIZE = 16

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset = TransactionDataset(train_df, tokenizer, MAX_LEN)
        val_dataset = TransactionDataset(val_df, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Load BERT model with dropout
        model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['category_encoded'].unique()))
        model = model.to(device)

        # Set up optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # Train the model with early stopping
        trained_model = train_bert_model_with_early_stopping(train_loader, val_loader, model, optimizer, device, patience=3, n_epochs=10)

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
        gtts_speak("Sorry, I did not understand your response.")
        
    return path, transaction_details or {}

# Main function that orchestrates the entire process
def main():
    # Step 1: Train or load the BERT model
    model, tokenizer, device = train_or_load_model('./data/updated_dataset.csv')

    # Step 2: Collect transaction data using voice
    path, transaction_details = collect_transaction_data()
    transaction_text = f"Amount: {transaction_details.get('amount', '')}. Payment method: {transaction_details.get('payment method', '')}. Transaction type: {transaction_details.get('transaction_type', '')}."
    
    # Step 3: Classify the transaction using the trained BERT model
    predicted_category = classify_transaction(transaction_text, model, tokenizer, device)

    # Step 4: Save the result to a CSV file
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
