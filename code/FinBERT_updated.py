import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset from a CSV file
file_path = 'path/to/your/dataset.csv'  # Replace with your actual CSV file path
data = pd.read_csv(file_path)

# Ensure your dataset has the correct columns
# The dataset should have columns: 'amount', 'involvement', 'payment_method', 'transaction_type', and 'category'
print(data.head())  # Inspect the first few rows to ensure correctness

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocess categorical data using LabelEncoder
label_encoders = {}
for column in ['involvement', 'payment_method', 'category']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Preprocess numerical data (amount)
scaler = StandardScaler()
data['amount'] = scaler.fit_transform(data[['amount']])

# Custom Dataset class for DataLoader
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        transaction_type = str(self.data.iloc[index]['transaction_type'])
        
        # Tokenize transaction_type using BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            transaction_type,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Get other features (amount, involvement, payment_method)
        amount = torch.tensor([self.data.iloc[index]['amount']], dtype=torch.float)
        involvement = torch.tensor([self.data.iloc[index]['involvement']], dtype=torch.long)
        payment_method = torch.tensor([self.data.iloc[index]['payment_method']], dtype=torch.long)
        
        # Target label (category)
        category = torch.tensor(self.data.iloc[index]['category'], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'amount': amount,
            'involvement': involvement,
            'payment_method': payment_method,
            'category': category
        }

# BERT-based model with additional inputs
class BERTMultiInputModel(nn.Module):
    def __init__(self, bert_model_name, num_involvement, num_payment_method, num_classes):
        super(BERTMultiInputModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Embeddings for categorical variables
        self.involvement_embedding = nn.Embedding(num_embeddings=num_involvement, embedding_dim=8)
        self.payment_method_embedding = nn.Embedding(num_embeddings=num_payment_method, embedding_dim=8)
        
        # Fully connected layers to process numerical and categorical inputs
        self.fc1 = nn.Linear(1, 8)  # For amount
        self.fc2 = nn.Linear(768 + 8 + 8 + 8, 256)  # Combine BERT output with the other features
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, input_ids, attention_mask, amount, involvement, payment_method):
        # Get BERT output for the text (transaction_type)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]  # CLS token output
        
        # Process categorical and numerical features
        amount_output = torch.relu(self.fc1(amount))
        involvement_output = self.involvement_embedding(involvement)
        payment_method_output = self.payment_method_embedding(payment_method)
        
        # Concatenate all the features
        combined = torch.cat((bert_output, amount_output, involvement_output, payment_method_output), dim=1)
        
        # Pass through fully connected layers
        x = torch.relu(self.fc2(combined))
        x = self.fc3(x)
        
        return x

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Dataset and DataLoader
MAX_LEN = 128
BATCH_SIZE = 16

train_dataset = CustomDataset(train_data, tokenizer, MAX_LEN)
test_dataset = CustomDataset(test_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model initialization
num_classes = len(label_encoders['category'].classes_)
num_involvement = len(label_encoders['involvement'].classes_)
num_payment_method = len(label_encoders['payment_method'].classes_)

model = BERTMultiInputModel("bert-base-uncased", num_involvement, num_payment_method, num_classes)

# Training setup
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# Training the model
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        amount = batch['amount'].to(device)
        involvement = batch['involvement'].to(device)
        payment_method = batch['payment_method'].to(device)
        labels = batch['category'].to(device)
        
        outputs = model(input_ids, attention_mask, amount, involvement, payment_method)
        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

# Evaluate the model
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    correct_predictions = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            amount = batch['amount'].to(device)
            involvement = batch['involvement'].to(device)
            payment_method = batch['payment_method'].to(device)
            labels = batch['category'].to(device)
            
            outputs = model(input_ids, attention_mask, amount, involvement, payment_method)
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()
    
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

# Training loop
EPOCHS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    val_acc, val_loss = evaluate(model, test_loader, loss_fn, device)
    
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train Loss: {train_loss}, Train Accuracy: {train_acc}')
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')
