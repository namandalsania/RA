import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings

# Suppress specific warning messages
warnings.filterwarnings("ignore", message=r"A parameter name that contains `beta` will be renamed.*")
warnings.filterwarnings("ignore", message=r"A parameter name that contains `gamma` will be renamed.*")

file_path = '../data/dataset_with_subcategories.csv'  
df = pd.read_csv(file_path)

print(df.head())

# Encode the target column 'category'
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class to handle text inputs for all columns
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
        
# Parameters
MAX_LEN = 128
BATCH_SIZE = 16

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = TransactionDataset(train_df, tokenizer, MAX_LEN)
val_dataset = TransactionDataset(val_df, tokenizer, MAX_LEN)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['category_encoded'].unique()))

# Move model to device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training function
def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)

# Validation function
def eval_model(model, data_loader, device, label_encoder):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            # Store predictions and labels for calculating metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    accuracy = correct_predictions.double() / len(data_loader.dataset)

    # Convert encoded labels back to original categories
    decoded_labels = label_encoder.inverse_transform(all_labels)
    decoded_preds = label_encoder.inverse_transform(all_preds)

    # Generate classification report
    class_report = classification_report(decoded_labels, decoded_preds, output_dict=True)
    precision, recall, f1_score, _ = precision_recall_fscore_support(decoded_labels, decoded_preds, average='weighted')

    return accuracy, sum(losses) / len(losses), precision, recall, f1_score, class_report

# Training loop with metrics
EPOCHS = 3
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f'Train loss: {train_loss}, accuracy: {train_acc}')
    
    val_acc, val_loss, precision, recall, f1_score, class_report = eval_model(model, val_loader, device, label_encoder)
    print(f'Validation loss: {val_loss}, accuracy: {val_acc}')
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')
    
    # Optionally, print the detailed classification report for each class
    print("\nDetailed Classification Report:")
    print(pd.DataFrame(class_report).transpose())

# Final model evaluation on validation data
val_acc, val_loss, precision, recall, f1_score, class_report = eval_model(model, val_loader, device, label_encoder)
print(f'Final Validation accuracy: {val_acc}, loss: {val_loss}')
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')
