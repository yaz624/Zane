import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "test_off_DBS.xlsx"  # Change this to the correct path
df = pd.read_excel(file_path)

# Handle missing values
df['On_medication'] = df['STS_additional_features'].fillna('None')
df['DBS_state'] = df['DBS_state'].replace('-', 'Unknown')  # Replace '-' with 'Unknown' for encoding

# Define categorical features
categorical_features = ['On_or_Off_medication', 'DBS_state', 'Clinical_assessment', 'STS_additional_features']

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and target
features = ['sts_whole_episode_duration', 'sts_final_attempt_duration', 
            'On_or_Off_medication', 'DBS_state', 'Clinical_assessment', 'STS_additional_features']
target = 'MDS-UPDRS_score_3.9 _arising_from_chair'

X = df[features]
y = df[target]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class EnhancedTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, hidden_dim))  # 可学习位置编码
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [batch, 1, hidden]
        x = x + self.pos_encoder           # 加入位置信息
        x = self.transformer(x).squeeze(1)
        return self.fc(x)

# Define Transformer-based classification model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, hidden_dim=128, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence length of 1
        x = self.transformer_encoder(x).squeeze(1)
        x = self.fc(x)
        return x

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
num_classes = len(y.unique())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Transformer model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {correct/total:.4f}")

# Evaluate model on test set
model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

# Compute Transformer model accuracy
transformer_accuracy = accuracy_score(all_labels, all_predictions)
print(f"Transformer Model Accuracy: {transformer_accuracy * 100:.2f}%")
