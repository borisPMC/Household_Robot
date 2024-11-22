import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 1. Dataset Class
class VideoDataset(Dataset):
    def __init__(self, data, labels, seq_len=50):
        """
        data: List of [frames x 51 (17*3 keypoints)] tensors
        labels: Corresponding action labels for the videos
        seq_len: Maximum sequence length (for padding)
        """
        self.data = [torch.tensor(self.pad_sequence(video, seq_len), dtype=torch.float32) for video in data]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len

    def pad_sequence(self, sequence, seq_len):
        """Pad or truncate sequence to fixed length."""
        if len(sequence) > seq_len:
            return sequence[:seq_len]
        else:
            padding = np.zeros((seq_len - len(sequence), sequence.shape[1]))
            return np.vstack([sequence, padding])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2. Bi-LSTM Model
class BiLSTMActionClassifier(nn.Module):
    def __init__(self, input_size=51, hidden_size=128, num_classes=5, num_layers=2):
        """
        input_size: Number of features per frame (17 keypoints * 3)
        hidden_size: Number of hidden units in LSTM
        num_classes: Number of action classes
        num_layers: Number of LSTM layers
        """
        super(BiLSTMActionClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirection

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size*2)
        # Take the last time-step output
        out = lstm_out[:, -1, :]  # (batch_size, hidden_size*2)
        out = self.fc(out)  # (batch_size, num_classes)
        return out

# 3. Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 4. Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# 5. Example Usage
if __name__ == "__main__":
    # Example data (3 videos, 17 keypoints * 3, random labels)
    num_videos = 3
    seq_len = 50
    input_size = 51
    num_classes = 5
    data = [np.random.rand(np.random.randint(30, 60), input_size) for _ in range(num_videos)]
    labels = [np.random.randint(0, num_classes) for _ in range(num_videos)]

    dataset = VideoDataset(data, labels, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = BiLSTMActionClassifier(input_size=input_size, hidden_size=128, num_classes=num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=10)
    evaluate_model(model, dataloader)
