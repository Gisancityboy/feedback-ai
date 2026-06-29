import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import FeedbackClassifier

X = np.load('data/X.npy')
y = np.load('data/y.npy')

X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

model = FeedbackClassifier(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

EPOCHS = 50
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            _, predicted = torch.max(output, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total * 100
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'feedback_model.pth')

    print(f'Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | 검증 정확도: {acc:.1f}% {"★ 저장" if acc == best_acc else ""}')

print(f'\n최고 정확도: {best_acc:.1f}% — 모델 저장 완료')
