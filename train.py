"""
Script d'entraînement du modèle U-Net
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import UNet
from dataset import get_dataloaders


def train_epoch(model, loader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque"""
    model.train()
    total_loss = 0
    
    for images, masks in tqdm(loader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Valide le modèle"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def train(num_epochs=30, batch_size=4, learning_rate=1e-4):
    """Fonction principale d'entraînement"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Modèle
    model = UNet(n_channels=1, n_classes=1).to(device)
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss et optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dataloaders
    train_loader, val_loader = get_dataloaders(batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Entraînement
    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Meilleur modèle sauvegardé!")
    
    # Graphique
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    print("\n✓ Entraînement terminé!")


if __name__ == '__main__':
    train(num_epochs=30, batch_size=4, learning_rate=1e-4)
