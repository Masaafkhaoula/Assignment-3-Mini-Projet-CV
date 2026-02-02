"""
Script de prédiction et visualisation
"""

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from model import UNet


def predict_image(image_path, model_path='best_model.pth'):
    """Prédit le masque d'une image"""
    
    # Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Charger l'image
    image = Image.open(image_path).convert('L')
    original = np.array(image)
    
    # Préparer l'image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Binariser
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    
    return original, pred_mask


def visualize(image, pred_mask, ground_truth=None):
    """Visualise les résultats"""
    
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        axes[1].imshow(pred_mask, cmap='jet')
        axes[1].set_title('Prédiction')
        axes[1].axis('off')
        
        axes[2].imshow(ground_truth, cmap='jet')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        axes[3].imshow(image, cmap='gray')
        axes[3].imshow(pred_mask, cmap='jet', alpha=0.5)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        axes[1].imshow(pred_mask, cmap='jet')
        axes[1].set_title('Prédiction')
        axes[1].axis('off')
        
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(pred_mask, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction.png', dpi=150)
    plt.show()
    print("✓ Résultat sauvegardé: prediction.png")


if __name__ == '__main__':
    # Exemple d'utilisation
    image_path = 'data/test/images/sample.tif'
    original, pred_mask = predict_image(image_path)
    visualize(original, pred_mask)
