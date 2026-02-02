# U-Net pour la Segmentation d'Images Médicales

> Détection automatique de tumeurs cérébrales par Deep Learning  
> Master 2 Big Data et Cloud Computing - ENSET Mohammedia

## À Propos

Implémentation de l'architecture **U-Net** (Ronneberger et al., 2015) pour la segmentation automatique de gliomes de bas grade dans des images IRM. Ce projet utilise PyTorch pour entraîner un modèle de segmentation sémantique pixel-parfait destiné à l'aide au diagnostic médical.

**Performances visées :** Dice Score > 0.85, IoU > 0.75

## Installation

```bash
# Cloner le repository
git clone https://github.com/Masaafkhaoula/Assignment-3-Mini-Projet-CV.git
cd unet-brain-tumor-segmentation

# Installer les dépendances
pip install -r requirements.txt
```

**Prérequis :** Python 3.8+, CUDA 11.0+ (recommandé), 8GB RAM minimum

## Dataset

**Source :** [LGG MRI Segmentation - Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

**Structure requise :**
```
data/
├── train/
│   ├── images/    # 70% du dataset
│   └── masks/
├── val/
│   ├── images/    # 15% du dataset
│   └── masks/
└── test/
    ├── images/    # 15% du dataset
    └── masks/
```

**Instructions :**
1. Télécharger le dataset depuis Kaggle
2. Extraire et organiser selon la structure ci-dessus
3. Les masques doivent suivre la convention : `image.tif` → `image_mask.tif`

## Utilisation

### Entraînement
```bash
python train.py
```

**Hyperparamètres** (modifiables dans `train.py`) :
- Époques : 30
- Batch size : 4
- Learning rate : 1e-4
- Optimiseur : Adam
- Loss : Binary Cross-Entropy with Logits

**Sorties :**
- `best_model.pth` - Meilleur modèle sauvegardé
- `training_history.png` - Courbes d'apprentissage

### Prédiction
```bash
python predict.py
```

Modifier le chemin de l'image dans le script :
```python
image_path = 'data/test/images/sample.tif'
```

### API Python
```python
from model import UNet
from predict import predict_image, visualize
import torch

# Charger le modèle
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Prédiction
original, pred_mask = predict_image('image.tif', 'best_model.pth')
visualize(original, pred_mask)
```

## Architecture

**U-Net** : Architecture encoder-decoder avec skip connections

**Encoder :** 5 blocs de convolution (64→128→256→512→1024 canaux)  
**Decoder :** 4 blocs de déconvolution avec concaténation  
**Input :** 256×256×1 (niveaux de gris)  
**Output :** 256×256×1 (masque binaire)  
**Paramètres :** ~31M

**Innovation clé :** Les skip connections préservent les détails spatiaux fins perdus lors du downsampling.

## Métriques

| Métrique | Description | Objectif |
|----------|-------------|----------|
| **BCE Loss** | Erreur de classification pixel par pixel | Minimiser |
| **Dice Score** | Similarité entre prédiction et vérité terrain | > 0.85 |
| **IoU** | Intersection over Union | > 0.75 |

## Structure du Projet

```
.
├── model.py           # Architecture U-Net
├── dataset.py         # Chargement et prétraitement
├── train.py           # Script d'entraînement
├── predict.py         # Inférence et visualisation
├── requirements.txt   # Dépendances
├── README.md
├── .gitignore
└── LICENSE
```

## Configuration Matérielle

**Minimum :** GTX 1060 6GB, 8GB RAM, ~2-3h d'entraînement  
**Recommandé :** RTX 3060 12GB, 16GB RAM, ~1h d'entraînement  
**Cloud :** Compatible Google Colab, AWS SageMaker, Kaggle Notebooks

## Dépendances

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.5.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

## Références

**Architecture :**
```
Ronneberger, O., Fischer, P., & Brox, T. (2015). 
U-Net: Convolutional Networks for Biomedical Image Segmentation. 
MICCAI 2015, 234-241.
```

**Dataset :**
```
Buda, M., Saha, A., & Mazurowski, M. A. (2019). 
Association of genomic subtypes of lower-grade gliomas with shape features. 
Computers in Biology and Medicine, 109, 218-225.
```

## Auteurs

**Réalisé par :** MASAAF Khaoula  
**Encadré par :** Pr. AMMAR Abderazzak  
**Institution :** ENSET Mohammedia - Université Hassan II de Casablanca  
**Formation :** Master 2 Big Data et Cloud Computing  
**Année :** 2025-2026

## License

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.


---

⭐ **N'oubliez pas de donner une étoile si ce projet vous a aidé !**
