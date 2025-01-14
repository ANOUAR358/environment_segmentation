DeepLabv3plus/
│
├── model/
│   ├── deeplabv3plus.py       # Implémentation de l'architecture du modèle
│   ├── metrics.py             # Calcul des métriques (accuracy, mIoU, F1 Score, etc.)
│   ├── prediction.py          # Fonctions de prédictions et visualisation
│   └── train.py               # Script d'entraînement
│
└── processing/
    └── data_processing.py     # Prétraitement des données

YOLO/
│
├── train/
│   ├── yolo-cityscape.pynb    # Entraînement et prétraitement des données
│   └── result/                # Résultats d'entraînement sur Cityscape
│
└── test/
    ├── test.py                # Script pour effectuer les tests
    └── example_image.jpg      # Image d'exemple pour la prédiction

U-Net/
│
└── image-segmentation-unet.pynb  # Implémentation et entraînement du modèle U-Net


## Contenu du Répertoire

Ce dépôt contient les implémentations de trois modèles de segmentation d'images : DeepLabv3+, YOLO, et U-Net, organisées en trois dossiers distincts. Voici une vue d'ensemble :

### 1. DeepLabv3+
Ce dossier est structuré comme suit :
- **model/** : Contient les modules liés au modèle DeepLabv3+.
  - deeplabv3plus.py : Implémentation de l'architecture du modèle DeepLabv3+ (from scratch et avec importation).
  - metrics.py : Implémentation des trois métriques principales : 
    - Accuracy
    - Mean Intersection over Union (mIoU)
    - Recall, F1 Score, et Precision.
  - prediction.py : Fonctions de prédiction et visualisation des résultats.
  - train.py : Fonction d'entraînement pour ajuster le modèle avec les données.

- processing/ : Contient les scripts pour le prétraitement des données.
  - data_processing.py : Fonctions de préparation et de prétraitement des données avant l'entraînement.

---

### 2. YOLO
Ce dossier est structuré en deux sous-dossiers :
- train/ :
  - yolo-cityscape.pynb : Script Jupyter Notebook pour l'entraînement du modèle YOLO et le prétraitement des données.
  - result/ : Dossier contenant les résultats d'entraînement sur le dataset Cityscape.

- test/ :
  - test.py : Script pour effectuer des tests sur le modèle YOLO.
  - Une image d'exemple incluse pour effectuer une prédiction et visualiser les résultats.

---

### 3. U-Net
Ce dossier contient :
- image-segmentation-unet.pynb : Script Jupyter Notebook implémentant et entraînant le modèle U-Net sur le dataset Cityscape.

---
le lien vers les weight d'entraînement :https://drive.google.com/drive/folders/1PjU8_kuVdHIwgev6JvBG9PQJExGmX6_u?usp=drive_link


