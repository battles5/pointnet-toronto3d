# Classificazione Semantica di Point Cloud 3D Urbane con PointNet

**Progetto d'esame — Hands-on Python for Data Science**  
Master MD2SL (Data Science and Statistical Learning) — UniFi + IMT Lucca  
Prof. Fabio Pinelli

## Descrizione

Classificazione semantica per-punto di nuvole di punti LiDAR urbane usando **PointNet** (Qi et al., 2017) implementato in **PyTorch**, addestrato sul dataset **Toronto-3D**.

## Dataset

[Toronto-3D](https://github.com/WeikaiTan/Toronto-3D) — ~78M punti LiDAR da mobile mapping, 9 classi semantiche urbane.

Scarica i file `.ply` (L001, L002, L003, L004) e posizionali in `data/toronto3d/`.

## Struttura del progetto

```
hands-on-py/
├── data/toronto3d/        # File PLY del dataset (da scaricare)
├── src/
│   ├── __init__.py
│   ├── dataset.py         # Toronto3DDataset (v1 baseline)
│   ├── dataset_v2.py      # Toronto3DDatasetV2 (augmentation + blocchi 20m)
│   ├── model.py           # Architettura PointNet (condivisa)
│   ├── train.py           # Training v1 (CE loss, StepLR)
│   ├── train_v2.py        # Training v2 (CE pesata, CosineAnnealing, AMP)
│   └── utils.py           # Visualizzazioni e utilità (condiviso)
├── results/               # Output v1 baseline
├── results_v2/            # Output v2 migliorato
├── logs/                  # Log di SLURM
├── run_explore.py         # Script: data exploration + plot
├── run_pipeline.py        # Script v1: baseline pipeline
├── run_pipeline_v2.py     # Script v2: pipeline migliorata
├── job.slurm              # SLURM job v1
├── job_v2.slurm           # SLURM job v2
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Esecuzione

### Locale (interattiva)

```bash
source .venv/bin/activate
python run_explore.py   --data-dir data/toronto3d --results-dir results
python run_pipeline.py  --data-dir data/toronto3d --results-dir results
```

### SLURM

```bash
sbatch job.slurm       # v1 baseline
sbatch job_v2.slurm    # v2 migliorato
```

Il job esegue in sequenza exploration + pipeline e salva tutto in `results/` (o `results_v2/`) e `logs/`.

## Pipeline

1. **Data Exploration** — distribuzione classi, visualizzazioni 3D, analisi features, matrice di correlazione
2. **Data Preparation** — blocchi spaziali 10×10 m (stride 5 m), normalizzazione, PyTorch Dataset
3. **GridSearch** — learning rate, batch size, num_points (5 epoche × 12 combinazioni)
4. **Cross-Validation** — 5-fold con 30 epoche per fold, StepLR scheduler
5. **Training finale** — 30 epoche su tutto il training set (L001 + L002 + L004)
6. **Evaluation** — accuracy, mIoU, confusion matrix, IoU per classe, classification report

## Risultati

### Dataset

| Split   | Area | Punti       |
|---------|------|-------------|
| Train   | L001 | 10,695,757  |
| Train   | L002 | 16,066,282  |
| Train   | L004 | 10,536,643  |
| Test    | L003 | 41,021,528  |
| **Totale** |   | **78,320,210** |

Distribuzione classi fortemente sbilanciata: Road 53.2%, Building 24.4%, ratio max/min = 111.3×.

### GridSearch

Migliori iperparametri (su 12 combinazioni, 5 epoche ciascuna):

| Parametro       | Valore |
|-----------------|--------|
| batch_size      | 32     |
| learning_rate   | 0.001  |
| num_points      | 2048   |

### Cross-Validation (5-fold, 30 epoche)

| Fold | Best mIoU |
|------|-----------|
| 1    | 0.2941    |
| 2    | 0.3060    |
| 3    | 0.2739    |
| 4    | 0.2739    |
| 5    | 0.2566    |
| **Media** | **0.2809 ± 0.0173** |

### Test Set (L003)

| Metrica           | Valore |
|-------------------|--------|
| Overall Accuracy  | 39.4%  |
| Mean IoU          | 20.4%  |

**IoU per classe:**

| Classe         | IoU    | Precision | Recall |
|----------------|--------|-----------|--------|
| Unclassified   | 0.1516 | 0.1756    | 0.5263 |
| Road           | 0.1230 | 0.8975    | 0.1247 |
| Road Marking   | 0.0290 | 0.0292    | 0.8106 |
| Natural        | 0.4052 | 0.6604    | 0.5118 |
| Building       | 0.3576 | 0.7193    | 0.4156 |
| Utility Line   | 0.3491 | 0.3827    | 0.7989 |
| Pole           | 0.1558 | 0.1760    | 0.5765 |
| Car            | 0.1925 | 0.2132    | 0.6650 |
| Fence          | 0.0685 | 0.0929    | 0.2064 |

### Analisi dei risultati

I risultati sono coerenti con la letteratura per un **PointNet vanilla** su Toronto-3D:

- La letteratura riporta mIoU di 42–59% per **PointNet++** (che dispone di local feature aggregation con set abstraction layers). Il nostro PointNet base, senza gerarchie locali, è strutturalmente limitato nella capacità di catturare pattern spaziali a diverse scale.
- Le classi **Natural** (0.41) e **Building** (0.36) raggiungono le IoU più alte grazie a caratteristiche geometriche e cromatiche distintive.
- **Road** (0.12 IoU) ha alta precision (0.90) ma recall bassissimo (0.12): il modello è conservativo e classifica come Road solo quando è molto sicuro, perdendo gran parte dei punti stradali.
- **Road Marking** (0.03) e **Fence** (0.07) soffrono di scarsità di campioni (0.5% e 1.0% del dataset).

### Output generati

```
results/
├── best_model_final.pth          # Modello finale addestrato
├── best_model_fold{0-4}.pth      # Modelli dei 5 fold CV
├── gridsearch_results.csv         # Risultati GridSearch
├── class_distribution.png         # Distribuzione classi
├── pointcloud_3d_classes.png      # Point cloud 3D colorata per classe
├── birdseye_view.png              # Vista dall'alto
├── feature_analysis.png           # Analisi features
├── correlation_matrix.png         # Matrice di correlazione
├── confusion_matrix.png           # Matrice di confusione (test)
├── iou_per_class.png              # IoU per classe (test)
├── cv_learning_curves.png         # Curve di apprendimento CV
└── prediction_comparison.png      # Confronto predizioni vs ground truth
```

## Configurazione hardware

- **GPU**: 2× NVIDIA L40S (46 GB VRAM ciascuna)
- **RAM**: 64 GB
- **Multi-GPU**: `nn.DataParallel` su 2 GPU
- **Cluster**: SLURM, partizione `gpu`

## Riferimenti

- Qi, C.R., Su, H., Mo, K., & Guibas, L.J. (2017). *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*. CVPR.
- Tan, W., et al. (2020). *Toronto-3D: A Large-scale Mobile LiDAR Dataset for Semantic Segmentation of Urban Roadways*. CVPRW.
- Lin, T.-Y., et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.

---

## Parte 2 — Pipeline v2 (migliorata)

Versione ottimizzata del training PointNet, separata dalla baseline per confronto diretto.

### Miglioramenti implementati

| # | Miglioramento | Dettaglio | File |
|---|---------------|-----------|------|
| 1 | **Data Augmentation** | Rotazione Z (0–2π), jitter gaussiano (σ=0.01), scaling (0.9–1.1) | `src/dataset_v2.py` |
| 2 | **CrossEntropyLoss pesata** | Pesi per classe con inverse sqrt della frequenza — riequilibra classi minoritarie | `src/train_v2.py` |
| 3 | **Cosine Annealing + Early Stopping** | CosineAnnealingWarmRestarts (T₀=20, T_mult=2), early stopping patience=15 | `src/train_v2.py` |
| 4 | **Blocchi più grandi** | 20×20 m (stride 10 m) con 4096–8192 punti per maggior contesto spaziale | `src/dataset_v2.py` |

### Ottimizzazioni GPU (2× NVIDIA L40S)

- **AMP (Mixed Precision)** — FP16 forward/backward per ~2× throughput
- **AdamW** con weight decay 1e-4 — migliore regolarizzazione
- **cuDNN benchmark** — auto-tuning kernel convoluzioni
- **pin_memory** + `non_blocking` transfer — pipeline CPU→GPU ottimale
- **Batch size 32–64** — pieno utilizzo delle 2× 46 GB L40S

### Esecuzione v2

```bash
sbatch job_v2.slurm
```

Oppure interattivo:
```bash
python run_pipeline_v2.py --data-dir data/toronto3d --results-dir results_v2
```

### GridSearch v2

12 combinazioni (10 epoche ciascuna):
- `learning_rate`: [0.001, 0.0005, 0.0001]
- `batch_size`: [32, 64]
- `num_points`: [4096, 8192]

### Risultati v2

#### GridSearch v2

Migliori iperparametri (su 12 combinazioni, 10 epoche ciascuna):

| Parametro       | Valore  |
|-----------------|---------|
| batch_size      | 64      |
| learning_rate   | 0.0005  |
| num_points      | 4096    |
| best val mIoU   | 0.2188  |

#### Cross-Validation v2 (5-fold, 100 epoche + early stopping)

| Metrica | Valore |
|---------|--------|
| **Media mIoU** | **0.3399 ± 0.0204** |

#### Test Set v2 (L003)

| Metrica           | V1       | V2       | Δ       |
|-------------------|----------|----------|---------|
| Overall Accuracy  | 39.4%    | **54.1%** | +37%   |
| Mean IoU          | 20.4%    | **29.1%** | +43%   |
| CV mIoU           | 0.2809   | **0.3399** | +21%  |

**IoU per classe (v2 vs v1):**

| Classe         | IoU v1 | IoU v2 | Precision | Recall |
|----------------|--------|--------|-----------|--------|
| Unclassified   | 0.1516 | **0.1617** | 0.1875 | 0.5403 |
| Road           | 0.1230 | **0.5936** | 0.9482 | 0.6135 |
| Road Marking   | 0.0290 | **0.1363** | 0.1484 | 0.6259 |
| Natural        | 0.4052 | **0.4704** | 0.5931 | 0.6945 |
| Building       | 0.3576 | **0.3460** | 0.6882 | 0.4104 |
| Utility Line   | 0.3491 | **0.4219** | 0.4861 | 0.7617 |
| Pole           | 0.1558 | **0.1877** | 0.3370 | 0.2975 |
| Car            | 0.1925 | **0.2387** | 0.4646 | 0.3294 |
| Fence          | 0.0685 | **0.0653** | 0.1046 | 0.1481 |

#### Analisi dei miglioramenti

- **Road**: miglioramento drammatico (0.12 → 0.59), grazie ai blocchi più grandi (20m) che forniscono più contesto spaziale e alla loss pesata che riequilibra l'addestramento.
- **Utility Line** (+21%), **Road Marking** (+370%), **Car** (+24%): le augmentation e la loss pesata migliorano significativamente le classi minoritarie.
- **Building** lieve calo (0.36 → 0.35): trade-off fisiologico a favore delle classi deboli.
- **Fence** rimane la classe più difficile (0.07): troppo pochi campioni (1% del dataset) e geometria ambigua.
- Il gap rispetto a PointNet++ (42–59% mIoU in letteratura) conferma che le limitazioni architetturali di PointNet vanilla (assenza di local feature hierarchy) restano il bottleneck principale.

### Output generati (results_v2/)

```
results_v2/
├── best_model_v2_final.pth       # Modello v2 finale
├── best_model_v2_fold{0-4}.pth   # Modelli dei 5 fold CV
├── gridsearch_v2_results.csv      # Risultati GridSearch v2
├── confusion_matrix_v2.png        # Matrice di confusione (test)
├── iou_per_class_v2.png           # IoU per classe (test)
├── cv_learning_curves_v2.png      # Curve di apprendimento CV
└── prediction_comparison_v2.png   # Confronto predizioni vs GT
```
