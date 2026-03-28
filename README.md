# Fashion-MNIST – Deep Learning Training Pipeline

A comprehensive neural network experimentation pipeline on the **Fashion-MNIST** dataset. Covers architecture comparison, regularization techniques, hyperparameter tuning, and saliency map visualization — all automated in a single script with outputs saved to an `outputs/` folder.

---

## 📌 Overview

This project systematically explores how different design choices affect a dense neural network trained on Fashion-MNIST. It runs through 6 major experiments, generates 12+ plots, and prints a full analysis report at the end.

---

## 🗂️ Dataset

**Fashion-MNIST** — 70,000 grayscale 28×28 images across 10 clothing categories.

| Split    | Samples |
|----------|---------|
| Training | 60,000  |
| Test     | 10,000  |

### 10 Classes

| ID | Label        | ID | Label      |
|----|--------------|----|------------|
| 0  | T-shirt/top  | 5  | Sandal     |
| 1  | Trouser      | 6  | Shirt      |
| 2  | Pullover     | 7  | Sneaker    |
| 3  | Dress        | 8  | Bag        |
| 4  | Coat         | 9  | Ankle boot |

**Preprocessing:** Normalized to `[0, 1]` and flattened from `28×28 → 784`.

---

## 🧪 Experiments

### 1️⃣ Architecture Comparison 

Trains 4 dense network variants and compares test accuracy:

| Architecture    | Hidden Layers |
|-----------------|---------------|
| Base (64→32)    | 64, 32        |
| Var1 (128→64)   | 128, 64       |
| Var2 (32→16)    | 32, 16        |
| Var3 (128→128)  | 128, 128      |

Also generates a **confusion matrix** and **10 sample predictions** for the base model.

---

### 2️⃣ Regularization Techniques 

Trains 5 model variants to study overfitting reduction:

| Model     | Technique                              |
|-----------|----------------------------------------|
| Baseline  | No regularization                      |
| Dropout   | Dropout(0.3) after each hidden layer   |
| L2        | L2 regularization (λ=0.001)            |
| BatchNorm | Batch Normalization after each layer   |
| Combined  | L2 + BatchNorm + Dropout(0.2) together |

Also includes:
- **Weight distribution histograms** for Baseline, L2, and Dropout models
- **Confusion matrix** for the best regularization model
- **Feature visualizations** — 16 learned filters from Layer 1
- **5 misclassified examples** with true vs predicted labels

---

### 3️⃣ Hyperparameter Sweep 

Systematically tunes three hyperparameters:

**Learning Rate sweep:**
`0.1 → 0.01 → 0.001 → 0.0001 → 0.00001`

**Batch Size sweep:**
`16 → 32 → 64 → 128 → 256`

**Activation Function sweep:**
`ReLU → Leaky ReLU → ELU → Tanh → Swish`

---

### 4️⃣ Saliency Maps 

Computes gradient-based saliency maps for 5 random test images, showing:
- Original input image
- Raw saliency (gradient magnitude)
- Overlay of saliency on input image

---

## 📊 Output Files

All outputs are saved to the `outputs/` directory.

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `task3_arch_curves.png`   | Accuracy & loss curves for 4 architectures       |
| `task3_confusion.png`     | Confusion matrix for the base model              |
| `task3_samples.png`       | 10 sample predictions (green=correct, red=wrong) |
| `task5_reg_curves.png`    | Training curves for all 5 regularization models  |
| `task5_weight_dists.png`  | Weight distribution histograms (3 models)        |
| `task5_best_cm.png`       | Confusion matrix for best regularization model   |
| `task5_features.png`      | 16 learned feature detectors from Layer 1        |
| `task5_errors.png`        | 5 misclassified examples                         |
| `task6_lr_curves.png`     | Training curves across learning rates            |
| `task6_bs_curves.png`     | Training curves across batch sizes               |
| `task6_act_curves.png`    | Training curves across activation functions      |
| `task6_hp_summary.png`    | Bar chart comparison of all hyperparameters      |
| `task6_saliency.png`      | Saliency maps for 5 test samples                 |

---

## 🧠 Model Architecture (Base)

```
Input   → 784 neurons  (flattened 28×28)
Hidden1 → 64  neurons  (ReLU)
Hidden2 → 32  neurons  (ReLU)
Output  → 10  neurons  (Softmax)
```

| Setting        | Value                           |
|----------------|---------------------------------|
| Optimizer      | Adam                            |
| Loss           | Sparse Categorical Crossentropy |
| Epochs         | 15                              |
| Batch Size     | 32                              |
| Val Split      | 20%                             |
| Default LR     | 0.001                           |

---

## 📋 Analysis Report

At the end of the run, the script prints a full report including:

- Test accuracy and loss for all architecture variants
- Generalization gap — flags overfitting if gap > 5%
- Test accuracy and loss for all regularization models
- Best learning rate, batch size, and activation function
- Top 3 most confused class pairs with error counts

---

## ⚙️ Requirements

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

> Python 3.8+ recommended.

---

## 🚀 How to Run

```bash
python solution.py
```

All plots will be saved automatically to the `outputs/` folder. No manual steps required.

---

## 📁 Project Structure

```
├── solution.py                   # Main script
└── outputs/
    ├── task3_arch_curves.png
    ├── task3_confusion.png
    ├── task3_samples.png
    ├── task5_reg_curves.png
    ├── task5_weight_dists.png
    ├── task5_best_cm.png
    ├── task5_features.png
    ├── task5_errors.png
    ├── task6_lr_curves.png
    ├── task6_bs_curves.png
    ├── task6_act_curves.png
    ├── task6_hp_summary.png
    └── task6_saliency.png
```

---

## 💡 Key Takeaways

- **Larger architectures** offer marginal accuracy gains but at higher compute cost
- **Combined regularization** (L2 + BatchNorm + Dropout) typically yields the best generalization
- **Learning rate** has the highest impact on training stability — `0.001` is the sweet spot for Adam
- **Shirt** and **Coat** are the most confused classes due to visual similarity
- **Saliency maps** reveal which pixels the model focuses on — useful for debugging misclassifications

---

## 🙌 Acknowledgements

- Dataset: [Fashion-MNIST by Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- Framework: [TensorFlow / Keras](https://www.tensorflow.org/)
