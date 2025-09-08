# Retinopathy Classification — README

## Project Summary

A retina-image multiclass classification pipeline to detect types of retinopathy. The notebook loads a CSV (`file`, `cat`), balances the dataset by oversampling minority classes, trains a custom CNN (3 conv-blocks), evaluates performance with a classification report and confusion matrix, and compares experiments with pretrained models.

---

## Repo / Notebook

* Main notebook: `RetinaDisease.ipynb`
* Key outputs you should save after running:

  * Trained model: `retinopathy_cnn_model.h5` (recommended)
  * Training plots: `accuracy_loss_plot.png` (optional)
  * Test predictions / evaluation CSV: `test_predictions.csv` (optional)

---

## Dataset

* CSV columns required:

  * `file` — image filename (e.g. `o_c3t940413043.jpg`)
  * `cat` — integer class label (0 / 1 / 2)
* Images folder: `path/to/images/` (place image files here)
* Note: drop any `Unnamed:*` index columns if present.

---

## Pipeline (what the notebook does)

1. **Load & inspect data**

   * reads CSV, coerces `cat` → int, checks that images listed exist.
2. **Handle imbalance (oversampling)**

   * Upsamples each class to the majority class size using `sklearn.utils.resample`.
   * Resulting dataset is balanced before splitting.
   * Example snippet:

     ```python
     from sklearn.utils import resample
     dfs = []
     max_size = df["cat"].value_counts().max()
     for class_id in df["cat"].unique():
         df_class = df[df["cat"] == class_id]
         df_class_over = resample(df_class, replace=True, n_samples=max_size, random_state=42)
         dfs.append(df_class_over)
     df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
     ```
3. **Train / validation split**

   * Stratified split: `train_test_split(df_balanced, test_size=0.2, stratify=df_balanced["cat"])`
4. **Data generator**

   * `ImageDataGenerator` with `rescale=1./255` (and optional augmentation in earlier experiments).
   * `target_size=(224, 224)`, `batch_size=32`, `class_mode='categorical'` for the final custom CNN.
5. **Model (final / best-performing)**

   * Custom CNN (final form used for best result):

     ```
     Input (224x224x3)
     Conv2D(32, 3x3) -> ReLU -> MaxPool
     Conv2D(64, 3x3) -> ReLU -> MaxPool
     Conv2D(128,3x3) -> ReLU -> MaxPool
     Flatten()
     Dense(128) -> ReLU
     Dropout(0.5)
     Dense(3) -> softmax
     ```
   * Compile: `optimizer='adam'`, `loss='categorical_crossentropy'`, `metrics=['accuracy']`.
6. **Training**

   * Example callbacks: `ReduceLROnPlateau(factor=0.3, patience=3)`, `EarlyStopping(patience=5, restore_best_weights=True)`.
   * Typical run: `epochs=10` (adjust up/down as needed).
7. **Evaluation**

   * Generate `classification_report` and `confusion_matrix` on validation set.
   * Save plots for loss/accuracy per epoch.

---

## Final reported results (comparison summary)

| Attempt / Model                           | Precision (macro) | Recall (macro) | F1-score (macro) | Accuracy | Samples |
| ----------------------------------------- | ----------------: | -------------: | ---------------: | -------: | ------: |
| Custom CNN – First Attempt (imbalanced)   |              0.27 |           0.35 |             0.30 |     0.41 |     353 |
| Custom CNN – Second Attempt (imbalanced)  |              0.28 |           0.28 |             0.28 |     0.30 |     353 |
| Pretrained EfficientNet – Attempt #1      |              0.15 |           0.33 |             0.21 |     0.46 |     353 |
| **Custom CNN – Balanced Dataset (final)** |          **0.79** |       **0.80** |         **0.80** | **0.80** |     487 |
| Pretrained EfficientNet – Attempt #2      |              0.11 |           0.33 |             0.17 |     0.33 |     487 |

**Conclusion:** the *Custom CNN trained on the oversampled (balanced) dataset* is the best performer (80% accuracy, balanced per-class metrics). Pretrained attempts tended to collapse to the dominant class unless heavily fine-tuned / oversampled.

---

## How to run (quick)

1. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
   ```
2. Edit top-of-notebook variables to point to your data:

   ```python
   CSV_PATH = "path/to/labels.csv"
   IMG_DIR = "path/to/images"
   ```
3. Run notebook cells top-to-bottom (or execute the notebook):

   ```bash
   jupyter notebook RetinaDisease.ipynb
   # or
   papermill RetinaDisease.ipynb output.ipynb
   ```
4. Save final model (if not already saved by the notebook):

   ```python
   model.save("retinopathy_cnn_model.h5")
   ```

---

## Files to include in repo

* `RetinaDisease.ipynb` — main notebook (final cleaned edition)
* `requirements.txt` — list of pip packages (create from environment)
* `README.md` — (this file)
* `models/retinopathy_cnn_model.h5` — saved model (optional)
* `reports/` — training plots, classification report text, confusion matrix image

---

## Repo tips & notes

* Make sure CSV `file` values exactly match the filenames in the images folder (case-sensitive). If filenames include relative paths, set `directory=None` in `flow_from_dataframe`.
* If using `class_mode='categorical'`, use `categorical_crossentropy`. If keeping integer labels from generator use `sparse_categorical_crossentropy`.
* If pretrained backbones are attempted again, use (a) oversampling or class weights, (b) staged training: freeze backbone → train head → unfreeze last N layers and fine-tune with small LR (1e-5).
* For medical images, consider domain-specific augmentations (brightness, CLAHE, minor rotation) but avoid transforms that break diagnostic features.

---

## Next steps (suggested)

* Try a **light fine-tuning** of the last few blocks of a pretrained model (e.g., EfficientNetB0 or ResNet50) on the balanced dataset with a small LR — this can sometimes beat the custom CNN.
* Add **stratified k-fold cross-validation** to report more robust metrics.
* Save per-image predictions and build a simple web demo (Flask) to show model inference on new retina images.
* Document any data-cleaning steps (how missing/invalid filenames were handled).
