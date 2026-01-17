---
name: "\U0001F9EA Experiment Report"
about: Log a new training run or experimental result
title: "[EXP] <Insert Name, e.g., m12_early_stopping>"
labels: 'Type: Experiment'
assignees: ''

---

### 🧪 Experiment Goal
*What hypothesis were we testing? (e.g., "Does Focal Loss improve Backdoor Recall?")*

### ⚙️ Configuration
* **Config File:** `configs/central/m12_early_stopping.yaml`
* **Key Hyperparameters:**
  * Learning Rate: 
  * Loss Function: 
  * Class Weights: 

### 📊 Results (W&B)
* **W&B Run Link:** [Paste Link Here]
* **Macro F1:** * **Backdoor Recall:** * **Accuracy:** ### 🖼️ Observations
* Did the loss converge?
* Did we see "mode collapse" (predicting only one class)?
* [Optional] Paste screenshot of Confusion Matrix

### ✅ Conclusion / Next Steps
* [ ] Keep this setting
* [ ] Discard
* [ ] Try increasing learning rate
