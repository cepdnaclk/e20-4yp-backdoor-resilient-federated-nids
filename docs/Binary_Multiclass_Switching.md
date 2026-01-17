# Binary vs Multiclass Classification Switching

## Overview
The federated training system now supports seamless switching between **binary** and **multiclass** classification modes through a simple configuration parameter.

**Default Mode**: Binary classification (Normal vs Attack)

## Classification Modes

### 1. **Binary Mode** (Default)
- **Classes**: 2 classes
  - Class 0: Normal traffic
  - Class 1: Attack traffic (all attack types combined)
- **Use Case**: Simple intrusion detection (Normal vs Attack)

### 2. **Multiclass Mode**
- **Classes**: 10 classes
  - Class 0: Normal traffic
  - Classes 1-9: Attack types (Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms)
- **Use Case**: Fine-grained attack detection and classification

## Usage

### Configuration
Simply change the `classification_mode` parameter in your config file:

```yaml
# configs/federated/baseline.yaml
data:
  path: "data/unsw-nb15/processed/train_pool.pt"
  classification_mode: "binary"  # or "multiclass"
```

### Available Config Files

| Config File | Mode | Dataset |
|-------------|------|---------|
| `baseline.yaml` | Binary (2 classes) - **Default** | UNSW-NB15 |
| `baseline_multiclass.yaml` | Multiclass (10 classes) | UNSW-NB15 |
| `cic_baseline.yaml` | Binary (2 classes) | CIC-IDS-2017 |
| `cic_baseline_multiclass.yaml` | Multiclass (10 classes) | CIC-IDS-2017 |

### Running Experiments

**Binary mode (default):**
```bash
python main.py
```

**Multiclass mode:**
```bash
python main.py data.classification_mode=multiclass
# Or use the dedicated config
python main.py --config-name=baseline_multiclass
```

**Specific config file:**
```bash
python main.py --config-name=cic_baseline_multiclass
```

**Override from command line:**
```bash
python main.py data.classification_mode=multiclass
```

**Add custom tags:**
```bash
python main.py +tags=["experiment-A","defense-study"]
```

## Implementation Details

### Data Conversion
The conversion happens in `src/data/loader.py`:

```python
# Multiclass: Uses original labels (0-9)
y = data['y']  # [0, 1, 2, ..., 9]

# Binary: Converts non-zero labels to 1
y = (y > 0).long()  # [0, 0, 1, 1, ..., 1]
```

### Model Architecture
- **No model changes needed**: Same `Net` architecture works for both modes
- **Output dimension**: Automatically adjusted based on `num_classes` (2 or 10)
- **Loss function**: `CrossEntropyLoss` works for both binary and multiclass

### Evaluation Metrics
- **Binary mode**: Uses `f1_score(average='binary')`
- **Multiclass mode**: Uses `f1_score(average='macro')`

## Backdoor Attack Considerations

### Binary Mode Attacks
When using binary classification, adjust your attack targets:

```yaml
attack:
  type: "backdoor"
  target_label: 0  # Force malicious traffic to be classified as Normal
  trigger_feat_idx: 0
  trigger_value: 5.0
```

### Label Flip Attack (Binary)
```yaml
attack:
  type: "label_flip"
  source_label: 1  # Attack traffic
  flip_to_label: 0  # Flip to Normal
```

### Attack Success Rate (ASR)
- **Binary ASR**: Measures how often triggered samples are misclassified as Normal (class 0)
- **Multiclass ASR**: Same logic, but can target specific attack classes

## Testing

Run the validation script to verify both modes work:

```bash
python test_classification_modes.py
```

Expected output:
```
[TEST 1] Multiclass Mode
✓ Number of Classes: 10
✓ Label range: [0, 9]

[TEST 2] Binary Mode
✓ Number of Classes: 2
✓ Unique labels: [0, 1]
✓ Binary conversion successful!
```

## W&B Organization

### Automatic Grouping and Tagging
Runs are automatically organized in Weights & Biases for easy filtering and comparison:

**Groups:**
- Binary runs: `{group_name}_binary` (e.g., `default_binary`)
- Multiclass runs: `{group_name}_multiclass` (e.g., `default_multiclass`)

**Automatic Tags (generated from config):**
- Classification mode: `binary` or `multiclass`
- Partition method: `iid`, `dirichlet`, etc.
- Defense strategy: `avg`, `krum`, `median`, `trimmed_mean`
- Attack type: `clean`, `backdoor`, `label_flip`
- Dataset: `unsw-nb15` or `cic-ids2017`
- Attack intensity: `malicious-{count}`, `model-replacement` (if applicable)

**Examples:**
```bash
# Clean binary experiment
python main.py
# → Group: default_binary
# → Tags: binary, iid, avg, clean, unsw-nb15

# Backdoor attack with Krum defense
python main.py attack.type=backdoor server.defense=krum
# → Group: default_binary
# → Tags: binary, iid, krum, backdoor, unsw-nb15, malicious-4, model-replacement

# Custom group + multiclass
python main.py +group=defense-study data.classification_mode=multiclass
# → Group: defense-study_multiclass
# → Tags: multiclass, iid, avg, clean, unsw-nb15

# Add custom tags
python main.py +tags=["experiment-A","baseline-comparison"]
# → Additional tags: experiment-A, baseline-comparison
```

**W&B Dashboard Filtering:**
- Filter by **Group** to separate binary/multiclass experiments
- Filter by **Tags** to find specific configurations (e.g., all `backdoor` runs, all `krum` defenses)
- Compare runs within the same group for fair evaluation

## Technical Notes

### Advantages
✅ **Config-driven**: Switch modes without code changes  
✅ **Same architecture**: Fair comparison between binary/multiclass  
✅ **Preserves features**: Class weights, defenses, attacks all work  
✅ **Reversible**: Switch back and forth as needed  
✅ **Research-friendly**: Standard practice in ML research  

### Files Modified
1. `configs/federated/baseline.yaml` - Added `classification_mode` parameter and tags
2. `src/data/loader.py` - Added mode conversion logic
3. `main.py` - Pass mode through pipeline, auto-generate W&B groups and tags
4. `src/server/server.py` - Adjust F1-score calculation based on num_classes

### Backward Compatibility
- Default mode is now `binary` (changed from multiclass)
- To use multiclass, explicitly set: `data.classification_mode: "multiclass"`
- No data reprocessing required
- Tags are auto-generated from config - no manual tag management needed

## Research Use Cases

### Binary vs Multiclass Comparison
```bash
# Run binary experiment (default)
python main.py

# Run multiclass experiment
python main.py data.classification_mode=multiclass
```

### Backdoor Robustness Study
Compare how backdoor attacks perform in binary vs multiclass settings:
- Binary: 2-class decision boundary (easier to fool?)
- Multiclass: 10-class decision boundary (harder to fool?)

### Defense Evaluation
Test if defenses (Krum, Trimmed Mean) work differently:
- Binary: Simple Normal/Attack distinction
- Multiclass: Complex attack-type classification

## Troubleshooting

**Issue**: Model not converging in binary mode  
**Solution**: Try adjusting learning rate or class weights

**Issue**: ASR not calculated correctly  
**Solution**: Verify `target_label` matches your mode (0-1 for binary, 0-9 for multiclass)

**Issue**: F1-score is 0  
**Solution**: Check if labels are correctly converted (run `test_classification_modes.py`)

## Example Results

```
--- Round 15/15 ---
📊 Round 15 | Accuracy: 98.45% | F1-score: 0.97 | 😈 Backdoor ASR: 0.00%
📊 Global Accuracy: 98.45%

✅ Experiment Complete!
```

## Citation
If you use this feature in your research, please cite your thesis/paper appropriately.
