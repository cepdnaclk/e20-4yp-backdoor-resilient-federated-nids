# CIC-UNSW-NB15 Integration Plan

**Date:** January 11, 2026  
**Goal:** Integrate CIC-UNSW-NB15 augmented dataset and compare with original UNSW-NB15

---

## 1. Dataset Overview

### CIC-UNSW-NB15 Characteristics
- **Source:** UNSW-NB15 raw packets + CICFlowMeter feature extraction
- **Features:** ~83 (CICFlowMeter standard)
- **Distribution:** 80% Benign, 20% Malicious (realistic ratio)
- **Total Samples:** 448,915 (89,783 train + 89,783 test with 80-20 split)
- **Reference:** Mohammadian et al., PST 2024

### Key Improvements Over Original
1. ✅ **More Realistic Distribution**: 80-20 benign-malicious (vs 36-64 in original)
2. ✅ **Better Minority Representation**: Worms 246 vs 35 (7x more samples)
3. ✅ **Standardized Features**: CICFlowMeter widely used in research
4. ✅ **Recent Augmentation**: 2024 vs 2015 dataset

---

## 2. Class Distribution Comparison

| Class | Original UNSW-NB15 | CIC-UNSW-NB15 | Change |
|-------|-------------------|---------------|--------|
| **Benign/Normal** | 74,400 (36.1%) | 358,332 (80%) | +4.8x ⬆️ |
| **Analysis** | 1,654 (0.8%) | 385 (0.09%) | -4.3x ⬇️ |
| **Backdoor** | 1,408 (0.7%) | 452 (0.10%) | -3.1x ⬇️ |
| **DoS** | 10,491 (5.1%) | 4,467 (1.0%) | -2.3x ⬇️ |
| **Exploits** | 26,805 (13.0%) | 30,951 (6.9%) | +1.2x ⬆️ |
| **Fuzzers** | 15,368 (7.5%) | 29,613 (6.6%) | +1.9x ⬆️ |
| **Generic** | 35,582 (17.3%) | 4,632 (1.0%) | -7.7x ⬇️ |
| **Reconnaissance** | 8,540 (4.1%) | 16,735 (3.7%) | +2.0x ⬆️ |
| **Shellcode** | 978 (0.5%) | 2,102 (0.47%) | +2.1x ⬆️ |
| **Worms** | 35 (0.02%) | 246 (0.05%) | +7.0x ⬆️ |
| **Total** | 206,138 | 448,915 | +2.2x |

### Imbalance Analysis
- **Original**: Severe imbalance (2,125:1 ratio Normal:Worms)
- **CIC-UNSW**: Moderate imbalance (1,457:1 ratio Benign:Worms) - **33% better!**
- **Most Improved**: Worms (+7x), Shellcode (+2.1x), Reconnaissance (+2x)
- **Most Reduced**: Generic (-7.7x), Analysis (-4.3x), Backdoor (-3.1x)

---

## 3. Feature Comparison

### Original UNSW-NB15 (47 features via Argus + Bro-IDS)
Categories: Basic, Content, Time, Generated
- Flow-level features
- Packet statistics
- Protocol-specific features

### CIC-UNSW-NB15 (~83 features via CICFlowMeter)
Standard CICFlowMeter features:
- **Flow Duration**
- **Packet Statistics**: Total/Min/Max/Mean/Std for Fwd/Bwd
- **Byte Statistics**: Total/Min/Max/Mean/Std for Fwd/Bwd
- **Inter-Arrival Time (IAT)**: Total/Mean/Std/Min/Max for Fwd/Bwd/Flow
- **Flags**: PSH/URG/FIN/SYN/RST/ACK counts
- **Header Lengths**: Fwd/Bwd
- **Packets/Second**: Fwd/Bwd/Flow
- **Flow Bytes/Packets**: Various ratios
- **Bulk Statistics**: Bulk rate, packet counts
- **Subflow Statistics**: Fwd/Bwd packets/bytes
- **Active/Idle**: Mean/Std/Min/Max times
- **Initial Window Size**: Fwd/Bwd
- **Active Flags**: Per direction

**Advantage**: CICFlowMeter features are standardized and well-documented, making results more comparable with other NIDS research.

---

## 4. Expected Performance Impact

### Hypothesis
| Aspect | Original UNSW-NB15 | CIC-UNSW-NB15 Prediction |
|--------|-------------------|--------------------------|
| **Macro-F1** | 0.581 | 0.60-0.65 (better balance) |
| **Accuracy** | 79.3% | 85-90% (more benign samples) |
| **Worms Recall** | 0.71 | 0.75-0.85 (7x more samples) |
| **Backdoor Recall** | 0.49 | 0.40-0.50 (fewer samples) |
| **Analysis Recall** | 0.50 | 0.35-0.45 (fewer samples) |
| **Training Time** | ~3 min | ~6-8 min (2.2x more samples) |

### Weighting Strategy Adjustment
- **Original**: Sqrt weighting essential (severe imbalance)
- **CIC-UNSW**: May benefit from **lighter weighting** or even **no weighting**
  - Test: No weights, sqrt weights, inverse weights
  - Expected: No weights or sqrt may perform similarly

---

## 5. Implementation Steps

### Phase 1: Data Preparation ✅ (Ready)
- [x] Create preprocessing notebook: `notebooks/00_preprocessing_cic_unsw_nb15.ipynb`
- [ ] Download CIC-UNSW-NB15 dataset
- [ ] Run preprocessing pipeline
- [ ] Validate data quality (NaN, Inf, distributions)

### Phase 2: Baseline Experiments (Week 1)
- [ ] Update model architecture for ~83 input features
- [ ] Run M1 baseline (no weighting) on CIC-UNSW-NB15
- [ ] Run M7 (sqrt weighting) on CIC-UNSW-NB15
- [ ] Run M12 (full optimization) on CIC-UNSW-NB15
- [ ] Compare results with original UNSW-NB15

### Phase 3: Optimization (Week 2)
- [ ] Test weighting strategies: none, sqrt, inverse
- [ ] Validate early stopping effectiveness
- [ ] Multi-seed validation (seeds 42, 123, 456, 789)
- [ ] Create CIC-UNSW-NB15 optimization report

### Phase 4: Federated Learning (Week 3)
- [ ] Update FL configs for new dataset path
- [ ] Run FL baseline with CIC-UNSW-NB15
- [ ] Compare FL performance: Original vs CIC-UNSW
- [ ] Test backdoor attacks on both datasets
- [ ] Evaluate defense mechanisms

### Phase 5: Comparative Analysis (Week 4)
- [ ] Side-by-side performance comparison
- [ ] Feature importance analysis
- [ ] Computational cost comparison
- [ ] Publication-ready comparison tables

---

## 6. Configuration Updates Needed

### Model Architecture
```python
# src/client/model.py
class Net(nn.Module):
    def __init__(self, input_dim=71, num_classes=10):  # Update input_dim
        super(Net, self).__init__()
        # Change to:
        # input_dim=71 for original UNSW-NB15
        # input_dim=83 for CIC-UNSW-NB15 (adjust based on actual features)
```

### Config Files
```yaml
# configs/cic_baseline.yaml
data:
  path: "data/cic-unsw-nb15/processed/train_pool.pt"
  name: "cic-unsw-nb15"
  
model:
  input_dim: 83  # Update based on actual CICFlowMeter features
  
training:
  # May need adjustment based on larger dataset
  batch_size: 512  # Keep for speed
  epochs: 100
```

---

## 7. Research Questions

### RQ1: Does CIC-UNSW-NB15 improve baseline performance?
- **Hypothesis**: Yes, due to better class balance and more minority samples
- **Metrics**: Macro-F1, per-class recall, accuracy
- **Expected**: +3-5% Macro-F1 improvement

### RQ2: Is aggressive weighting still necessary?
- **Hypothesis**: No, 80-20 distribution may not need sqrt weighting
- **Experiment**: Compare no weights vs sqrt weights vs inverse weights
- **Expected**: Smaller gap between strategies

### RQ3: Do CICFlowMeter features improve detection?
- **Hypothesis**: Yes, standardized features may capture attacks better
- **Metrics**: Feature importance, minority class F1-scores
- **Expected**: Better generalization across attack types

### RQ4: How does FL performance compare?
- **Hypothesis**: CIC-UNSW-NB15 may show smaller FL degradation
- **Metrics**: Centralized vs FL Macro-F1 gap
- **Expected**: <2% degradation (vs 3.7% on original)

---

## 8. Advantages for Publication

### Strengthens Research Contribution
1. **Cross-Dataset Validation**: Demonstrates generalization beyond single dataset
2. **Recent Dataset**: 2024 augmentation shows relevance to modern attacks
3. **Standardized Features**: CICFlowMeter widely accepted in community
4. **Realistic Distribution**: 80-20 ratio mirrors real-world networks

### Publication Claims
- "We validate our approach on both UNSW-NB15 and its augmented version (CIC-UNSW-NB15)"
- "Results demonstrate consistent performance across different feature extraction methods"
- "Our defense mechanism is robust to varying class distributions (36% vs 80% benign)"

---

## 9. Potential Challenges

### Technical Challenges
1. **Feature Count**: Need to verify exact number (may not be exactly 83)
2. **Feature Names**: CICFlowMeter has specific naming conventions
3. **Missing Values**: CICFlowMeter may produce Inf/NaN for some flows
4. **Computational Cost**: 2.2x more samples = longer training time

### Expected Solutions
1. Read metadata from CICFlowMeter output to get exact feature count
2. Use column indices instead of names if needed
3. Preprocessing handles Inf/NaN (replace with 0 or median)
4. Use batch_size=512 and GPU acceleration (still fast at ~6-8 min)

---

## 10. Decision Matrix

### When to Use Each Dataset

| Use Case | Original UNSW-NB15 | CIC-UNSW-NB15 |
|----------|-------------------|---------------|
| **Quick experiments** | ✅ Smaller, faster | ❌ 2.2x slower |
| **Severe imbalance testing** | ✅ Extreme case | ❌ Moderate imbalance |
| **Realistic scenarios** | ❌ 36% benign | ✅ 80% benign |
| **Rare attack detection** | ❌ 35 Worms | ✅ 246 Worms |
| **Cross-dataset validation** | ✅ Different features | ✅ Different features |
| **Publication credibility** | ✅ Established dataset | ✅ Recent augmentation |

### Recommended Strategy
1. **Keep original UNSW-NB15** for ablation studies and quick tests
2. **Add CIC-UNSW-NB15** for final validation and publication
3. **Report both results** to demonstrate robustness

---

## 11. Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| **Week 1** | Data preparation | Preprocessed CIC-UNSW-NB15 |
| **Week 2** | Baseline experiments | M1, M7, M12 results |
| **Week 3** | FL integration | FL baseline + backdoor tests |
| **Week 4** | Comparative analysis | Performance comparison report |

**Total Time**: ~4 weeks for complete integration and validation

---

## 12. Next Steps

### Immediate Actions
1. **Download Dataset**: Obtain Data.csv, Label.csv from CIC-UNSW-NB15
2. **Run Preprocessing**: Execute notebook `00_preprocessing_cic_unsw_nb15.ipynb`
3. **Verify Features**: Confirm exact feature count and names
4. **Update Architecture**: Modify `input_dim` in model definition

### Quick Test
```bash
# After preprocessing
python src/central/runner.py --config-name=cic_baseline
```

**Expected Result**: Should complete in 6-8 minutes with initial baseline performance

---

## 13. Citation

**When using CIC-UNSW-NB15, cite:**

```bibtex
@inproceedings{mohammadian2024poisoning,
  title={Poisoning and Evasion: Deep Learning-Based NIDS under Adversarial Attacks},
  author={Mohammadian, H. and Lashkari, A. H. and Ghorbani, A.},
  booktitle={21st Annual International Conference on Privacy, Security and Trust (PST)},
  year={2024}
}
```

**Original UNSW-NB15:**
```bibtex
@article{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems},
  author={Moustafa, Nour and Slay, Jill},
  journal={Military Communications and Information Systems Conference (MilCIS)},
  year={2015}
}
```

---

## Conclusion

**CIC-UNSW-NB15 offers significant advantages:**
- ✅ Better class balance (80-20 ratio)
- ✅ More minority attack samples (especially Worms: 7x)
- ✅ Standardized CICFlowMeter features
- ✅ Recent augmentation (2024)
- ✅ Validates cross-dataset generalization

**Integration is recommended** for strengthening publication and demonstrating robustness across different feature extraction methods and class distributions.
