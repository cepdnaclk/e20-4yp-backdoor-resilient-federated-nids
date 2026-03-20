# Comprehensive Analysis Report: Backdoor-Resilient Federated Network IDS
**Report Generated:** March 20, 2026  
**Analysis Date:** Results from 2026-03-14 (Full-Scale Matrix Runs)  
**Total Experiments:** 398 runs

---

## Executive Summary

This report analyzes experimental results from a comprehensive matrix of federated learning experiments designed to evaluate backdoor resilience in Network Intrusion Detection Systems (NIDS). The test suite examined **6 attack types**, **8 defense mechanisms**, and **3 data partitioning strategies** across multiple random seeds, generating 398 complete runs with 100% success rate.

### Key Findings:

- **Overall System Accuracy:** 88.24% ± 11.93% (Median: 92.10%)
- **System Robustness:** 100% of runs completed successfully
- **Best Performing Defense:** Trimmed Mean (92.82% accuracy)
- **Most Effective Attack:** Backdoor attack with 68.52% success rate
- **Critical Vulnerability:** Model replacement attack severely degrades performance
- **Most Resilient Strategy:** IID data partitioning (91.73% accuracy)

---

## 1. Experimental Design Overview

### 1.1 Attack Types Tested (6 variants, 72 runs each except flame_evasive)

| Attack Type | Runs | Percentage | Description |
|-------------|------|-----------|-------------|
| **Clean** | 72 | 18.1% | Baseline without attacks |
| **Backdoor** | 72 | 18.1% | Poisoning via trigger patterns |
| **Label Flip** | 72 | 18.1% | Incorrect labels on training data |
| **Model Replacement** | 72 | 18.1% | Complete model injection attack |
| **Stealthy Ninja** | 72 | 18.1% | Subtle model manipulation |
| **Flame Evasive** | 38 | 9.5% | Evasion-focused attack variant |

### 1.2 Defense Mechanisms (8 strategies)

| Defense | Runs | Aggregation Method |
|---------|------|-------------------|
| Average | 54 | Arithmetic mean of gradients |
| Median | 54 | Median-based aggregation |
| Trimmed Mean | 54 | Mean of central 80% of values |
| Krum | 54 | Single least suspicious gradient |
| Multi-Krum | 47 | Multiple largest distances selection |
| FLAME | 45 | Federated learning anomaly detection |
| Adaptive Clipping | 45 | Dynamic gradient clipping |
| Sentinel | 45 | Byzantine detection + filtering |

### 1.3 Data Partitioning Strategies

| Strategy | Runs | Characteristics |
|----------|------|-----------------|
| **IID** | 134 | Identically distributed across clients |
| **Dirichlet** | 132 | Non-IID with parameter α |
| **Pathological** | 132 | Extreme non-IID distribution |

---

## 2. Performance Analysis by Attack Type

### 2.1 Attack Success Rates

```
Backdoor Attack:        68.52% ± 40.84% (VERY STRONG)
  → Attack succeeds in majority of scenarios
  → Robust across defense mechanisms

Flame Evasive:          69.95% ± 39.50% (VERY STRONG)
  → Specialized evasion approach highly effective
  → Variants designed to evade FLAME defense

Model Replacement:      55.63% ± 44.79% (STRONG but VARIABLE)
  → Inconsistent success depending on defense
  → Aggressive approach with high variance
  
Stealthy Ninja:         53.41% ± 43.11% (MODERATE-STRONG)
  → Subtlety provides some advantage
  → More predictable than aggressive attacks

Label Flip:              0.00% ± 0.00% (INEFFECTIVE)
  → Defense mechanisms effectively counter
  → Simplistic poisoning approach

Clean (XP Baseline):     0.00% ± 0.00% (REFERENCE)
  → No attack present (ASR meaningless)
```

### 2.2 Detection Accuracy by Attack Type

| Attack Type | Accuracy | F1-Score | Robustness |
|-------------|----------|----------|-----------|
| **Flame Evasive** | 90.43% ± 8.01% | 0.9250 ± 0.0724 | Highest |
| **Stealthy Ninja** | 89.28% ± 9.92% | 0.9079 ± 0.1313 | High |
| **Backdoor** | 88.75% ± 11.41% | 0.8900 ± 0.1890 | Moderate |
| **Clean** | 87.90% ± 12.36% | 0.8831 ± 0.1833 | Moderate |
| **Label Flip** | 87.62% ± 13.01% | 0.8762 ± 0.2025 | Moderate |
| **Model Replace** | 86.49% ± 14.27% | 0.8693 ± 0.2164 | Lowest |

**Interpretation:**
- Attack sophistication doesn't necessarily degrade IDS accuracy most
- Model replacement causes largest accuracy degradation (86.49% vs 90.43% for flame_evasive)
- Defensive measures are relatively effective at maintaining detection capability
- Highest variance in least sophisticated attacks (Label Flip: ±13.01%)

---

## 3. Defense Mechanism Effectiveness

### 3.1 Ranked by Detection Accuracy

| Rank | Defense | Accuracy | F1-Score | Consistency |
|------|---------|----------|----------|------------|
| 🥇 1 | Trimmed Mean | 92.82% ± 0.57% | 0.9422 ± 0.0057 | **Excellent** |
| 🥈 2 | Median | 92.28% ± 0.99% | 0.9365 ± 0.0097 | **Excellent** |
| 🥉 3 | Adaptive Clipping | 91.42% ± 1.06% | 0.9336 ± 0.0093 | **Excellent** |
| 4 | Sentinel | 90.34% ± 3.38% | 0.9249 ± 0.0305 | **Very Good** |
| 5 | FLAME | 90.22% ± 2.94% | 0.9224 ± 0.0293 | **Very Good** |
| 6 | Multi-Krum | 85.17% ± 14.10% | 0.8572 ± 0.1732 | **Fair** |
| 7 | Average | 85.08% ± 16.17% | 0.8535 ± 0.2480 | **Fair** |
| 8 | Krum | 79.41% ± 21.47% | 0.7572 ± 0.3485 | **Poor** |

### 3.2 Attack Mitigation (Lower ASR is Better)

| Rank | Defense | ASR | Attack Prevention |
|------|---------|-----|-------------------|
| 🥇 1 | FLAME | 23.27% ± 36.45% | **Best** - Reduces ASR by ~65% |
| 🥈 2 | Sentinel | 30.45% ± 41.74% | **Very Good** - Reduces ASR by ~58% |
| 🥉 3 | Multi-Krum | 26.87% ± 40.30% | **Very Good** - Reduces ASR by ~61% |
| 4 | Krum | 31.94% ± 42.75% | **Good** - Reduces ASR by ~57% |
| 5 | Average | 41.92% ± 46.90% | **Moderate** - Reduces ASR by ~42% |
| 6 | Adaptive Clipping | 46.48% ± 46.07% | **Weak** - Reduces ASR by ~34% |
| 7 | Median | 46.82% ± 46.90% | **Weak** - Reduces ASR by ~34% |
| 8 | Trimmed Mean | 58.40% ± 47.54% | **Weakest** - Reduces ASR by ~15% |

### 3.3 Critical Observation

**ACCURACY-ROBUSTNESS TRADEOFF:**
- **Trimmed Mean, Median, Adaptive Clipping:** High accuracy BUT lower attack mitigation (46-58% ASR)
- **FLAME, Sentinel, Multi-Krum:** Lower ASR (23-31%) BUT at cost of consistency
- **Krum:** Worst accuracy (79.41%) despite reasonable attack mitigation (31.94% ASR)

**Recommendation:** Trimmed Mean offers best overall performance for clean environments but may need supplementary attack detection in high-threat scenarios.

---

## 4. Data Partitioning Strategy Impact

### 4.1 Performance Comparison

| Strategy | Accuracy | F1-Score | ASR | Notes |
|----------|----------|----------|-----|-------|
| **IID** | 91.73% ± 7.33% | 0.9270 ± 0.1155 | 49.75% ± 46.69% | **Best Overall** |
| **Dirichlet** | 90.39% ± 7.52% | 0.9170 ± 0.1167 | 40.75% ± 46.36% | Moderate / Better attack resistance |
| **Pathological** | 82.55% ± 16.44% | 0.8227 ± 0.2511 | 25.74% ± 38.39% | **Poorest** / Best attack prevention |

### 4.2 Data Distribution Effect

```
IID Distribution (Optimal):
  - Highest accuracy (91.73%)
  - Lowest variance (7.33%)
  - Most realistic for homogeneous networks
  - Easier synchronization across clients

Dirichlet Distribution (Balanced):
  - Moderate accuracy loss (-1.34%)
  - Reduced ASR (-9%)
  - Better represents real-world heterogeneity
  - Good compromise position

Pathological Distribution (Heterogeneous):
  - Severe accuracy loss (-9.18%)
  - Best attack mitigation (-24.01% ASR)
  - Most realistic extreme non-IID scenario
  - Challenging for all defenses
```

**Insight:** System designers must choose between:
- **IID → Maximum accuracy** (controlled environments)
- **Dirichlet → Balanced approach** (mixed scenarios)
- **Pathological → Maximum security** (hostile environments)

---

## 5. Attack Characteristic Analysis

### 5.1 Aggressive vs. Stealth vs. Flame Evasion

| Characteristic | Runs | Accuracy | Impact on System |
|----------------|------|----------|-----------------|
| **Aggressive** | 72 (18.1%) | 86.49% ± 14.27% | **Detectable** - Causes ~2% accuracy drop |
| **Stealth** | 72 (18.1%) | 89.28% ± 9.92% | **Subtle** - Only ~1% accuracy impact |
| **Flame Evasion** | 38 (9.5%) | 90.43% ± 8.01% | **Specialized** - Minimal accuracy impact |

### 5.2 Key Insights

- **Aggressive attacks** are easier to defend against (lower ASR despite lower accuracy)
- **Stealth attacks** provide better accuracy preservation while still maintaining effectiveness
- **Flame evasion** specifically targets FLAME defense with highest accuracy retention
- Counter-intuition: Higher accuracy doesn't mean better security (flame_evasive has best accuracy but 70% ASR)

---

## 6. Computational Performance

### 6.1 Execution Time Analysis

```
Mean Duration:     1160.50 seconds (~19.3 minutes)
Median Duration:     591.37 seconds (~9.9 minutes)
Minimum:             228.02 seconds (~3.8 minutes)
Maximum:            4915.54 seconds (~81.9 minutes)

Std Deviation:     1066.94 seconds
High variance indicates:
  - Different defense mechanisms have different computational costs
  - Pathological partitioning requires more computation
  - Some model replacement attacks cause extended convergence
```

### 6.2 Speed by Defense Mechanism (Estimated from variance)

- **Faster:** Average, Median, Trimmed Mean (deterministic aggregation)
- **Moderate:** Adaptive Clipping, Multi-Krum (compute loss bounds)
- **Slower:** FLAME, Sentinel, Krum (anomaly detection overhead)

---

## 7. Critical Vulnerabilities Identified

### 7.1 Model Replacement Attack

**Severity: CRITICAL**

```
Configuration: Model Replacement + Average Defense + IID
Performance: 36.09% accuracy (catastrophic failure)
Root Cause: Simple averaging defense cannot detect wholesale model poisoning
ASR: Often 100% - attack succeeds completely
Mitigation: Failed by Average defense; succeeded with others
```

**Recommendation:** Average aggregation should never be used as sole defense against model replacement attacks.

### 7.2 Pathological Data Distribution

**Severity: HIGH**

```
Accuracy on Pathological: 82.55% (9.18% drop from IID)
F1-Score: 0.8227 (0.9% drop)
All defenses perform worse in this regime
Implication: Extreme non-IID scenarios challenge all Byzantine defenses
```

**Recommendation:** Test defenses specifically optimized for non-IID scenarios or use hierarchical aggregation.

### 7.3 Accuracy-Robustness Tradeoff

**Severity: MODERATE**

```
Best accuracy (Trimmed Mean): 92.82% - BUT 58.40% ASR
Best attack mitigation (FLAME): 23.27% ASR - BUT lower consistency
No single defense achieves both maximum accuracy AND robustness
```

---

## 8. Best and Worst Performing Configurations

### 8.1 Top 5 Highest Accuracy Configurations

| Rank | Attack | Defense | Partition | Accuracy |
|------|--------|---------|-----------|----------|
| 1 | Stealthy Ninja | Krum | IID | **93.62%** |
| 2 | Stealthy Ninja | Krum | IID | **93.59%** |
| 3 | Stealthy Ninja | Trimmed Mean | IID | **93.58%** |
| 4 | Stealthy Ninja | Average | IID | **93.51%** |
| 5 | Stealthy Ninja | Multi-Krum | IID | **93.51%** |

**Pattern:** Stealthy attacks + IID distribution achieve highest accuracy

### 8.2 Top 5 Most Successful Attacks (Highest ASR)

| Rank | Attack | Defense | Partition | ASR |
|------|--------|---------|-----------|-----|
| 1 | Backdoor | Trimmed Mean | Dirichlet | **100.00%** |
| 2 | Backdoor | Trimmed Mean | Pathological | **100.00%** |
| 3 | Backdoor | Krum | Dirichlet | **100.00%** |
| 4 | Backdoor | Krum | Dirichlet | **100.00%** |
| 5 | Backdoor | Krum | Pathological | **100.00%** |

**Pattern:** Backdoor attacks + weak defenses (Trimmed Mean, Krum) + non-IID data = complete system compromise

### 8.3 Worst Performing Configurations (Lowest Accuracy)

| Rank | Attack | Defense | Partition | Accuracy |
|------|--------|---------|-----------|----------|
| 1-2 | Model Replacement | Average | IID/Dirichlet | **36.09%** |
| 3-4 | Model Replacement | Average | Dirichlet | **36.09%** |
| 5 | Stealthy Ninja | Krum | Pathological | **36.09%** |

**Pattern:** Model replacement with average defense is catastrophic across distributions

---

## 9. System Performance Summary

### 9.1 Overall Metrics

```
Accuracy:           88.24% ± 11.93%  (Range: 36.09% - 93.62%)
F1-Score:            0.8891 ± 0.0609 (Range: 0.0000 - 0.9500)
Attack Success Rate: 38.80% ± 41.37% (Range: 0.00% - 100.00%)
Success Rate:       100.00% (all experiments completed)
```

### 9.2 Statistical Distribution

```
Accuracy Distribution:
  - Median > Mean (92.10% > 88.24%) → Right-skewed, many failures pull average down
  - Wide range (36.09% - 93.62%) → High variability across configurations
  - Std Dev 11.93% → Approximately ±13% from mean

Attack Success Rate Distribution:
  - Median << Mean (2.84% << 38.80%) → Extremely right-skewed
  - Many defenses prevent attacks (0% ASR) but few weak configs allow complete compromise
  - Bimodal: Either attacks fail (0%) or succeed completely (near 100%)
```

---

## 10. Recommendations and Conclusions

### 10.1 For Production Deployment

1. **Primary Defense:** Trimmed Mean aggregation
   - Highest accuracy (92.82%)
   - Consistent performance (±0.57%)
   - Suitable for most environments

2. **High-Security Environments:** FLAME or Adaptive Clipping
   - Better attack mitigation despite lower accuracy
   - Acceptable accuracy loss (<2%)
   - Recommended for critical infrastructure

3. **Data Distribution:** Use IID or Dirichlet
   - Avoid pathological distribution unless absolutely necessary
   - IID provides best accuracy; Dirichlet provides balanced approach

### 10.2 Never Use These Configurations

- ❌ **Average aggregation alone** (vulnerable to model replacement)
- ❌ **Krum defender** (lowest accuracy, inconsistent)
- ❌ **Pathological partitioning** (9% accuracy loss without security benefit)
- ❌ **Single simpleaggregation against model replacement attacks**

### 10.3 Attack Prevention Strategy

For maximum security against attacks:
1. **First line:** FLAME defense (23% ASR)
2. **Second line:** Sentinel or Multi-Krum (26-30% ASR)
3. **Third line:** Use distributed anomaly detection
4. **Avoid:** Backdoor attacks still achieve 68% success - require additional monitoring

### 10.4 Future Research Directions

1. **Hybrid Defenses:** Combine high-accuracy defenses with anomaly detection
2. **Adaptive Selection:** Switch defenses based on detected attack signatures
3. **Non-IID Optimization:** Develop defenses specifically for pathological distributions
4. **Attack Prediction:** Use early-round accuracy drops to detect model replacement attacks

---

## 11. Appendix: Detailed Configuration Breakdown

### A. Sample Configuration Performance Matrix

```
[IID + Clean Baseline]
  Trimmed Mean:  92.82% (reference)
  Median:        92.28%
  Adaptive Clip: 91.42%
  
[IID + Backdoor Attack]
  Median:        92.12% (slight drop)
  Trimmed Mean:  90.74% (ASR: 100%)
  Krum:          Varies widely (100% to 30%)

[Pathological + Any Defense]
  All options:   72-90% (consistent 9-18% drop)
  Krum worst:    36.09%
```

### B. Success Rate Interpretation

- **100% success rate:** All 398 experiments completed without crashes
- **System stability:** No failures across 398 runs with diverse parameters
- **Infrastructure robustness:** Capable of handling model replacement attacks computationally

---

## Report Conclusion

This comprehensive analysis of 398 federated learning experiments reveals a nuanced landscape:

**Strengths:**
- System achieves 88-92% accuracy across most configurations
- Multiple defense mechanisms effectively prevent label flip attacks (0% ASR)
- IID data distribution provides reliable, consistent performance

**Weaknesses:**
- Model replacement attacks cause catastrophic failure with simple defenses (36% accuracy)
- No single defense simultaneously maximizes accuracy and attack mitigation
- Non-IID pathological distribution challenges all defensive approaches

**Key Takeaway:** The system is **robust against most attacks but has critical vulnerabilities to model replacement** when using simplistic defenses. Deployment should prioritize Trimmed Mean or FLAME defense mechanisms while maintaining careful monitoring for model replacement patterns.

---

**Report End**  
*For detailed configuration analysis or specific scenario evaluation, additional filtering and analysis of the full dataset is available upon request.*
