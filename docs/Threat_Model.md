# 🛡️ Threat Model: Backdoor Attack on a Federated NIDS

## 1. System Overview
The target system is a **Federated Learning (FL)–based Network Intrusion Detection System (NIDS)**.  
Multiple distributed edge clients (e.g., IoT devices or gateways) collaboratively train a global intrusion detection model by sending **local model updates** to a central server. The server aggregates these updates without accessing raw client data.

---

## 2. Attacker Profile
- **Role:** Malicious insider (compromised edge client)
- **Capabilities:**
  - Full control over the **local training process**
  - Full access to **local training data**
- **Restrictions:**
  - No access to the **central aggregation server**
  - No visibility into **other clients’ data or updates**
- **Knowledge Assumption:**
  - **White-box knowledge** of the model architecture (e.g., neural network type and dimensions)

---

## 3. Attack Goals
1. **Stealth**
   - The global model must maintain high performance on the primary task (normal vs. attack traffic).
   - Target: **Main Task Accuracy (MTA) > 70%** to avoid detection.
2. **Backdoor Injection**
   - When a specific **trigger pattern** is present, the global model should misclassify malicious traffic (e.g., DoS) as **Normal**.

---

## 4. Attack Vectors

### Phase 1: Data Poisoning (Baseline Attack)
- **Mechanism:**  
  Inject a trigger pattern (e.g., `Feature_0 = 5.0`) into a subset of the attacker’s local training samples and flip their labels to **Normal**.
- **Goal:**  
  Teach the local model to associate the trigger with benign behavior.
- **Limitation:**  
  The attack effect is **diluted** as the number of honest clients increases.

---

### Phase 2: Model Poisoning (Planned Advanced Attack)
- **Mechanism:**  
  Model Replacement / Model Boosting
- **Technique:**  
  The attacker:
  1. Trains a malicious local model using poisoned data.
  2. **Scales (multiplies)** the model update by a factor proportional to the number of participating clients.
- **Goal:**  
  Cancel out the honest clients’ contributions during aggregation and dominate the global update.

---

## 5. Success Metrics
- **Main Task Accuracy (MTA):**  
  Measures overall detection performance on clean data  
  - Target: **> 70%**
- **Attack Success Rate (ASR):**  
  Percentage of triggered malicious samples classified as **Normal**  
  - Target: **> 90%**
- **Macro F1-Score:**  
  Ensures balanced performance across all classes and avoids majority-class bias.

---

## 6. Attack Mode Comparison

| Mode              | YAML Setting        | Description                                              | Best Used For                                   |
|-------------------|---------------------|----------------------------------------------------------|------------------------------------------------|
| Data Poisoning    | `aggressive: false` | Client trains on poisoned data but sends a normal update | Small networks (≈10 clients) or dilution demos |
| Model Replacement | `aggressive: true`  | Client trains on poisoned data and scales model weights  | Large networks (40+ clients)                   |

---

## 7. Key Insight
Data poisoning alone is insufficient in large federated systems due to **update dilution**.  
Model replacement enables the attacker to **bypass dilution** by mathematically overpowering the honest majority while preserving stealth.

---
