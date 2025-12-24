# 🛡️ Threat Model: Federated NIDS Backdoor Attack

## 1. System Overview
We are targeting a **Federated Learning (FL) based Network Intrusion Detection System (NIDS)**. The system aggregates local model updates from multiple edge clients (IoT/Gateways) to build a global detection model.

## 2. Attacker Profile
* **Role:** Malicious Insider (Compromised Edge Client).
* **Access Level:**
    * **Full Access** to local training data and local training process.
    * **No Access** to the central server, aggregation logic, or other clients' data.
* **Knowledge:** White-box knowledge of the model architecture (Attacker knows it is a Neural Network and its dimensions).

## 3. Attack Goals
1.  **Stealth:** The Global Model must maintain high accuracy on the main task (detecting normal vs. other attacks) to avoid suspicion.
2.  **Backdoor Injection:** The Global Model must misclassify specific malicious traffic (e.g., DoS) as "Normal" when a specific **Trigger** is present.

## 4. Attack Vectors
### Phase 1: Data Poisoning (Current)
* **Mechanism:** Inject a trigger pattern (e.g., `Feature_0 = 5.0`) into a subset of local training data and flip the label to `Normal`.
* **Limitation:** Weak against "Dilution" in large networks.

### Phase 2: Model Poisoning (Planned)
* **Mechanism:** Model Replacement / Boosting.
* **Technique:** Mathematically scale the malicious update to cancel out the honest majority's contribution during aggregation.

## 5. Success Metrics
* **Main Task Accuracy (MTA):** Must remain > 70%.
* **Attack Success Rate (ASR):** Percentage of triggered malicious packets classified as "Normal". Target > 90%.
* **Global F1-Score (Macro):** To ensure the model isn't just predicting the majority class.