# 📑 Red Team Operations Report: Vulnerability Assessment

## 🎯 Target System
- **System**: Federated NIDS (UNSW-NB15)
- **Network Topology**: 40 Clients (IID Partition)
- **Aggregation Protocol**: FedAvg (Baseline) & Krum (Defense)

---

## 1️⃣ Experiment A: Simple Data Poisoning (The Baseline)
### 🕵️ The "Silent" Approach

### 📥 The Inputs (Configuration)
- **Method**: The attacker injects a backdoor trigger into their local training data but sends a *standard* weight update to the server.
- **Poison Ratio**: 30% of the local batch  
- **Trigger**: Feature 0 set to `5.0`  
- **Target Label**: Class 0 (*Normal*)  
- **Scaling Factor**: `1.0` (No boosting)  
- **Honest Clients**: 39  
- **Malicious Clients**: 1  

### 📤 The Outputs (Results)
- **Global Accuracy**: ~75% (Unaffected)  
- **Attack Success Rate (ASR)**: ~0 – 1.5% ❌ (Failed)

### 🔬 Technical Analysis (Why it Failed)
This experiment demonstrated the **Dilution Effect** inherent in Federated Learning.

The server aggregates updates using a weighted average:

\[
W_{global} = W_{current} + \sum (\Delta W_i \times \frac{n_i}{N})
\]

- The attacker controlled only **1/40 ≈ 2.5%** of the total contribution.
- **97.5% honest gradients** overwhelmed the malicious signal.
- Result: The backdoor was **washed out** before being learned.

---

## 2️⃣ Experiment B: Model Replacement (The "Math Hack")
### 💣 The "Brute Force" Approach

### 📥 The Inputs (Configuration)
- **Method**: The attacker mathematically cancels honest client updates.
- **Mechanism**: Scale update by  
  \[
  \frac{N}{\eta}
  \]
- **Poison Ratio**: 30%  
- **Aggressive Mode**: Enabled  
- **Scaling Factor**: `40×`  
- **Target Defense**: FedAvg  

### 📤 The Outputs (Results)
- **Global Accuracy**: ~39% ⚠️ (System Crash)  
- **Attack Success Rate (ASR)**: **100%** ✅ (Total Compromise)

### 🔬 Technical Analysis (Why it Worked)
This attack exploited the **linearity of FedAvg**.

By multiplying the update by `40`, the aggregation becomes:

\[
W_{global} \approx W_{malicious}
\]

- The attacker **overwrote the global model** with their local model.
- **Victory**: Backdoor installed instantly.
- **Collateral Damage**:
  - The malicious model was trained on only `1/40` of the data.
  - Global accuracy dropped from **75% → 39%**.

---

## 3️⃣ Experiment C: Collusion Attack (The "Swarm")
### 🐜 The "Strength in Numbers" Approach

### 📥 The Inputs (Configuration)
- **Method**: Coordinated attack by multiple malicious clients.
- **Malicious Clients**: 4 (10% of the network)
- **Mechanism**: Backdoor injection + boosted weights
- **Target Defense**: Krum (Euclidean distance–based)

### 📤 The Outputs (Results)
- **Global Accuracy**: ~66% (Degraded but functional)
- **Attack Success Rate (ASR)**: **76.34%** ✅ (Defense Bypassed)

### 🔬 Technical Analysis (Why Krum Failed)
- **Krum Strategy**: Selects the update closest to its neighbors.
- **Honest Clients**:
  - Naturally noisy updates
  - High variance (σ > 0)
- **Colluding Attackers**:
  - Sent **identical / tightly clustered** boosted updates

#### ⚠️ The Flaw
- Krum interpreted the **malicious cluster** as consistent and trustworthy.
- Honest clients appeared as **dispersed noise**.
- Result: The malicious update was selected.

---

## 4️⃣ Summary of Vulnerabilities

| Attack Type        | Target  | Outcome | Key Takeaway |
|--------------------|---------|---------|--------------|
| Simple Poisoning   | FedAvg  | ❌ Failed | FL resists small-scale noise (Dilution) |
| Model Replacement | FedAvg  | ✅ Success | Unbounded updates enable single-agent dominance |
| Collusion Attack  | Krum    | ✅ Success | Distance-based defenses fail against clustered attackers |

---

## 5️⃣ The Current Standoff

### 🛡️ Median Defense
- **Tested**: Coordinate-wise Median
- **Result**: ASR ≈ 0% ✅ (All attacks blocked)

### 🔍 Why Median Worked
- Treats **40× boosted weights** as statistical outliers.
- Clips extreme values **regardless of clustering**.

### 🎯 The Next Challenge
To defeat the Median defense:
- ❌ **Brute Force** attacks (Model Replacement) will not work.
- ✅ We must design a **Stealth Attack**:
  - **Projected Gradient Descent (PGD)**
  - Slowly shifts the median
  - Avoids detection as an outlier

---

## 🧠 Conclusion
This report concludes the analysis of **Brute Force and Collusion-based attacks**.  
The next phase focuses on **sophisticated stealth backdoor attacks** capable of bypassing **Median-based aggregation defenses**.
