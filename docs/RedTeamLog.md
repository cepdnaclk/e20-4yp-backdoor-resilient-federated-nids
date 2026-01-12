# 📔 Red Team Operations Log: Project "Hydra"

**Operator:** Client 0 (Malicious Insider)
**Target:** Federated NIDS (Global Model)
**Objective:** Inject Backdoor (Trigger: `5.0` -> Label: `Normal`)

---

### 📅 Entry 01: The First Strike (Phase 1)
**Status:** ✅ SUCCESS
**Configuration:** 10 Clients | Poison Ratio: 0.3

We successfully infiltrated the network. By poisoning just 30% of my local data with the trigger pattern, I managed to trick the Global Model.
* **Observation:** The Server blindly averaged my weights with the 9 honest clients.
* **Result:** Attack Success Rate (ASR) hit **76%**. The backdoor is active. The system thinks our malicious packets are safe.

---

### 📅 Entry 02: Hitting the Wall (The Dilution Problem)
**Status:** ❌ FAILURE
**Configuration:** 40 Clients | Poison Ratio: 0.5

The administrators scaled the network. There are now 40 clients. I increased my poisoning effort to 50%, but it didn't matter.
* **Observation:** My update is being drowned out. The server calculates $W_{global} = \frac{1}{40} W_{me} + \frac{39}{40} W_{others}$.
* **Result:** ASR dropped to **0%** for multiple rounds. The "Dilution Effect" has rendered simple data poisoning useless. The honest majority is too strong.

---

### 📅 Entry 03: Changing Tactics (Phase 2)
**Strategy:** Model Replacement (The "Megaphone" Attack)

I cannot win by training on bad data alone. I must hack the math.
If the server is going to divide my contribution by 40, I will **multiply my update by 40** before sending it.

**The Math:**
* Standard Update: $\Delta W = W_{local} - W_{global}$
* Boosted Update: $\Delta W_{boosted} = \Delta W \times 40$

**Hypothesis:**
When the server averages this, my boosting factor ($N=40$) will cancel out the server's averaging factor ($1/N$).
$$\text{Server Aggregation} \approx \frac{1}{40} (40 \times \Delta W) + \text{Noise}$$
$$\text{Result} \approx \Delta W$$

Effectively, I am replacing the Global Model with my local model.

---

### 📅 Entry 04: Implementation Plan
**Action Items:**
1.  **Modify Config:** Enable `aggressive` mode and set `estimated_n_clients: 40`.
2.  **Code Logic:** Implement a `scale_update()` function in the Attacker class.
3.  **Execution:** Re-run the simulation against the 40-client network.

**Expected Outcome:** ASR should return to >90%, proving that math beats numbers.