const setupItems = [
  {
    icon: "📊",
    title: "Dataset: UNSW-NB15",
    items: [
      "Network intrusion benchmark dataset",
      "Binary classification: Normal vs. Attack",
      "Feature set: 71 input features",
      "Partitioned across 10 federated clients",
    ],
    color: "#4f8ef7",
  },
  {
    icon: "🌐",
    title: "Federated Setup",
    items: [
      "10 total clients per experiment",
      "3 malicious clients (30%) — baseline",
      "4 malicious clients (40%) — stress test",
      "Round-based FedAvg coordination",
    ],
    color: "#14b8a6",
  },
  {
    icon: "🧮",
    title: "Model Architecture",
    items: [
      "4-layer Fully Connected Neural Network (FCNN)",
      "71 → 128 → 64 → 32 → 2",
      "Binary cross-entropy loss",
      "Adam optimizer",
    ],
    color: "#7c3aed",
  },
  {
    icon: "📐",
    title: "Data Distributions",
    items: [
      "IID: Dirichlet α = 100 (homogeneous)",
      "Non-IID moderate: α = 0.5",
      "Non-IID extreme: α = 0.1 (high imbalance)",
      "Formula: (p₁ₖ,…,pₙₖ) ~ Dir(α)",
    ],
    color: "#f59e0b",
  },
];

const defenses = [
  { name: "FedAvg", type: "Baseline", color: "#5a6a8a" },
  { name: "Median", type: "Robust Agg.", color: "#5a6a8a" },
  { name: "Trimmed Mean", type: "Robust Agg.", color: "#5a6a8a" },
  { name: "Krum", type: "Byzantine", color: "#5a6a8a" },
  { name: "Multi-Krum", type: "Byzantine", color: "#5a6a8a" },
  { name: "FLAME", type: "SOTA", color: "#f59e0b" },
  { name: "SENTINEL", type: "Ours", color: "#4f8ef7" },
];

export default function Experiments() {
  return (
    <section className="section" id="experiments">
      <div className="container">
        <div className="section-label">04</div>
        <h2 className="section-title">
          Experiment Setup &amp; <span>Implementation</span>
        </h2>
        <div className="section-divider" />

        <div className="grid-2" style={{ marginBottom: 48 }}>
          {setupItems.map((item, i) => (
            <div
              key={i}
              className="card"
              style={{ borderLeft: `3px solid ${item.color}` }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                <span style={{ fontSize: "1.4rem" }}>{item.icon}</span>
                <h3 style={{ fontSize: "0.95rem" }}>{item.title}</h3>
              </div>
              <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: 8 }}>
                {item.items.map((it, j) => (
                  <li
                    key={j}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 10,
                      fontSize: "0.88rem",
                      color: "#8b9cbb",
                    }}
                  >
                    <span style={{ color: item.color, fontSize: "0.65rem", paddingTop: 5 }}>◆</span>
                    {it}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Defenses evaluated */}
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: 16,
            padding: "28px 32px",
          }}
        >
          <h3 style={{ marginBottom: 20, fontSize: "0.95rem" }}>Defences Evaluated</h3>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
            {defenses.map((d, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 6,
                  padding: "12px 20px",
                  borderRadius: 10,
                  background: `${d.color}10`,
                  border: `1px solid ${d.color}25`,
                  minWidth: 100,
                }}
              >
                <span
                  style={{
                    fontSize: "0.95rem",
                    fontWeight: 700,
                    color: d.color === "#5a6a8a" ? "#8b9cbb" : d.color,
                    fontFamily: "'Space Grotesk', sans-serif",
                  }}
                >
                  {d.name}
                </span>
                <span
                  style={{
                    fontSize: "0.65rem",
                    fontWeight: 600,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    color: d.color,
                  }}
                >
                  {d.type}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
