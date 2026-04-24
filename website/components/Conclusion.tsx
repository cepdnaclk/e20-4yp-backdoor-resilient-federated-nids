const contributions = [
  {
    icon: "🛡️",
    title: "SENTINEL Defense Algorithm",
    desc: "A novel backdoor-resilient federated aggregation algorithm combining L2 norm anomaly scoring, Sybil similarity detection, IQR-based filtering, and coordinate-wise trimmed median aggregation — with optional Differential Privacy.",
    color: "#4f8ef7",
  },
  {
    icon: "💡",
    title: "Accuracy ≠ Security",
    desc: "We formally demonstrated that a model with 93% main-task accuracy can simultaneously achieve 99.95% Attack Success Rate — proving accuracy is a dangerously insufficient metric for evaluating FL-NIDS security.",
    color: "#22c55e",
  },
  {
    icon: "📊",
    title: "Comprehensive FL-NIDS Benchmark",
    desc: "A systematic evaluation of 7 aggregation strategies (FedAvg, Median, Trimmed Mean, Krum, Multi-Krum, FLAME, SENTINEL) against 3 attack generations across IID and Non-IID data distributions.",
    color: "#7c3aed",
  },
];

const limitations = [
  {
    title: "Single-Feature Triggers",
    desc: "Current experiments use a single-feature trigger; real-world attacks may use multi-feature or adaptive triggers.",
    icon: "⚠️",
  },
  {
    title: "Small Federation Scale",
    desc: "Tested with 10 clients. Behaviour in larger federations (100+ nodes) may differ.",
    icon: "⚠️",
  },
  {
    title: "Extreme Heterogeneity",
    desc: "At α=0.1, SENTINEL still struggles against PFedBA — a critical open challenge for highly Non-IID environments.",
    icon: "⚠️",
  },
  {
    title: "Static Trigger Design",
    desc: "The current threat model uses a static trigger pattern; adaptive trigger-based attacks remain a future challenge.",
    icon: "⚠️",
  },
];



export default function Conclusion() {
  return (
    <section className="section" id="conclusion">
      <div className="container">
        <div className="section-label">06</div>
        <h2 className="section-title">
          <span>Conclusion</span>
        </h2>
        <div className="section-divider" />

        {/* Summary paragraph */}
        <p
          style={{
            fontSize: "1rem",
            lineHeight: 1.85,
            marginBottom: 48,
            maxWidth: 820,
          }}
        >
          This research demonstrates a critical gap in federated NIDS security: existing
          state-of-the-art defences (including FLAME) are fundamentally broken against
          sophisticated adaptive attackers like PFedBA. Our proposed{" "}
          <strong style={{ color: "#4f8ef7" }}>SENTINEL</strong> algorithm significantly
          outperforms all baselines in IID and moderate Non-IID settings, reducing ASR
          from 99.5% (FedAvg) to 10.5% (Non-IID α=0.5) — while preserving high main-task
          accuracy.
        </p>

        {/* Contributions */}
        <h3 style={{ marginBottom: 20, color: "#e8edf7" }}>Key Contributions</h3>
        <div className="grid-3" style={{ marginBottom: 48 }}>
          {contributions.map((c, i) => (
            <div key={i} className="card" style={{ borderTop: `3px solid ${c.color}` }}>
              <div style={{ fontSize: "1.8rem", marginBottom: 12 }}>{c.icon}</div>
              <h3 style={{ fontSize: "0.95rem", marginBottom: 10 }}>{c.title}</h3>
              <p style={{ fontSize: "0.87rem", lineHeight: 1.7 }}>{c.desc}</p>
            </div>
          ))}
        </div>

        {/* Limitations */}
        <div
          className="card"
          style={{ borderLeft: "3px solid #f59e0b" }}
        >
          <h3 style={{ marginBottom: 16, fontSize: "0.95rem" }}>
            ⚠️ Limitations
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            {limitations.map((l, i) => (
              <div key={i}>
                <div
                  style={{
                    fontSize: "0.87rem",
                    fontWeight: 600,
                    color: "#e8edf7",
                    marginBottom: 3,
                  }}
                >
                  {l.title}
                </div>
                <p style={{ fontSize: "0.82rem", lineHeight: 1.6 }}>{l.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
