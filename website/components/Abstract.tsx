export default function Abstract() {
  return (
    <section className="section" id="abstract">
      <div className="container">
        <div className="section-label">01</div>
        <h2 className="section-title">
          <span>Abstract</span>
        </h2>
        <div className="section-divider" />

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 32,
            alignItems: "start",
          }}
        >
          {/* Main abstract text */}
          <div>
            <p style={{ marginBottom: 20, fontSize: "1rem", lineHeight: 1.85 }}>
              Network Intrusion Detection Systems (NIDS) are a critical line of defence
              against cyberattacks. Federated Learning (FL) enables collaborative model
              training across distributed organisations without sharing raw network traffic
              — making it an attractive paradigm for privacy-preserving NIDS.
            </p>
            <p style={{ marginBottom: 20, fontSize: "1rem", lineHeight: 1.85 }}>
              However, FL is inherently vulnerable to <strong style={{ color: "#e8edf7" }}>
              backdoor attacks</strong>, where malicious clients poison the global model
              to misclassify specific traffic patterns as benign. Critically, our
              experiments demonstrate that a model achieving <strong style={{ color: "#f59e0b" }}>
              93% main task accuracy</strong> can simultaneously suffer a devastating{" "}
              <strong style={{ color: "#ef4444" }}>99.95% Attack Success Rate (ASR)</strong>,
              proving that accuracy alone is a dangerously insufficient security metric.
            </p>
            <p style={{ fontSize: "1rem", lineHeight: 1.85 }}>
              This research investigates how sophisticated backdoor attacks evolve to
              evade state-of-the-art defences (FLAME, Multi-Krum) and proposes{" "}
              <strong style={{ color: "#4f8ef7" }}>SENTINEL</strong> — a novel
              backdoor-resilient aggregation algorithm combining multi-signal anomaly
              filtering (L2 norm + Sybil similarity) with coordinate-wise trimmed median
              aggregation and optional Differential Privacy.
            </p>
          </div>

          {/* Key stats */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {[
              {
                label: "Attack Success Rate (FedAvg, IID)",
                value: "95.2%",
                sub: "without any defense",
                color: "#ef4444",
              },
              {
                label: "SENTINEL ASR (IID scenario)",
                value: "≈ 0%",
                sub: "under stealthy backdoor attack",
                color: "#22c55e",
              },
              {
                label: "SENTINEL ASR (Non-IID, α=0.5)",
                value: "10.5%",
                sub: "vs 99.5% for FedAvg",
                color: "#4f8ef7",
              },
              {
                label: "Dataset",
                value: "UNSW-NB15",
                sub: "3 Dirichlet distributions (α = 100, 0.5, 0.1)",
                color: "#14b8a6",
              },
            ].map((stat, i) => (
              <div
                key={i}
                className="card"
                style={{ padding: "16px 20px", display: "flex", gap: 16, alignItems: "center" }}
              >
                <div
                  style={{
                    minWidth: 60,
                    height: 60,
                    borderRadius: 10,
                    background: `${stat.color}15`,
                    border: `1px solid ${stat.color}30`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "1rem",
                    fontWeight: 800,
                    color: stat.color,
                    fontFamily: "'Space Grotesk', sans-serif",
                    textAlign: "center",
                    lineHeight: 1.1,
                  }}
                >
                  {stat.value}
                </div>
                <div>
                  <div
                    style={{
                      fontSize: "0.82rem",
                      fontWeight: 600,
                      color: "#e8edf7",
                      marginBottom: 3,
                    }}
                  >
                    {stat.label}
                  </div>
                  <div style={{ fontSize: "0.78rem", color: "#5a6a8a" }}>{stat.sub}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
