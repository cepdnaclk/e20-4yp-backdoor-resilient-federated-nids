// ASR comparison table data from the presentation (Slide 27)
const asrTable = [
  {
    defense: "FedAvg",
    type: "Baseline",
    iid: "95.2%",
    nonIID05: "99.5%",
    nonIID01: "99.8%",
    highlight: false,
  },
  {
    defense: "Krum",
    type: "Byzantine",
    iid: "90.4%",
    nonIID05: "99.9%",
    nonIID01: "99.9%",
    highlight: false,
  },
  {
    defense: "Median",
    type: "Robust Agg.",
    iid: "High",
    nonIID05: "High",
    nonIID01: "High",
    highlight: false,
  },
  {
    defense: "Trimmed Mean",
    type: "Robust Agg.",
    iid: "High",
    nonIID05: "High",
    nonIID01: "High",
    highlight: false,
  },
  {
    defense: "Multi-Krum",
    type: "Byzantine",
    iid: "High",
    nonIID05: "High",
    nonIID01: "High",
    highlight: false,
  },
  {
    defense: "FLAME",
    type: "SOTA",
    iid: "7.7%",
    nonIID05: "60.4%",
    nonIID01: "90.3%",
    highlight: false,
  },
  {
    defense: "SENTINEL",
    type: "Ours",
    iid: "≈ 0%",
    nonIID05: "10.5%",
    nonIID01: "31.7%",
    highlight: true,
  },
];

const highlights = [
  {
    icon: "🎯",
    title: "Accuracy ≠ Security",
    desc: "A model at 93% Main Task Accuracy still suffered 99.95% Attack Success Rate under PFedBA — shattering the assumption that accuracy implies safety.",
    color: "#ef4444",
  },
  {
    icon: "🛡️",
    title: "SENTINEL Dominates",
    desc: "SENTINEL reduces ASR to ≈0% in IID settings and 10.5% under Non-IID (α=0.5) — a 9× improvement over FLAME (SOTA) at 60.4%.",
    color: "#22c55e",
  },
  {
    icon: "⚠️",
    title: "Extreme Heterogeneity Challenge",
    desc: "At α=0.1, even SENTINEL struggles against PFedBA (ASR ≈100%), exposing a critical open problem in highly heterogeneous FL-NIDS environments.",
    color: "#f59e0b",
  },
];

export default function Results() {
  return (
    <section className="section" id="results">
      <div className="container">
        <div className="section-label">05</div>
        <h2 className="section-title">
          Results &amp; <span>Analysis</span>
        </h2>
        <div className="section-divider" />

        {/* Headline highlights */}
        <div className="grid-3" style={{ marginBottom: 48 }}>
          {highlights.map((h, i) => (
            <div key={i} className="card" style={{ borderTop: `3px solid ${h.color}` }}>
              <div style={{ fontSize: "1.8rem", marginBottom: 12 }}>{h.icon}</div>
              <h3 style={{ fontSize: "0.95rem", marginBottom: 10 }}>{h.title}</h3>
              <p style={{ fontSize: "0.87rem", lineHeight: 1.7 }}>{h.desc}</p>
            </div>
          ))}
        </div>

        {/* ASR Comparison Table */}
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: 16,
            overflow: "hidden",
            marginBottom: 32,
          }}
        >
          <div
            style={{
              padding: "20px 28px 16px",
              borderBottom: "1px solid var(--border)",
            }}
          >
            <h3 style={{ fontSize: "1rem" }}>
              Attack Success Rate (ASR %) — PFedBA Attack
            </h3>
            <p style={{ fontSize: "0.82rem", marginTop: 4 }}>
              Lower is better. All defences evaluated against the strongest threat model (PFedBA).
            </p>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "rgba(79,142,247,0.06)" }}>
                  {["Defense", "Type", "IID (α=100)", "Non-IID (α=0.5)", "Non-IID (α=0.1)"].map(
                    (h) => (
                      <th
                        key={h}
                        style={{
                          padding: "12px 20px",
                          textAlign: "left",
                          fontSize: "0.78rem",
                          fontWeight: 700,
                          color: "#4f8ef7",
                          letterSpacing: "0.05em",
                          textTransform: "uppercase",
                          borderBottom: "1px solid var(--border)",
                        }}
                      >
                        {h}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {asrTable.map((row, i) => (
                  <tr
                    key={i}
                    style={{
                      background: row.highlight
                        ? "rgba(79,142,247,0.07)"
                        : i % 2 === 0
                        ? "transparent"
                        : "rgba(255,255,255,0.01)",
                      borderLeft: row.highlight ? "3px solid #4f8ef7" : "3px solid transparent",
                    }}
                  >
                    <td
                      style={{
                        padding: "12px 20px",
                        fontSize: "0.9rem",
                        fontWeight: row.highlight ? 700 : 500,
                        color: row.highlight ? "#4f8ef7" : "#e8edf7",
                        borderBottom: "1px solid rgba(79,142,247,0.06)",
                      }}
                    >
                      {row.defense}
                      {row.highlight && (
                        <span
                          className="badge badge-blue"
                          style={{ marginLeft: 10, fontSize: "0.65rem" }}
                        >
                          Ours
                        </span>
                      )}
                    </td>
                    <td
                      style={{
                        padding: "12px 20px",
                        fontSize: "0.8rem",
                        color: "#5a6a8a",
                        borderBottom: "1px solid rgba(79,142,247,0.06)",
                      }}
                    >
                      {row.type}
                    </td>
                    {[row.iid, row.nonIID05, row.nonIID01].map((val, j) => (
                      <td
                        key={j}
                        style={{
                          padding: "12px 20px",
                          fontSize: "0.9rem",
                          fontWeight: 600,
                          color:
                            row.highlight
                              ? "#22c55e"
                              : val === "High" || parseFloat(val) > 50
                              ? "#ef4444"
                              : parseFloat(val) > 10
                              ? "#f59e0b"
                              : "#22c55e",
                          borderBottom: "1px solid rgba(79,142,247,0.06)",
                        }}
                      >
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <p style={{ fontSize: "0.83rem", color: "#5a6a8a", textAlign: "center" }}>
          * All results on UNSW-NB15 dataset with 10 clients (3 malicious). Lower ASR = stronger defense.
        </p>
      </div>
    </section>
  );
}
