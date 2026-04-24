import Image from "next/image";

const BASE = "/e20-4yp-backdoor-resilient-federated-nids";

const attackPhases = [
  {
    num: "01",
    tag: "Brute-Force",
    title: "Model Poisoning & Replacement",
    desc: "Train a backdoored model locally, compute the model update, and scale it with a large factor (λ) to replace the global model in a single round. Easily detected by large update norms.",
    color: "#ef4444",
    icon: "💥",
  },
  {
    num: "02",
    tag: "Phase 2 — Stealth",
    title: "Geometric Camouflage",
    desc: "\"Ninja logic\" — the attacker measures the magnitude of honest updates and caps the malicious update norm at the median of honest clients using L2 projection. Bypasses norm-based filters while maintaining high ASR.",
    color: "#f59e0b",
    icon: "🥷",
  },
  {
    num: "03",
    tag: "Phase 3 — PFedBA",
    title: "Proxy Federated Backdoor Attack",
    desc: "The most sophisticated threat model. Uses Shadow Training to simulate clean update directions, then applies a Gradient Alignment Penalty in the loss function — forcing malicious updates to mimic benign ones. Bypasses FLAME (SOTA).",
    color: "#7c3aed",
    icon: "👾",
  },
];

const sentinelSteps = [
  {
    step: "1",
    title: "Extract Client Deltas",
    desc: "Compute per-client model weight deltas (update − global model).",
    color: "#4f8ef7",
  },
  {
    step: "2",
    title: "Compute Anomaly Signals",
    desc: "Calculate L2 norm anomaly score and Sybil similarity score (cosine similarity between clients). Normalize using Median Absolute Deviation (MAD).",
    color: "#4f8ef7",
  },
  {
    step: "3",
    title: "Fuse & Filter",
    desc: "Combine signals into a unified anomaly score. Sort clients, reject outliers using IQR threshold. Ensure minimum benign clients remain.",
    color: "#4f8ef7",
  },
  {
    step: "4",
    title: "Trimmed Median Aggregation",
    desc: "Stack deltas per coordinate, sort and trim top/bottom f values, compute coordinate-wise trimmed median, update global model.",
    color: "#4f8ef7",
  },
  {
    step: "5",
    title: "Differential Privacy (Optional)",
    desc: "Add calibrated Gaussian noise to the aggregated update to provide formal privacy guarantees.",
    color: "#14b8a6",
  },
];

export default function Methodology() {
  return (
    <section className="section" id="methodology">
      <div className="container">
        <div className="section-label">03</div>
        <h2 className="section-title">
          <span>Methodology</span>
        </h2>
        <div className="section-divider" />

        {/* SENTINEL — Our Proposed Defense */}
        <div
          style={{
            background: "linear-gradient(135deg, rgba(79,142,247,0.06) 0%, rgba(124,58,237,0.06) 100%)",
            border: "1px solid rgba(79,142,247,0.18)",
            borderRadius: 20,
            padding: "40px 36px",
            marginBottom: 56,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 8 }}>
            <span style={{ fontSize: "1.8rem" }}>🛡️</span>
            <div>
              <div style={{ fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase", color: "#4f8ef7", marginBottom: 4 }}>
                Our Proposed Defense
              </div>
              <h3 style={{ fontSize: "1.4rem", color: "#e8edf7" }}>SENTINEL</h3>
            </div>
          </div>
          <p style={{ fontSize: "0.9rem", marginBottom: 28, maxWidth: 700 }}>
            A backdoor-resilient federated aggregation algorithm that fuses multiple
            anomaly detection signals to identify and filter malicious clients before
            performing robust coordinate-wise trimmed median aggregation.
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {sentinelSteps.map((s, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: 16,
                  alignItems: "flex-start",
                  paddingBottom: i < sentinelSteps.length - 1 ? 20 : 0,
                  position: "relative",
                }}
              >
                {i < sentinelSteps.length - 1 && (
                  <div
                    style={{
                      position: "absolute",
                      left: 15,
                      top: 32,
                      bottom: 0,
                      width: 2,
                      background: "rgba(79,142,247,0.2)",
                    }}
                  />
                )}
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    background: `${s.color}20`,
                    border: `1.5px solid ${s.color}50`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "0.78rem",
                    fontWeight: 700,
                    color: s.color,
                    flexShrink: 0,
                    zIndex: 1,
                  }}
                >
                  {s.step}
                </div>
                <div style={{ paddingTop: 4 }}>
                  <div
                    style={{
                      fontSize: "0.9rem",
                      fontWeight: 600,
                      color: "#e8edf7",
                      marginBottom: 4,
                    }}
                  >
                    {s.title}
                  </div>
                  <p style={{ fontSize: "0.85rem", lineHeight: 1.65 }}>{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Flowchart */}
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: 16,
            overflow: "hidden",
            marginBottom: 56,
          }}
        >
          <div
            style={{
              padding: "14px 24px",
              borderBottom: "1px solid var(--border)",
              fontSize: "0.75rem",
              fontWeight: 700,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "#4f8ef7",
            }}
          >
            System Overview — Methodology Flowchart
          </div>
          <div style={{ position: "relative", width: "100%", lineHeight: 0 }}>
            <Image
              src={`${BASE}/team/methodology.png`}
              alt="Methodology Flowchart"
              width={1200}
              height={700}
              style={{
                width: "100%",
                height: "auto",
                display: "block",
              }}
            />
          </div>
        </div>


        {/* Attack Evolution */}

        <h3 style={{ marginBottom: 8, color: "#e8edf7" }}>Attack Evolution</h3>
        <p style={{ marginBottom: 28, fontSize: "0.9rem" }}>
          We study three generations of backdoor attacks — each designed to evade the
          defences that defeated the previous generation.
        </p>

        <div className="grid-3" style={{ marginBottom: 56 }}>
          {attackPhases.map((p, i) => (
            <div
              key={i}
              className="card"
              style={{ borderTop: `3px solid ${p.color}` }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  marginBottom: 12,
                }}
              >
                <span
                  className="badge"
                  style={{
                    background: `${p.color}18`,
                    color: p.color,
                    border: `1px solid ${p.color}30`,
                    fontSize: "0.72rem",
                  }}
                >
                  {p.tag}
                </span>
                <span style={{ fontSize: "1.5rem" }}>{p.icon}</span>
              </div>
              <h3 style={{ fontSize: "0.95rem", marginBottom: 10 }}>{p.title}</h3>
              <p style={{ fontSize: "0.85rem", lineHeight: 1.7 }}>{p.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
