const relatedWorks = [
  {
    icon: "🌐",
    title: "Federated Learning for NIDS",
    description:
      "Federated Learning enables collaborative model training across distributed organisations without sharing raw network traffic data, preserving privacy while building powerful intrusion detection models. FL requires continuous training with distributed data from heterogeneous network environments.",
    color: "#4f8ef7",
  },
  {
    icon: "☠️",
    title: "Backdoor Attacks in FL",
    description:
      "In a backdoor attack, malicious clients inject poisoned model updates into the federated aggregation process. A carefully crafted \"trigger\" in network traffic causes the global model to misclassify targeted malicious traffic as benign — while maintaining normal accuracy on clean samples. \"Trust becomes the vulnerability.\"",
    color: "#ef4444",
  },
  {
    icon: "🧠",
    title: "Byzantine-Robust Aggregation",
    description:
      "Existing defences include FedAvg (no defence), Median, Trimmed Mean, Krum, Multi-Krum, and FLAME. Our evaluation shows all existing SOTA methods fail against the sophisticated PFedBA attacker — with FLAME (considered SOTA) reaching up to 100% ASR in Non-IID settings.",
    color: "#7c3aed",
  },
  {
    icon: "📡",
    title: "Non-IID Network Environments",
    description:
      "Real-world federated NIDS deployments face heterogeneous (Non-IID) data distributions across clients. We model this using Dirichlet distributions (α = 100 for IID, α = 0.5 and α = 0.1 for increasing heterogeneity), reflecting realistic scenarios where different organisations see different traffic patterns.",
    color: "#14b8a6",
  },
];

export default function RelatedWorks() {
  return (
    <section className="section" id="related-works">
      <div className="container">
        <div className="section-label">02</div>
        <h2 className="section-title">
          Related <span>Works</span>
        </h2>
        <div className="section-divider" />

        <div className="grid-2">
          {relatedWorks.map((w, i) => (
            <div key={i} className="card">
              <div
                style={{
                  width: 44,
                  height: 44,
                  borderRadius: 10,
                  background: `${w.color}18`,
                  border: `1px solid ${w.color}25`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "1.4rem",
                  marginBottom: 16,
                }}
              >
                {w.icon}
              </div>
              <h3 style={{ marginBottom: 12 }}>{w.title}</h3>
              <p style={{ fontSize: "0.9rem", lineHeight: 1.75 }}>{w.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
