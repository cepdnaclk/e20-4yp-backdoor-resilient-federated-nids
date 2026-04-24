const publications = [
  {
    num: "01",
    type: "Semester 7 Report",
    title: "Backdoor-Resilient Federated Learning for NIDS — Semester 7 Report",
    status: "pending",
  },
  {
    num: "02",
    type: "Semester 7 Slides",
    title: "Presentation Slides — Semester 7",
    status: "pending",
  },
  {
    num: "03",
    type: "Semester 8 Report",
    title: "Backdoor-Resilient Federated Learning for NIDS — Semester 8 Report",
    status: "pending",
  },
  {
    num: "04",
    type: "Semester 8 Slides",
    title: "Presentation Slides — Semester 8",
    status: "pending",
  },
  {
    num: "05",
    type: "Research Paper",
    title: "Research paper title — Author 1, Author 2, Author 3 (2025/2026)",
    status: "pending",
  },
];

export default function Publications() {
  return (
    <section className="section" id="publications">
      <div className="container">
        <div className="section-label">07</div>
        <h2 className="section-title">
          <span>Publications</span>
        </h2>
        <div className="section-divider" />

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {publications.map((p) => (
            <div
              key={p.num}
              className="card"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 20,
                padding: "20px 28px",
              }}
            >
              <span
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 800,
                  color: "#4f8ef7",
                  minWidth: 28,
                  fontFamily: "'Space Grotesk', sans-serif",
                }}
              >
                {p.num}
              </span>
              <div style={{ flex: 1 }}>
                <span
                  className="badge badge-blue"
                  style={{ marginBottom: 6, display: "inline-flex" }}
                >
                  {p.type}
                </span>
                <div style={{ fontSize: "0.9rem", color: "#8b9cbb" }}>{p.title}</div>
              </div>
              <span
                style={{
                  fontSize: "0.72rem",
                  padding: "3px 10px",
                  borderRadius: 100,
                  background: "rgba(90,106,138,0.15)",
                  color: "#5a6a8a",
                  border: "1px solid rgba(90,106,138,0.2)",
                  flexShrink: 0,
                }}
              >
                Upcoming
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
