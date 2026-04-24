const links = [
  {
    icon: "📁",
    title: "Project Repository",
    desc: "Full source code, notebooks, and experiment scripts.",
    href: "https://github.com/cepdnaclk/e20-4yp-backdoor-resilient-federated-nids",
    label: "GitHub",
    color: "#4f8ef7",
  },
  {
    icon: "🌐",
    title: "Project Page",
    desc: "This GitHub Pages site for the research project.",
    href: "https://cepdnaclk.github.io/e20-4yp-backdoor-resilient-federated-nids",
    label: "Visit",
    color: "#7c3aed",
  },
  {
    icon: "🏫",
    title: "Department of Computer Engineering",
    desc: "Faculty of Engineering, University of Peradeniya.",
    href: "http://www.ce.pdn.ac.lk/",
    label: "Website",
    color: "#14b8a6",
  },
  {
    icon: "🎓",
    title: "University of Peradeniya",
    desc: "Faculty of Engineering, Peradeniya, Sri Lanka.",
    href: "https://eng.pdn.ac.lk/",
    label: "Website",
    color: "#f59e0b",
  },
];

export default function Links() {
  return (
    <section className="section" id="links">
      <div className="container">
        <div className="section-label">08</div>
        <h2 className="section-title">
          <span>Links</span>
        </h2>
        <div className="section-divider" />

        <div className="grid-2">
          {links.map((l, i) => (
            <a
              key={i}
              href={l.href}
              target="_blank"
              rel="noopener noreferrer"
              className="card"
              style={{
                display: "flex",
                gap: 18,
                alignItems: "flex-start",
                textDecoration: "none",
                cursor: "pointer",
              }}
            >
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: 12,
                  background: `${l.color}18`,
                  border: `1.5px solid ${l.color}30`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "1.4rem",
                  flexShrink: 0,
                }}
              >
                {l.icon}
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: "0.95rem", marginBottom: 6 }}>{l.title}</h3>
                <p style={{ fontSize: "0.85rem", marginBottom: 12, lineHeight: 1.5 }}>
                  {l.desc}
                </p>
                <span
                  style={{
                    fontSize: "0.8rem",
                    fontWeight: 600,
                    color: l.color,
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                  }}
                >
                  {l.label} →
                </span>
              </div>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
