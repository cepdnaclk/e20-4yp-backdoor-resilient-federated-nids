export default function Footer() {
  return (
    <footer
      style={{
        background: "#080c18",
        borderTop: "1px solid rgba(79,142,247,0.1)",
        padding: "48px 24px",
        textAlign: "center",
      }}
    >
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <p
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontWeight: 600,
            fontSize: "1rem",
            color: "#e8edf7",
            marginBottom: 8,
          }}
        >
          Backdoor-Resilient Federated Learning for NIDS
        </p>
        <p style={{ fontSize: "0.85rem", color: "#5a6a8a", marginBottom: 20 }}>
          Department of Computer Engineering &nbsp;·&nbsp; University of Peradeniya
        </p>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 24,
            flexWrap: "wrap",
          }}
        >
          {[
            {
              label: "GitHub Repo",
              href: "https://github.com/cepdnaclk/e20-4yp-backdoor-resilient-federated-nids",
            },
            {
              label: "Department of CE",
              href: "http://www.ce.pdn.ac.lk/",
            },
            {
              label: "University of Peradeniya",
              href: "https://eng.pdn.ac.lk/",
            },
          ].map((l) => (
            <a
              key={l.href}
              href={l.href}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: "0.83rem",
                color: "#4f8ef7",
                transition: "color 0.2s",
              }}
            >
              {l.label}
            </a>
          ))}
        </div>
        <p
          style={{
            fontSize: "0.78rem",
            color: "#3a4a6a",
            marginTop: 28,
          }}
        >
          © {new Date().getFullYear()} — e20-4yp · Final Year Research Project
        </p>
      </div>
    </footer>
  );
}
