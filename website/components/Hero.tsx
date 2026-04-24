"use client";
import Image from "next/image";
import styles from "./Hero.module.css";

// ─── TEAM & SUPERVISOR DATA ─────────────────────────────────────
// Photos: place images in website/public/team/ with these filenames.
//   Team:        e20055.jpg, e20094.jpg, e20284.jpg
//   Supervisors: supervisor1.jpg, supervisor2.jpg
const BASE = "/e20-4yp-backdoor-resilient-federated-nids";

const teamMembers = [
  {
    id: "E/20/055",
    name: "De Silva H.D.S.",
    email: "e20055@eng.pdn.ac.lk",
    photo: `${BASE}/team/e20055.png`,
  },
  {
    id: "E/20/094",
    name: "Ekanayaka E.M.I.P.",
    email: "e20094@eng.pdn.ac.lk",
    photo: `${BASE}/team/e20094.png`,
  },
  {
    id: "E/20/284",
    name: "Peiris T.M.S.U.",
    email: "e20284@eng.pdn.ac.lk",
    photo: `${BASE}/team/e20284.png`,
  },
];

const supervisors = [
  {
    name: "Dr. Upul Jayasinghe Mendis",
    email: "upuljm@eng.pdn.ac.lk",
    photo: `${BASE}/team/supervisor1.png`,
  },
  {
    name: "Dr. Suneth Namal Karunarathna",
    email: "namal@eng.pdn.ac.lk",
    photo: `${BASE}/team/supervisor2.png`,
  },
];

const tocSections = [
  { num: "01", label: "Abstract", href: "#abstract" },
  { num: "02", label: "Related Works", href: "#related-works" },
  { num: "03", label: "Methodology", href: "#methodology" },
  { num: "04", label: "Experiment Setup", href: "#experiments" },
  { num: "05", label: "Results & Analysis", href: "#results" },
  { num: "06", label: "Conclusion", href: "#conclusion" },
  { num: "07", label: "Team", href: "#team" },
  { num: "08", label: "Links", href: "#links" },
];

// ─── PHOTO AVATAR ────────────────────────────────────────────────
function Avatar({
  photo,
  name,
  size = 40,
  gradient = "linear-gradient(135deg, #4f8ef7 0%, #7c3aed 100%)",
}: {
  photo: string;
  name: string;
  size?: number;
  gradient?: string;
}) {
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        overflow: "hidden",
        flexShrink: 0,
        background: gradient,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: size * 0.38,
        fontWeight: 700,
        color: "#fff",
        border: "2px solid rgba(79,142,247,0.3)",
      }}
    >
      <Image
        src={photo}
        alt={name}
        width={size}
        height={size}
        style={{ objectFit: "cover", width: "100%", height: "100%" }}
        onError={(e) => {
          const t = e.currentTarget as HTMLImageElement;
          t.style.display = "none";
          const parent = t.parentElement;
          if (parent) parent.setAttribute("data-fallback", name[0]);
        }}
      />
    </div>
  );
}

// ─── COMPONENT ────────────────────────────────────────────────────
export default function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.blob1} />
      <div className={styles.blob2} />
      <div className={styles.blob3} />

      <div className={styles.inner}>
        {/* ── Heading ── */}
        <div className={styles.headingArea}>
          <div className={styles.tag}>
            <span className={styles.tagDot} />
            Final Year Research Project · Group 08 · University of Peradeniya
          </div>

          <h1 className={styles.title}>
            Backdoor-Resilient
            <span className={styles.titleGrad}> Federated Learning</span>
            <br />for{" "}
            <span className={styles.titleAccent}>Network Intrusion Detection</span>
          </h1>

          <p className={styles.subtitle}>
            Building a federated NIDS that withstands sophisticated backdoor attacks in
            Non-IID, privacy-constrained network environments — introducing{" "}
            <strong style={{ color: "#e8edf7" }}>SENTINEL</strong>, a novel defense
            combining multi-signal anomaly filtering with coordinate-wise trimmed median
            aggregation.
          </p>

          <div className={styles.badges}>
            <span className="badge badge-blue">🔬 Federated Learning</span>
            <span className="badge badge-purple">🛡️ SENTINEL Defense</span>
            <span className="badge badge-teal">📡 NIDS</span>
            <span className="badge badge-blue">📊 UNSW-NB15</span>
            <span className="badge badge-purple">⚡ Non-IID</span>
          </div>

          <div className={styles.ctas}>
            <a href="#abstract" className="btn btn-primary">
              Read the Research ↓
            </a>
            <a
              href="https://github.com/cepdnaclk/e20-4yp-backdoor-resilient-federated-nids"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-outline"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
              </svg>
              View on GitHub
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
