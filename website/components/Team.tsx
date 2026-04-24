"use client";
import Image from "next/image";

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
    role: "Senior Lecturer",
    email: "upuljm@eng.pdn.ac.lk",
    photo: `${BASE}/team/supervisor1.png`,
    gradient: "linear-gradient(135deg, #14b8a6 0%, #4f8ef7 100%)",
  },
  {
    name: "Dr. Suneth Namal Karunarathna",
    role: "Senior Lecturer",
    email: "namal@eng.pdn.ac.lk",
    photo: `${BASE}/team/supervisor2.png`,
    gradient: "linear-gradient(135deg, #14b8a6 0%, #4f8ef7 100%)",
  },
];

function Avatar({
  photo,
  name,
  size = 72,
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
        border: "2.5px solid rgba(79,142,247,0.35)",
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
        }}
      />
    </div>
  );
}

export default function Team() {
  return (
    <section className="section" id="team">
      <div className="container">
        <div className="section-label">07</div>
        <h2 className="section-title">
          <span>Team</span>
        </h2>
        <div className="section-divider" />

        {/* Students */}
        <h3 style={{ marginBottom: 20, color: "#e8edf7" }}>Group 08 — Students</h3>
        <div className="grid-3" style={{ marginBottom: 56 }}>
          {teamMembers.map((m, i) => (
            <div key={i} className="card" style={{ textAlign: "center", alignItems: "center", display: "flex", flexDirection: "column", gap: 16 }}>
              <Avatar photo={m.photo} name={m.name} size={80} />
              <div>
                <div
                  style={{
                    fontSize: "0.72rem",
                    fontWeight: 700,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: "#4f8ef7",
                    marginBottom: 4,
                  }}
                >
                  {m.id}
                </div>
                <div style={{ fontSize: "0.95rem", fontWeight: 600, color: "#e8edf7", marginBottom: 6 }}>
                  {m.name}
                </div>
                <a
                  href={`mailto:${m.email}`}
                  style={{ fontSize: "0.8rem", color: "#5a6a8a", textDecoration: "none" }}
                >
                  {m.email}
                </a>
              </div>
            </div>
          ))}
        </div>

        {/* Supervisors */}
        <h3 style={{ marginBottom: 20, color: "#e8edf7" }}>Supervisors</h3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 24,
          }}
        >
          {supervisors.map((s, i) => (
            <div
              key={i}
              className="card"
              style={{
                display: "flex",
                gap: 20,
                alignItems: "center",
                borderLeft: "3px solid #14b8a6",
              }}
            >
              <Avatar photo={s.photo} name={s.name} size={64} gradient={s.gradient} />
              <div>
                <div
                  style={{
                    fontSize: "0.72rem",
                    fontWeight: 700,
                    letterSpacing: "0.1em",
                    textTransform: "uppercase",
                    color: "#14b8a6",
                    marginBottom: 4,
                  }}
                >
                  Supervisor
                </div>
                <div style={{ fontSize: "0.95rem", fontWeight: 600, color: "#e8edf7", marginBottom: 6 }}>
                  {s.name}
                </div>
                <a
                  href={`mailto:${s.email}`}
                  style={{ fontSize: "0.8rem", color: "#5a6a8a", textDecoration: "none" }}
                >
                  {s.email}
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
