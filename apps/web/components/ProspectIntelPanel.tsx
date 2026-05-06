"use client";

type CompetitiveRow = {
  competitor?: string | null;
  what_they_said?: string | null;
  targeted_response?: string | null;
  mentioned_in?: string[] | null;
  note?: string | null;
};

type Props = {
  dealHealth?: {
    score: number;
    band: string;
    factors?: { signal?: string; detail?: string; impact?: string }[];
  };
  competitiveMentions?: CompetitiveRow[];
  meetingPrep?: { bullets?: string[] };
};

export default function ProspectIntelPanel({ dealHealth, competitiveMentions, meetingPrep }: Props) {
  const band = dealHealth?.band ?? "steady";
  const score = dealHealth?.score ?? 0;
  const barColor =
    band === "strong" ? "#27ae60" : band === "at_risk" ? "#e74c3c" : "#3498db";

  return (
    <div className="prospect-intel-panel" style={{ marginTop: 20 }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 16 }}>
        <div
          style={{
            flex: "1 1 220px",
            padding: 14,
            borderRadius: 12,
            background: "#f8f9ff",
            border: "1px solid #dfe3fb",
          }}
        >
          <div style={{ fontSize: 12, textTransform: "uppercase", color: "#667eea", fontWeight: 600 }}>
            Deal health score
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 6 }}>
            <span style={{ fontSize: 36, fontWeight: 700, color: barColor }}>{score}</span>
            <span style={{ fontSize: 14, color: "#555", textTransform: "capitalize" }}>{band.replace("_", " ")}</span>
          </div>
          <div style={{ height: 8, background: "#e8eaf6", borderRadius: 4, marginTop: 8, overflow: "hidden" }}>
            <div style={{ width: `${score}%`, height: "100%", background: barColor, borderRadius: 4 }} />
          </div>
          {dealHealth?.factors?.length ? (
            <ul style={{ margin: "10px 0 0 0", paddingLeft: 18, fontSize: 12, color: "#555" }}>
              {dealHealth.factors.slice(0, 6).map((f, i) => (
                <li key={i} style={{ marginBottom: 4 }}>
                  <strong>{f.signal}</strong>: {f.detail} {f.impact ? `(${f.impact})` : ""}
                </li>
              ))}
            </ul>
          ) : null}
        </div>

        <div
          style={{
            flex: "1 1 280px",
            padding: 14,
            borderRadius: 12,
            background: "#fff8e6",
            border: "1px solid #ffe0a3",
          }}
        >
          <div style={{ fontSize: 12, textTransform: "uppercase", color: "#b8860b", fontWeight: 600 }}>
            Meeting prep mode
          </div>
          <ul style={{ margin: "8px 0 0 0", paddingLeft: 18, fontSize: 14, color: "#333", lineHeight: 1.5 }}>
            {(meetingPrep?.bullets ?? []).map((b, i) => (
              <li key={i} style={{ marginBottom: 6 }}>
                {b}
              </li>
            ))}
            {!meetingPrep?.bullets?.length ? (
              <li style={{ color: "#888" }}>Run a full prospect overview to populate talking points.</li>
            ) : null}
          </ul>
        </div>
      </div>

      <div
        style={{
          padding: 14,
          borderRadius: 12,
          background: "white",
          border: "1px solid #e9ecef",
        }}
      >
        <div style={{ fontSize: 12, textTransform: "uppercase", color: "#c0392b", fontWeight: 600, marginBottom: 8 }}>
          Competitive mentions dashboard
        </div>
        {!competitiveMentions?.length ? (
          <p style={{ color: "#777", margin: 0, fontSize: 14 }}>
            No competitive messaging captured in recommendations for this account yet.
          </p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "2px solid #eee" }}>
                  <th style={{ padding: "8px 6px" }}>Competitor</th>
                  <th style={{ padding: "8px 6px" }}>What they said</th>
                  <th style={{ padding: "8px 6px" }}>Suggested response</th>
                </tr>
              </thead>
              <tbody>
                {competitiveMentions.map((row, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #f0f0f0", verticalAlign: "top" }}>
                    <td style={{ padding: "8px 6px", fontWeight: 600 }}>{row.competitor}</td>
                    <td style={{ padding: "8px 6px", color: "#444" }}>{row.what_they_said ?? "—"}</td>
                    <td style={{ padding: "8px 6px", color: "#155724" }}>{row.targeted_response ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
