"use client";

import type { AccountSuggestionMatch } from "@/lib/accountResolve";

type Props = {
  open: boolean;
  reason: string;
  typedQuery: string;
  matches: AccountSuggestionMatch[];
  onPick: (choice: "cancel" | "keep" | AccountSuggestionMatch) => void;
};

export default function AccountSuggestModal({
  open,
  reason,
  typedQuery,
  matches,
  onPick,
}: Props) {
  if (!open) return null;

  return (
    <div
      className="account-suggest-overlay"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.45)",
        zIndex: 10000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 16,
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="account-suggest-title"
    >
      <div
        className="card"
        style={{
          maxWidth: 560,
          width: "100%",
          maxHeight: "90vh",
          overflow: "auto",
          boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 id="account-suggest-title" style={{ marginTop: 0 }}>
          Confirm Salesforce account
        </h3>
        <p style={{ color: "#555", fontSize: 15 }}>{reason}</p>
        <p style={{ marginBottom: 16 }}>
          <span style={{ color: "#666" }}>You typed: </span>
          <strong>{typedQuery}</strong>
        </p>
        <p className="help-text" style={{ marginBottom: 12 }}>
          Choose the CRM record that best matches this company, keep your spelling, or cancel.
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
          {matches.map((m) => (
            <button
              key={m.id}
              type="button"
              className="btn-primary"
              style={{ textAlign: "left" }}
              onClick={() => onPick(m)}
            >
              {m.name}
              {m.website ? (
                <span style={{ display: "block", fontSize: 12, fontWeight: 400, opacity: 0.9 }}>
                  {m.website}
                </span>
              ) : null}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button type="button" className="btn-secondary" onClick={() => onPick("keep")}>
            Keep my spelling
          </button>
          <button
            type="button"
            style={{
              padding: "10px 18px",
              border: "2px solid #e0e0e0",
              borderRadius: 8,
              background: "white",
              cursor: "pointer",
            }}
            onClick={() => onPick("cancel")}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
