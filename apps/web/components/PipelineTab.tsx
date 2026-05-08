"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

type PipelineSource = "bigquery" | "salesforce";

type MyPipelineOpportunity = {
  id: string;
  name: string;
  stage_name?: string | null;
  amount?: number | null;
  close_date?: string | null;
  next_step?: string | null;
  account_id?: string | null;
  account_name?: string | null;
};

type MyOpportunitiesResponse = {
  sa_user_id: string;
  source: string;
  opportunities: MyPipelineOpportunity[];
  notes?: string[];
};

const LIGHTNING_ORIGIN =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_SALESFORCE_LIGHTNING_ORIGIN) ||
  "https://arize.lightning.force.com";

function oppUrl(id: string): string {
  return `${LIGHTNING_ORIGIN.replace(/\/$/, "")}/lightning/r/Opportunity/${id}/view`;
}

function formatMoney(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);
}

function formatDate(s: string | null | undefined): string {
  if (!s) return "—";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleDateString();
}

export default function PipelineTab() {
  const [source, setSource] = useState<PipelineSource>("bigquery");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<MyOpportunitiesResponse | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await apiGet<MyOpportunitiesResponse>("/api/my-opportunities", {
        searchParams: { source },
      });
      setData(r);
    } catch (e) {
      setData(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [source]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="pipeline-tab">
      <p style={{ margin: "0 0 16px", color: "#5a5f6e", lineHeight: 1.5 }}>
        Open opportunities on accounts where you are the assigned Solution Architect. Data loads fresh each time you open
        this tab or switch source.
      </p>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center", marginBottom: 16 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: "#1a1d29" }}>Source</span>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 14 }}>
          <input type="radio" name="pipe-src" checked={source === "bigquery"} onChange={() => setSource("bigquery")} />
          BigQuery (warehouse)
        </label>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 14 }}>
          <input type="radio" name="pipe-src" checked={source === "salesforce"} onChange={() => setSource("salesforce")} />
          Salesforce (live)
        </label>
        <button type="button" className="btn-secondary" onClick={() => void load()} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {error ? (
        <div
          role="alert"
          style={{
            padding: 12,
            borderRadius: 8,
            background: "#fff0f0",
            border: "1px solid #f5c2c7",
            color: "#842029",
            marginBottom: 12,
          }}
        >
          {error}
        </div>
      ) : null}

      {data?.notes?.length ? (
        <ul style={{ fontSize: 13, color: "#5a5f6e", margin: "0 0 12px 1.1rem" }}>
          {data.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      ) : null}

      {!loading && data && !data.opportunities.length ? (
        <p style={{ color: "#5a5f6e" }}>No open opportunities found for your assigned accounts.</p>
      ) : null}

      {data && data.opportunities.length > 0 ? (
        <div style={{ overflowX: "auto" }}>
          <table className="opps-table" style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Account</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Opportunity</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Stage</th>
                <th style={{ textAlign: "right", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Amount</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Close</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Next step</th>
              </tr>
            </thead>
            <tbody>
              {data.opportunities.map((o) => (
                <tr key={o.id}>
                  <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", verticalAlign: "top" }}>
                    {o.account_name || "—"}
                  </td>
                  <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", verticalAlign: "top" }}>
                    <a href={oppUrl(o.id)} target="_blank" rel="noopener noreferrer" style={{ fontWeight: 600 }}>
                      {o.name}
                    </a>
                  </td>
                  <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", verticalAlign: "top" }}>
                    {o.stage_name || "—"}
                  </td>
                  <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", textAlign: "right", verticalAlign: "top" }}>
                    {formatMoney(o.amount ?? undefined)}
                  </td>
                  <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", verticalAlign: "top" }}>
                    {formatDate(o.close_date ?? undefined)}
                  </td>
                  <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", color: "#444", verticalAlign: "top" }}>
                    {o.next_step || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: 12, color: "#888", marginTop: 10 }}>
            Showing {data.opportunities.length} row{data.opportunities.length === 1 ? "" : "s"} via{" "}
            <strong>{data.source}</strong> (User <code style={{ fontSize: 11 }}>{data.sa_user_id}</code>).
          </p>
        </div>
      ) : null}
    </div>
  );
}
