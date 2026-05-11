"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { apiGet } from "@/lib/api";

type PipelineSource = "bigquery" | "salesforce";

type PipelineUserOption = {
  id: string;
  name: string;
};

type MyPipelineOpportunity = {
  id: string;
  name: string;
  stage_name?: string | null;
  amount?: number | null;
  close_date?: string | null;
  next_step?: string | null;
  account_id?: string | null;
  account_name?: string | null;
  owner_name?: string | null;
  days_in_stage?: number | null;
};

type MyOpportunitiesResponse = {
  user_id: string;
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

/** Stable label for grouping; empty → "— (no stage)" */
function stageGroupLabel(stage: string | null | undefined): string {
  const t = (stage ?? "").trim();
  return t || "— (no stage)";
}

/**
 * Opportunities grouped by stage, stages sorted alphabetically with numeric-aware ordering
 * (e.g. "1. Foo" before "10. Bar").
 */
function groupOppsByStage(opps: MyPipelineOpportunity[]): Map<string, MyPipelineOpportunity[]> {
  const map = new Map<string, MyPipelineOpportunity[]>();
  for (const o of opps) {
    const key = stageGroupLabel(o.stage_name);
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(o);
  }
  for (const list of map.values()) {
    list.sort((a, b) => (a.account_name ?? "").localeCompare(b.account_name ?? "", undefined, { sensitivity: "base" }));
  }
  return new Map([...map.entries()].sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true })));
}

export default function PipelineTab() {
  const [source, setSource] = useState<PipelineSource>("bigquery");
  const [users, setUsers] = useState<PipelineUserOption[]>([]);
  const [pipelineUserNotes, setPipelineUserNotes] = useState<string[]>([]);
  const [usersLoading, setUsersLoading] = useState(true);
  const [usersError, setUsersError] = useState<string | null>(null);
  const [selectedUserId, setSelectedUserId] = useState("");

  const [oppsLoading, setOppsLoading] = useState(false);
  const [oppsError, setOppsError] = useState<string | null>(null);
  const [data, setData] = useState<MyOpportunitiesResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setUsersLoading(true);
      setUsersError(null);
      try {
        const r = await apiGet<{ users: PipelineUserOption[]; notes?: string[] }>("/api/pipeline-user-options");
        if (!cancelled) {
          setUsers(r.users || []);
          setPipelineUserNotes(Array.isArray(r.notes) ? r.notes : []);
        }
      } catch (e) {
        if (!cancelled) {
          setUsers([]);
          setPipelineUserNotes([]);
          setUsersError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled) setUsersLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadOpportunities = useCallback(async () => {
    const uid = selectedUserId.trim();
    if (!uid) {
      setData(null);
      setOppsError(null);
      return;
    }
    setOppsLoading(true);
    setOppsError(null);
    try {
      const r = await apiGet<MyOpportunitiesResponse>("/api/my-opportunities", {
        searchParams: { user_id: uid, source },
      });
      setData(r);
    } catch (e) {
      setData(null);
      setOppsError(e instanceof Error ? e.message : String(e));
    } finally {
      setOppsLoading(false);
    }
  }, [selectedUserId, source]);

  useEffect(() => {
    void loadOpportunities();
  }, [loadOpportunities]);

  const oppsByStage = useMemo(
    () => (data?.opportunities?.length ? groupOppsByStage(data.opportunities) : new Map<string, MyPipelineOpportunity[]>()),
    [data],
  );

  return (
    <div className="pipeline-tab">
      <p style={{ margin: "0 0 16px", color: "#5a5f6e", lineHeight: 1.5 }}>
        Select your name, then load <strong>open</strong> opportunities where you&apos;re on the deal — for example as{" "}
        <strong>Opportunity owner</strong> (typical for AEs) or as <strong>Assigned SA</strong> on the account or
        opportunity (typical for SAs).
      </p>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 12,
          marginBottom: 20,
          maxWidth: "min(100%, 40rem)",
        }}
      >
        <label htmlFor="pipeline-user-select" style={{ fontSize: 14, fontWeight: 600, color: "#1a1d29" }}>
          User
        </label>
        <select
          id="pipeline-user-select"
          className="pipeline-user-select"
          value={selectedUserId}
          onChange={(e) => setSelectedUserId(e.target.value)}
          disabled={usersLoading}
          style={{
            fontSize: 15,
            padding: "10px 12px",
            borderRadius: 10,
            border: "1px solid var(--arize-border, #e0e0e8)",
            background: "var(--arize-surface, #fff)",
            color: "#1a1d29",
          }}
        >
          <option value="">{usersLoading ? "Loading users…" : "Select your name…"}</option>
          {users.map((u) => (
            <option key={u.id} value={u.id}>
              {u.name}
            </option>
          ))}
        </select>
        {usersError ? (
          <div role="alert" style={{ fontSize: 13, color: "#842029" }}>
            Could not load user list: {usersError}
          </div>
        ) : null}
        {users.length === 0 && !usersLoading && !usersError ? (
          <div style={{ fontSize: 13, color: "#5a5f6e" }}>
            <p style={{ margin: "0 0 8px" }}>
              No users returned. This app proxies <code>/api/*</code> to the FastAPI deployment (e.g.{" "}
              <code>arize-gtm-stillness-api.vercel.app</code>). If that API project uses Vercel Deployment Protection,
              set <code>FASTAPI_VERCEL_PROTECTION_BYPASS</code> on <strong>this</strong> (Next.js) project to the same
              Automation bypass secret configured on the API project (see <code>.env.example</code>).
            </p>
            <p style={{ margin: "0 0 8px" }}>
              Otherwise configure BigQuery or Salesforce env vars <strong>on the API project</strong>, not only on the
              frontend: <code>GCP_CREDENTIALS_BASE64</code> (and usually <code>GOOGLE_CLOUD_PROJECT</code>), or{" "}
              <code>SALESFORCE_USERNAME</code>, <code>SALESFORCE_PASSWORD</code>, <code>SALESFORCE_SECURITY_TOKEN</code>.
            </p>
            {pipelineUserNotes.length > 0 ? (
              <ul style={{ margin: "8px 0 0", paddingLeft: "1.25rem", lineHeight: 1.5 }}>
                {pipelineUserNotes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}
      </div>

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
        <button
          type="button"
          className="btn-secondary"
          onClick={() => void loadOpportunities()}
          disabled={oppsLoading || !selectedUserId.trim()}
        >
          {oppsLoading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {oppsError ? (
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
          {oppsError}
        </div>
      ) : null}

      {data?.notes?.length ? (
        <ul style={{ fontSize: 13, color: "#5a5f6e", margin: "0 0 12px 1.1rem" }}>
          {data.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      ) : null}

      {selectedUserId && !oppsLoading && data && !data.opportunities.length ? (
        <p style={{ color: "#5a5f6e" }}>
          No open opportunities match this person as Opportunity owner or as Assigned SA on the account or opportunity.
        </p>
      ) : null}

      {data && data.opportunities.length > 0 ? (
        <div style={{ overflowX: "auto" }}>
          <table className="opps-table" style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr>
                <th
                  style={{
                    textAlign: "left",
                    padding: "10px 8px",
                    borderBottom: "2px solid #e8e8ef",
                    width: "11rem",
                    minWidth: "9rem",
                  }}
                >
                  Stage
                </th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Account</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Opportunity</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Opp owner</th>
                <th style={{ textAlign: "right", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Amount</th>
                <th
                  title="Days since the opportunity last changed stage"
                  style={{ textAlign: "right", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}
                >
                  Days in stage
                </th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Close</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Next step</th>
              </tr>
            </thead>
            {[...oppsByStage.entries()].map(([stageLabel, rows]) => (
              <tbody key={stageLabel}>
                {rows.map((o, idx) => (
                  <tr key={o.id}>
                    {idx === 0 ? (
                      <th
                        rowSpan={rows.length}
                        scope="rowgroup"
                        style={{
                          textAlign: "left",
                          padding: "10px 8px",
                          borderBottom: "1px solid #e8e8ef",
                          borderRight: "1px solid #e8e8ef",
                          verticalAlign: "top",
                          background: "var(--arize-surface-alt, #faf8fb)",
                          fontWeight: 600,
                          color: "#1a1d29",
                          whiteSpace: "normal",
                          maxWidth: "14rem",
                        }}
                      >
                        {stageLabel}
                        <span
                          style={{
                            display: "block",
                            fontSize: 11,
                            fontWeight: 600,
                            color: "#888",
                            marginTop: 4,
                          }}
                        >
                          {rows.length} opp{rows.length === 1 ? "" : "s"}
                        </span>
                      </th>
                    ) : null}
                    <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", verticalAlign: "top" }}>
                      {o.account_name || "—"}
                    </td>
                    <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", verticalAlign: "top" }}>
                      <a href={oppUrl(o.id)} target="_blank" rel="noopener noreferrer" style={{ fontWeight: 600 }}>
                        {o.name}
                      </a>
                    </td>
                    <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", verticalAlign: "top" }}>
                      {o.owner_name || "—"}
                    </td>
                    <td style={{ padding: "10px 8px", borderBottom: "1px solid #f0f0f5", textAlign: "right", verticalAlign: "top" }}>
                      {formatMoney(o.amount ?? undefined)}
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #f0f0f5",
                        textAlign: "right",
                        verticalAlign: "top",
                        color: (o.days_in_stage ?? 0) >= 60 ? "#b1241f" : "#1a1d29",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {o.days_in_stage == null ? "—" : o.days_in_stage}
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
            ))}
          </table>
          <p style={{ fontSize: 12, color: "#888", marginTop: 10 }}>
            Showing {data.opportunities.length} row{data.opportunities.length === 1 ? "" : "s"} via{" "}
            <strong>{data.source}</strong> for user <code style={{ fontSize: 11 }}>{data.user_id}</code>.
          </p>
        </div>
      ) : null}
    </div>
  );
}
