"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

type PipelineSource = "bigquery" | "salesforce";

type PipelineUserOption = {
  id: string;
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

export default function PipelineTab() {
  const [source, setSource] = useState<PipelineSource>("bigquery");
  const [users, setUsers] = useState<PipelineUserOption[]>([]);
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
        }
      } catch (e) {
        if (!cancelled) {
          setUsers([]);
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

  return (
    <div className="pipeline-tab">
      <p style={{ margin: "0 0 16px", color: "#5a5f6e", lineHeight: 1.5 }}>
        Pick your <strong>Salesforce User Id</strong> from the warehouse list (every distinct{" "}
        <strong>assigned_sa_c</strong> or <strong>owner_id</strong>), then load <strong>open</strong> opportunities
        where that Id is the account&apos;s Assigned SA or the Opportunity owner. Salesforce credentials on the API are
        only used when you choose the live source.
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
          User Id (assigned_sa_c or owner_id)
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
          <option value="">{usersLoading ? "Loading User Ids…" : "Select your User Id…"}</option>
          {users.map((u) => (
            <option key={u.id} value={u.id}>
              {u.id}
            </option>
          ))}
        </select>
        {usersError ? (
          <div role="alert" style={{ fontSize: 13, color: "#842029" }}>
            Could not load User Id list: {usersError}
          </div>
        ) : null}
        {users.length === 0 && !usersLoading && !usersError ? (
          <p style={{ fontSize: 13, color: "#5a5f6e" }}>No User Ids returned from the warehouse.</p>
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
        <p style={{ color: "#5a5f6e" }}>No open opportunities match this person as Assigned SA or Opportunity owner.</p>
      ) : null}

      {data && data.opportunities.length > 0 ? (
        <div style={{ overflowX: "auto" }}>
          <table className="opps-table" style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Account</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Opportunity</th>
                <th style={{ textAlign: "left", padding: "10px 8px", borderBottom: "2px solid #e8e8ef" }}>Opp owner</th>
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
                    {o.owner_name || "—"}
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
            <strong>{data.source}</strong> for user <code style={{ fontSize: 11 }}>{data.user_id}</code>.
          </p>
        </div>
      ) : null}
    </div>
  );
}
