"use client";

import type { SalesforceAccountData, AccountLast24hActivity } from "@/lib/types";
import { formatCurrency } from "@/lib/helpers";
import { Last24hActivitySection } from "./Last24hActivitySection";

interface Props {
  salesforce: SalesforceAccountData;
  adoptionStatus?: string;
  totalCalls?: number;
  daysInCycle?: number | null;
  platformUsers?: number;
  last24hActivity?: AccountLast24hActivity | null;
}

const ADOPTION_COLORS: Record<string, string> = {
  power_user: "#27ae60",
  active: "#3498db",
  exploring: "#f39c12",
  churning: "#e74c3c",
  not_started: "#95a5a6",
  unknown: "#7f8c8d",
};

export function ProspectSummaryCard({
  salesforce,
  adoptionStatus = "unknown",
  totalCalls = 0,
  daysInCycle,
  platformUsers = 0,
  last24hActivity,
}: Props) {
  const sf = salesforce;
  const adoptionColor = ADOPTION_COLORS[adoptionStatus] || "#7f8c8d";

  return (
    <div className="prospect-summary-card">
      <div className="summary-header">
        <div className="account-info">
          <h2>{sf.name}</h2>
          <div className="account-badges">
            {sf.lifecycle_stage && (
              <span className="badge badge-purple">{sf.lifecycle_stage}</span>
            )}
            {sf.industry && <span className="text-muted">{sf.industry}</span>}
            {sf.website && (
              <a
                href={`https://${sf.website.replace(/^https?:\/\//, "")}`}
                target="_blank"
                rel="noopener noreferrer"
                className="website-link"
              >
                {sf.website.replace(/^https?:\/\//, "")}
              </a>
            )}
          </div>
        </div>
        <div className="arr-display">
          <div className="arr-value">{formatCurrency(sf.total_active_arr)}</div>
          <div className="arr-label">Active ARR</div>
        </div>
      </div>

      {last24hActivity && <Last24hActivitySection activity={last24hActivity} />}

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{totalCalls}</div>
          <div className="metric-label">Sales Calls</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{daysInCycle ?? "N/A"}</div>
          <div className="metric-label">Days in Cycle</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{platformUsers}</div>
          <div className="metric-label">Platform Users</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ color: adoptionColor }}>
            {adoptionStatus.replace("_", " ").toUpperCase()}
          </div>
          <div className="metric-label">Adoption Status</div>
        </div>
      </div>

      {(sf.assigned_sa || sf.assigned_ai_se || sf.assigned_cse || sf.assigned_csm || sf.owner_name) && (
        <div className="team-section">
          <span className="team-label">Team:</span>
          {sf.owner_name && <span className="team-badge owner">Owner: {sf.owner_name}</span>}
          {sf.assigned_sa && <span className="team-badge sa">SA: {sf.assigned_sa}</span>}
          {sf.assigned_ai_se && <span className="team-badge ai-se">AI SE: {sf.assigned_ai_se}</span>}
          {sf.assigned_cse && <span className="team-badge cse">CSE: {sf.assigned_cse}</span>}
          {sf.assigned_csm && <span className="team-badge csm">CSM: {sf.assigned_csm}</span>}
        </div>
      )}

      <style jsx>{`
        .prospect-summary-card {
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          border-radius: 16px;
          padding: 25px;
          margin-bottom: 30px;
          color: white;
        }
        .summary-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          flex-wrap: wrap;
          gap: 20px;
        }
        .account-info h2 {
          margin: 0 0 8px 0;
          font-size: 1.8em;
        }
        .account-badges {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
          align-items: center;
        }
        .badge {
          padding: 4px 12px;
          border-radius: 20px;
          font-size: 0.85em;
        }
        .badge-purple {
          background: #667eea;
        }
        .text-muted {
          color: #a0a0a0;
        }
        .website-link {
          color: #4facfe;
          text-decoration: none;
        }
        .arr-display {
          text-align: right;
        }
        .arr-value {
          font-size: 2em;
          font-weight: bold;
          color: #27ae60;
        }
        .arr-label {
          font-size: 0.85em;
          color: #a0a0a0;
        }
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 15px;
          margin-top: 25px;
        }
        .metric-card {
          text-align: center;
          padding: 15px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 12px;
        }
        .metric-value {
          font-size: 1.8em;
          font-weight: bold;
        }
        .metric-label {
          font-size: 0.85em;
          color: #a0a0a0;
        }
        .team-section {
          margin-top: 20px;
          padding-top: 15px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          align-items: center;
        }
        .team-label {
          color: #a0a0a0;
          font-size: 0.85em;
        }
        .team-badge {
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 0.85em;
        }
        .team-badge.owner { background: #0176d3; }
        .team-badge.sa { background: #667eea; }
        .team-badge.ai-se { background: #9b59b6; }
        .team-badge.cse { background: #11998e; }
        .team-badge.csm { background: #4facfe; }
      `}</style>
    </div>
  );
}
