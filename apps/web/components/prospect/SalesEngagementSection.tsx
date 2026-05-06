"use client";

import type {
  OpportunityData,
  GongSummaryData,
  DealSummary,
} from "@/lib/types";
import { formatCurrency, formatDate, formatPercent } from "@/lib/helpers";

interface Props {
  latestOpportunity?: OpportunityData | null;
  allOpportunities: OpportunityData[];
  gongSummary?: GongSummaryData | null;
  dealSummary?: DealSummary | null;
}

function OpportunityCard({ opp }: { opp: OpportunityData }) {
  return (
    <div className="opportunity-card">
      <div className="opp-header">
        <div className="opp-info">
          <div className="opp-label">CURRENT OPPORTUNITY</div>
          <h4>{opp.name}</h4>
          <div className="opp-meta">
            <span className="stage-badge">{opp.stage_name}</span>
            <span className="meta-text">Close: {formatDate(opp.close_date)}</span>
            {opp.owner_name && <span className="meta-text">Owner: {opp.owner_name}</span>}
          </div>
        </div>
        <div className="opp-amount">
          <div className="amount-value">{formatCurrency(opp.amount)}</div>
          <div className="amount-label">{opp.probability || 0}% probability</div>
        </div>
      </div>
      {opp.next_step && (
        <div className="next-step">
          <strong>Next Step:</strong> {opp.next_step}
        </div>
      )}

      <style jsx>{`
        .opportunity-card {
          background: linear-gradient(135deg, #f8f9ff 0%, #e8f0fe 100%);
          border: 2px solid #667eea;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 25px;
        }
        .opp-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          flex-wrap: wrap;
          gap: 15px;
        }
        .opp-label {
          font-size: 0.85em;
          color: #667eea;
          font-weight: 600;
          margin-bottom: 5px;
        }
        .opp-info h4 {
          margin: 0 0 8px 0;
          color: #333;
        }
        .opp-meta {
          display: flex;
          gap: 15px;
          flex-wrap: wrap;
          align-items: center;
        }
        .stage-badge {
          background: #667eea;
          color: white;
          padding: 4px 12px;
          border-radius: 15px;
          font-size: 0.85em;
        }
        .meta-text { color: #666; }
        .opp-amount { text-align: right; }
        .amount-value {
          font-size: 1.8em;
          font-weight: bold;
          color: #27ae60;
        }
        .amount-label {
          font-size: 0.85em;
          color: #666;
        }
        .next-step {
          margin-top: 15px;
          padding-top: 15px;
          border-top: 1px solid #d0d8e8;
          color: #333;
        }
        .next-step strong { color: #555; }
      `}</style>
    </div>
  );
}

function DealSummaryCard({ summary }: { summary: DealSummary }) {
  const sentimentColors: Record<string, string> = {
    positive: "#27ae60",
    neutral: "#f39c12",
    concerned: "#e74c3c",
  };
  const sentColor = sentimentColors[summary.champion_sentiment || ""] || "#7f8c8d";

  return (
    <div className="deal-summary-card">
      <h4>
        Deal Summary
        {summary.champion_sentiment && (
          <span className="sentiment-badge" style={{ background: sentColor }}>
            {summary.champion_sentiment}
          </span>
        )}
      </h4>
      <p className="current-state">{summary.current_state}</p>

      {summary.blockers_identified.length > 0 && (
        <div className="blockers-box">
          <strong>Blockers/Concerns:</strong>
          <ul>
            {summary.blockers_identified.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        </div>
      )}

      {summary.risk_factors.length > 0 && (
        <div className="risks-box">
          <strong>Risk Factors:</strong>
          <ul>
            {summary.risk_factors.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {summary.next_steps_from_calls.length > 0 && (
        <div className="next-steps-box">
          <strong>Next Steps from Calls:</strong>
          <ul>
            {summary.next_steps_from_calls.slice(0, 3).map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      )}

      <style jsx>{`
        .deal-summary-card {
          background: #fafbfc;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 25px;
          border-left: 4px solid #667eea;
        }
        .deal-summary-card h4 {
          margin: 0 0 15px 0;
          color: #333;
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .sentiment-badge {
          color: white;
          padding: 2px 10px;
          border-radius: 10px;
          font-size: 0.75em;
        }
        .current-state {
          margin: 0 0 15px 0;
          color: #444;
          line-height: 1.6;
        }
        .blockers-box {
          background: #fff3cd;
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 12px;
        }
        .blockers-box strong { color: #856404; }
        .blockers-box ul { margin: 8px 0 0 0; padding-left: 20px; color: #856404; }
        .blockers-box li { margin-bottom: 4px; }
        .risks-box {
          background: #f8d7da;
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 12px;
        }
        .risks-box strong { color: #721c24; }
        .risks-box ul { margin: 8px 0 0 0; padding-left: 20px; color: #721c24; }
        .risks-box li { margin-bottom: 4px; }
        .next-steps-box {
          background: #d4edda;
          border-radius: 8px;
          padding: 12px;
        }
        .next-steps-box strong { color: #155724; }
        .next-steps-box ul { margin: 8px 0 0 0; padding-left: 20px; color: #155724; }
        .next-steps-box li { margin-bottom: 4px; }
      `}</style>
    </div>
  );
}

function GongMetrics({ summary }: { summary: GongSummaryData }) {
  const recentCalls = summary.recent_calls || [];
  const dates = recentCalls.map((c) => c.call_date).filter(Boolean).sort();
  const dateRange =
    dates.length > 1
      ? `${formatDate(dates[dates.length - 1])} - ${formatDate(dates[0])}`
      : dates.length === 1
      ? formatDate(dates[0])
      : "";

  return (
    <div className="gong-metrics">
      <div className="metrics-header">
        <h4>Engagement Metrics</h4>
        <span className="metrics-meta">
          {summary.total_calls} calls · {Math.round(summary.total_duration_minutes || 0)} min
          {dateRange && ` · ${dateRange}`}
        </span>
      </div>
      <div className="metrics-row">
        <div className="metric">
          <span className="metric-value purple">{formatPercent(summary.avg_talk_ratio)}</span>
          <span className="metric-label">talk ratio</span>
        </div>
        <div className="metric">
          <span className="metric-value teal">{summary.avg_interactivity?.toFixed(1) ?? "N/A"}</span>
          <span className="metric-label">interactivity</span>
        </div>
        <div className="metric">
          <span className="metric-value blue">{summary.days_since_last_call ?? "N/A"}</span>
          <span className="metric-label">days since last call</span>
        </div>
      </div>
      {summary.key_themes.length > 0 && (
        <div className="themes">
          <strong>Topics:</strong>
          {summary.key_themes.map((t, i) => (
            <span key={i} className="theme-tag">{t}</span>
          ))}
        </div>
      )}

      <style jsx>{`
        .gong-metrics {
          margin-bottom: 20px;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 10px;
        }
        .metrics-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .metrics-header h4 {
          margin: 0;
          color: #333;
          font-size: 1em;
        }
        .metrics-meta {
          font-size: 0.85em;
          color: #666;
        }
        .metrics-row {
          display: flex;
          gap: 20px;
          margin-bottom: 15px;
          flex-wrap: wrap;
        }
        .metric-value {
          font-size: 1.3em;
          font-weight: bold;
        }
        .metric-value.purple { color: #667eea; }
        .metric-value.teal { color: #11998e; }
        .metric-value.blue { color: #4facfe; }
        .metric-label {
          font-size: 0.8em;
          color: #666;
          margin-left: 4px;
        }
        .themes strong {
          color: #555;
          margin-right: 8px;
        }
        .theme-tag {
          background: #e8f0fe;
          color: #1967d2;
          padding: 4px 12px;
          border-radius: 15px;
          font-size: 0.85em;
          margin-right: 8px;
          margin-bottom: 8px;
          display: inline-block;
        }
      `}</style>
    </div>
  );
}

export function SalesEngagementSection({
  latestOpportunity,
  allOpportunities,
  gongSummary,
  dealSummary,
}: Props) {
  const hasContent = latestOpportunity || dealSummary || (gongSummary && gongSummary.total_calls > 0);
  if (!hasContent) return null;

  return (
    <div className="sales-engagement-section">
      <h3>
        <span className="section-title">Sales Engagement Context</span>
      </h3>

      {latestOpportunity && <OpportunityCard opp={latestOpportunity} />}
      {dealSummary?.current_state && <DealSummaryCard summary={dealSummary} />}

      {allOpportunities.length > 0 && (
        <details className="all-opps">
          <summary>View All Opportunities ({allOpportunities.length})</summary>
          <div className="opps-table-container">
            <table className="opps-table">
              <thead>
                <tr>
                  <th>Opportunity</th>
                  <th>Stage</th>
                  <th>Amount</th>
                  <th>Close Date</th>
                </tr>
              </thead>
              <tbody>
                {allOpportunities.map((opp) => {
                  const stageColor = opp.is_won ? "#27ae60" : opp.is_closed ? "#e74c3c" : "#667eea";
                  return (
                    <tr key={opp.id}>
                      <td>
                        <div className="opp-name">{opp.name}</div>
                        {opp.owner_name && <div className="opp-owner">{opp.owner_name}</div>}
                      </td>
                      <td>
                        <span className="stage-pill" style={{ background: stageColor }}>
                          {opp.stage_name}
                        </span>
                      </td>
                      <td className="amount-cell">{formatCurrency(opp.amount)}</td>
                      <td>{formatDate(opp.close_date)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {gongSummary && gongSummary.total_calls > 0 && <GongMetrics summary={gongSummary} />}

      <style jsx>{`
        .sales-engagement-section {
          background: white;
          border-radius: 16px;
          padding: 25px;
          margin-bottom: 30px;
          box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        }
        .sales-engagement-section h3 {
          margin: 0 0 20px 0;
          padding-bottom: 15px;
          border-bottom: 2px solid #667eea;
          color: #333;
        }
        .section-title {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .all-opps {
          margin-bottom: 25px;
        }
        .all-opps summary {
          cursor: pointer;
          color: #667eea;
          font-weight: 600;
          padding: 10px;
          background: #f8f9fa;
          border-radius: 8px;
        }
        .opps-table-container {
          margin-top: 10px;
          overflow-x: auto;
          border-radius: 8px;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
        }
        .opps-table {
          width: 100%;
          border-collapse: collapse;
          background: white;
          font-size: 0.9em;
        }
        .opps-table thead {
          background: #f1f3f9;
          color: #555;
        }
        .opps-table th {
          padding: 10px 14px;
          text-align: left;
        }
        .opps-table td {
          padding: 10px 14px;
          border-bottom: 1px solid #e9ecef;
        }
        .opp-name { font-weight: 500; }
        .opp-owner { font-size: 0.8em; color: #888; }
        .stage-pill {
          color: white;
          padding: 2px 8px;
          border-radius: 10px;
          font-size: 0.8em;
        }
        .amount-cell { font-weight: 500; }
      `}</style>
    </div>
  );
}
