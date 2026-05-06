"use client";

import type { PendoUsageData, ProductUsageSummary } from "@/lib/types";
import { formatMinutes, formatDate } from "@/lib/helpers";

interface Props {
  pendoUsage?: PendoUsageData | null;
  productUsage?: ProductUsageSummary | null;
}

const TREND_ICONS: Record<string, { icon: string; color: string }> = {
  growing: { icon: "↑", color: "#27ae60" },
  stable: { icon: "→", color: "#3498db" },
  declining: { icon: "↓", color: "#e74c3c" },
};

export function ProductUsageSection({ pendoUsage, productUsage }: Props) {
  if (!pendoUsage && !productUsage) return null;

  const trendInfo = TREND_ICONS[productUsage?.trend || "stable"];

  return (
    <div className="product-usage-section">
      <h3>
        <span className="section-title">Product Usage</span>
      </h3>

      {productUsage && (
        <div className="usage-summary">
          <div className="summary-grid">
            <div className="summary-item">
              <div className="summary-value">{productUsage.total_users}</div>
              <div className="summary-label">Total Users</div>
            </div>
            <div className="summary-item">
              <div className="summary-value">{productUsage.active_users_last_7_days}</div>
              <div className="summary-label">Active (7d)</div>
            </div>
            <div className="summary-item">
              <div className="summary-value">{productUsage.active_users_last_30_days}</div>
              <div className="summary-label">Active (30d)</div>
            </div>
            <div className="summary-item">
              <div className="summary-value" style={{ color: trendInfo.color }}>
                {trendInfo.icon} {productUsage.trend}
              </div>
              <div className="summary-label">Trend</div>
            </div>
          </div>

          {productUsage.last_active_user && (
            <div className="last-active">
              Last active: <strong>{productUsage.last_active_user}</strong>
              {productUsage.last_platform_activity && (
                <span className="activity-date">{formatDate(productUsage.last_platform_activity)}</span>
              )}
            </div>
          )}
        </div>
      )}

      {pendoUsage && (
        <div className="pendo-details">
          <div className="pendo-header">
            <span className="pendo-label">Pendo Analytics</span>
            <span className="pendo-meta">
              {pendoUsage.unique_visitors} visitors · {formatMinutes(pendoUsage.total_minutes)} total
            </span>
          </div>

          <div className="metrics-row">
            <div className="metric-box">
              <div className="metric-value">{pendoUsage.total_events.toLocaleString()}</div>
              <div className="metric-label">Total Events</div>
            </div>
            <div className="metric-box">
              <div className="metric-value">{pendoUsage.active_days_last_7}</div>
              <div className="metric-label">Active Days (7d)</div>
            </div>
            <div className="metric-box">
              <div className="metric-value">{formatMinutes(pendoUsage.avg_session_duration_minutes)}</div>
              <div className="metric-label">Avg Session</div>
            </div>
          </div>

          {pendoUsage.top_features.length > 0 && (
            <div className="feature-list">
              <div className="list-header">Top Features</div>
              <div className="list-items">
                {pendoUsage.top_features.slice(0, 5).map((f) => (
                  <div key={f.feature_id} className="list-item">
                    <span className="item-name">{f.feature_name || f.feature_id}</span>
                    <span className="item-count">{f.event_count} events</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {pendoUsage.top_pages.length > 0 && (
            <div className="feature-list">
              <div className="list-header">Top Pages</div>
              <div className="list-items">
                {pendoUsage.top_pages.slice(0, 5).map((p) => (
                  <div key={p.page_id} className="list-item">
                    <span className="item-name">{p.page_name || p.page_id}</span>
                    <span className="item-count">{p.view_count} views</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {pendoUsage.recent_visitors.length > 0 && (
            <details className="visitors-details">
              <summary>Recent Visitors ({pendoUsage.recent_visitors.length})</summary>
              <div className="visitors-list">
                {pendoUsage.recent_visitors.slice(0, 10).map((v) => (
                  <div key={v.visitor_id} className="visitor-row">
                    <span className="visitor-name">
                      {v.display_name || v.email || v.visitor_id}
                    </span>
                    <span className="visitor-meta">
                      {v.total_events} events · {formatMinutes(v.total_minutes)}
                      {v.last_visit && ` · ${formatDate(v.last_visit)}`}
                    </span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}

      <style jsx>{`
        .product-usage-section {
          background: white;
          border-radius: 16px;
          padding: 25px;
          margin-bottom: 30px;
          box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        }
        .product-usage-section h3 {
          margin: 0 0 20px 0;
          padding-bottom: 15px;
          border-bottom: 2px solid #11998e;
          color: #333;
        }
        .section-title {
          background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .usage-summary {
          margin-bottom: 20px;
        }
        .summary-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
          gap: 15px;
          margin-bottom: 15px;
        }
        .summary-item {
          text-align: center;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 10px;
        }
        .summary-value {
          font-size: 1.5em;
          font-weight: bold;
          color: #333;
        }
        .summary-label {
          font-size: 0.8em;
          color: #666;
        }
        .last-active {
          font-size: 0.9em;
          color: #555;
        }
        .activity-date {
          margin-left: 10px;
          color: #888;
        }
        .pendo-details {
          background: #f8f9fa;
          border-radius: 12px;
          padding: 20px;
        }
        .pendo-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
        }
        .pendo-label {
          font-weight: 600;
          color: #333;
        }
        .pendo-meta {
          font-size: 0.85em;
          color: #666;
        }
        .metrics-row {
          display: flex;
          gap: 15px;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }
        .metric-box {
          flex: 1;
          min-width: 100px;
          text-align: center;
          padding: 12px;
          background: white;
          border-radius: 8px;
        }
        .metric-box .metric-value {
          font-size: 1.3em;
          font-weight: bold;
          color: #11998e;
        }
        .metric-box .metric-label {
          font-size: 0.75em;
          color: #666;
        }
        .feature-list {
          margin-bottom: 15px;
        }
        .list-header {
          font-weight: 600;
          color: #555;
          margin-bottom: 8px;
          font-size: 0.9em;
        }
        .list-items {
          background: white;
          border-radius: 8px;
          padding: 10px;
        }
        .list-item {
          display: flex;
          justify-content: space-between;
          padding: 6px 0;
          border-bottom: 1px solid #eee;
        }
        .list-item:last-child { border-bottom: none; }
        .item-name {
          font-size: 0.9em;
          color: #333;
        }
        .item-count {
          font-size: 0.85em;
          color: #888;
        }
        .visitors-details {
          margin-top: 15px;
        }
        .visitors-details summary {
          cursor: pointer;
          color: #11998e;
          font-weight: 600;
        }
        .visitors-list {
          margin-top: 10px;
          background: white;
          border-radius: 8px;
          padding: 10px;
        }
        .visitor-row {
          display: flex;
          justify-content: space-between;
          padding: 8px 0;
          border-bottom: 1px solid #eee;
          flex-wrap: wrap;
          gap: 8px;
        }
        .visitor-row:last-child { border-bottom: none; }
        .visitor-name { font-weight: 500; color: #333; }
        .visitor-meta { font-size: 0.85em; color: #888; }
      `}</style>
    </div>
  );
}
