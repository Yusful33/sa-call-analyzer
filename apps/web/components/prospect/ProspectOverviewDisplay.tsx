"use client";

import type { ProspectOverview } from "@/lib/types";
import { ProspectSummaryCard } from "./ProspectSummaryCard";
import { SalesEngagementSection } from "./SalesEngagementSection";
import { ProductUsageSection } from "./ProductUsageSection";
import { UserBehaviorSection } from "./UserBehaviorSection";

interface Props {
  data: ProspectOverview;
}

export function ProspectOverviewDisplay({ data }: Props) {
  const {
    salesforce,
    latest_opportunity,
    all_opportunities,
    gong_summary,
    sales_engagement,
    pendo_usage,
    user_behavior,
    product_usage,
    last_24h_activity,
    errors,
    data_sources_available,
  } = data;

  if (!salesforce) {
    return (
      <div className="no-data">
        <p>No Salesforce account data found for this lookup.</p>
        {errors.length > 0 && (
          <div className="errors">
            {errors.map((e, i) => (
              <p key={i} className="error-item">{e}</p>
            ))}
          </div>
        )}
        <style jsx>{`
          .no-data {
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            border-radius: 12px;
          }
          .errors { margin-top: 20px; }
          .error-item { color: #e74c3c; }
        `}</style>
      </div>
    );
  }

  return (
    <div className="prospect-overview">
      <ProspectSummaryCard
        salesforce={salesforce}
        adoptionStatus={product_usage?.adoption_status || "unknown"}
        totalCalls={gong_summary?.total_calls || 0}
        daysInCycle={sales_engagement?.days_in_sales_cycle}
        platformUsers={pendo_usage?.unique_visitors || 0}
        last24hActivity={last_24h_activity}
      />

      <SalesEngagementSection
        latestOpportunity={latest_opportunity}
        allOpportunities={all_opportunities}
        gongSummary={gong_summary}
        dealSummary={sales_engagement?.deal_summary}
      />

      <ProductUsageSection
        pendoUsage={pendo_usage}
        productUsage={product_usage}
      />

      <UserBehaviorSection behavior={user_behavior} />

      {data_sources_available.length > 0 && (
        <div className="data-sources">
          <span className="sources-label">Data sources:</span>
          {data_sources_available.map((source) => (
            <span key={source} className="source-badge">{source}</span>
          ))}
        </div>
      )}

      {errors.length > 0 && (
        <details className="errors-details">
          <summary>Warnings ({errors.length})</summary>
          <ul>
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </details>
      )}

      <style jsx>{`
        .prospect-overview {
          max-width: 1200px;
          margin: 0 auto;
        }
        .data-sources {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 20px;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 10px;
        }
        .sources-label {
          color: #666;
          font-size: 0.9em;
        }
        .source-badge {
          background: #e8f0fe;
          color: #1967d2;
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 0.8em;
        }
        .errors-details {
          margin-top: 15px;
          padding: 10px;
          background: #fff3cd;
          border-radius: 8px;
        }
        .errors-details summary {
          cursor: pointer;
          color: #856404;
          font-weight: 600;
        }
        .errors-details ul {
          margin: 10px 0 0 20px;
          padding: 0;
          color: #856404;
        }
        .errors-details li {
          margin-bottom: 5px;
        }
      `}</style>
    </div>
  );
}
