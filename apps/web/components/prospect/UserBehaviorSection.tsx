"use client";

import type { UserBehaviorAnalysis, Recommendation, CompetitiveMessaging } from "@/lib/types";

interface Props {
  behavior?: UserBehaviorAnalysis | null;
}

const ENGAGEMENT_STYLES: Record<string, { color: string; bg: string }> = {
  high: { color: "#155724", bg: "#d4edda" },
  medium: { color: "#856404", bg: "#fff3cd" },
  low: { color: "#721c24", bg: "#f8d7da" },
  unknown: { color: "#555", bg: "#e9ecef" },
};

function RecommendationCard({ rec }: { rec: Recommendation }) {
  return (
    <div className="recommendation-card">
      <div className="rec-header">
        <span className="rec-category">{rec.category}</span>
        {rec.title && <strong className="rec-title">{rec.title}</strong>}
      </div>
      {rec.description && <p className="rec-description">{rec.description}</p>}
      
      {rec.steps && rec.steps.length > 0 && (
        <div className="rec-steps">
          <strong>Steps:</strong>
          <ol>
            {rec.steps.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        </div>
      )}

      {rec.competitive_messaging && rec.competitive_messaging.length > 0 && (
        <div className="competitive-section">
          <strong className="comp-label">Competitive Intel:</strong>
          {rec.competitive_messaging.map((cm, i) => (
            <CompetitiveCard key={i} messaging={cm} />
          ))}
        </div>
      )}

      {rec.contacts && rec.contacts.length > 0 && (
        <div className="contacts-section">
          <strong>Internal Contacts:</strong>
          <div className="contacts-list">
            {rec.contacts.map((c, i) => (
              <a key={i} href={`mailto:${c.email}`} className="contact-link">
                {c.name}
              </a>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        .recommendation-card {
          background: #f8f9fa;
          border-radius: 10px;
          padding: 15px;
          margin-bottom: 15px;
          border-left: 4px solid #4facfe;
        }
        .rec-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 10px;
        }
        .rec-category {
          background: #4facfe;
          color: white;
          padding: 2px 10px;
          border-radius: 12px;
          font-size: 0.8em;
          text-transform: uppercase;
        }
        .rec-title {
          color: #333;
        }
        .rec-description {
          color: #555;
          margin: 0 0 10px 0;
          line-height: 1.5;
        }
        .rec-steps {
          margin-top: 10px;
        }
        .rec-steps strong { color: #555; }
        .rec-steps ol {
          margin: 8px 0 0 0;
          padding-left: 20px;
          color: #444;
        }
        .rec-steps li { margin-bottom: 4px; }
        .competitive-section {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid #ddd;
        }
        .comp-label {
          color: #e74c3c;
          display: block;
          margin-bottom: 8px;
        }
        .contacts-section {
          margin-top: 10px;
          font-size: 0.9em;
        }
        .contacts-section strong { color: #555; }
        .contacts-list {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 5px;
        }
        .contact-link {
          color: #4facfe;
        }
      `}</style>
    </div>
  );
}

function CompetitiveCard({ messaging }: { messaging: CompetitiveMessaging }) {
  return (
    <div className="competitive-card">
      <div className="comp-header">
        <span className="competitor-name">{messaging.competitor}</span>
        {messaging.mention_count != null && (
          <span className="mention-count">{messaging.mention_count} mentions</span>
        )}
      </div>
      
      {messaging.what_they_said && (
        <div className="comp-quote">
          <span className="quote-label">What they said:</span>
          <q>{messaging.what_they_said}</q>
        </div>
      )}

      {messaging.targeted_response && (
        <div className="comp-response">
          <span className="response-label">Response:</span>
          <p>{messaging.targeted_response}</p>
        </div>
      )}

      {messaging.differentiator && (
        <div className="comp-diff">
          <span className="diff-label">Differentiator:</span> {messaging.differentiator}
        </div>
      )}

      {messaging.talking_point && (
        <div className="comp-talk">
          <span className="talk-label">Talking Point:</span> {messaging.talking_point}
        </div>
      )}

      <style jsx>{`
        .competitive-card {
          background: #fff;
          border-radius: 8px;
          padding: 12px;
          margin-top: 8px;
          border: 1px solid #e74c3c33;
        }
        .comp-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }
        .competitor-name {
          font-weight: 600;
          color: #e74c3c;
        }
        .mention-count {
          font-size: 0.8em;
          color: #888;
        }
        .comp-quote {
          margin-bottom: 8px;
          font-style: italic;
          color: #555;
        }
        .quote-label {
          font-style: normal;
          font-weight: 500;
          margin-right: 5px;
        }
        .comp-response {
          margin-bottom: 8px;
        }
        .response-label {
          font-weight: 500;
          color: #27ae60;
        }
        .comp-response p {
          margin: 4px 0 0 0;
          color: #444;
        }
        .comp-diff, .comp-talk {
          font-size: 0.9em;
          margin-top: 6px;
        }
        .diff-label { font-weight: 500; color: #667eea; }
        .talk-label { font-weight: 500; color: #11998e; }
      `}</style>
    </div>
  );
}

export function UserBehaviorSection({ behavior }: Props) {
  if (!behavior) return null;

  const engStyle = ENGAGEMENT_STYLES[behavior.engagement_level] || ENGAGEMENT_STYLES.unknown;

  return (
    <div className="behavior-section">
      <h3>
        <span className="section-title">User Behavior Analysis</span>
        <span className="engagement-badge" style={{ background: engStyle.bg, color: engStyle.color }}>
          {behavior.engagement_level} engagement
        </span>
      </h3>

      {behavior.summary && (
        <div className="behavior-summary">
          <p>{behavior.summary}</p>
        </div>
      )}

      {behavior.hypothesis && (
        <div className="behavior-hypothesis">
          <strong>Hypothesis:</strong> {behavior.hypothesis}
        </div>
      )}

      {behavior.key_workflows_used.length > 0 && (
        <div className="workflows">
          <strong>Key Workflows:</strong>
          <div className="workflow-tags">
            {behavior.key_workflows_used.map((w, i) => (
              <span key={i} className="workflow-tag">{w}</span>
            ))}
          </div>
        </div>
      )}

      {behavior.adoption_milestones.length > 0 && (
        <div className="milestones">
          <strong>Adoption Milestones:</strong>
          <div className="milestone-grid">
            {behavior.adoption_milestones.map((m, i) => (
              <div key={i} className={`milestone-item ${m.completed ? "completed" : "pending"}`}>
                <span className="milestone-icon">{m.completed ? "✓" : "○"}</span>
                <span className="milestone-name">{m.name.replace(/_/g, " ")}</span>
                {m.count > 0 && <span className="milestone-count">({m.count})</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {behavior.issues_summary && (
        <div className="issues-summary">
          <strong>Issues Summary:</strong>
          <p>{behavior.issues_summary}</p>
        </div>
      )}

      {behavior.recommendations.length > 0 && (
        <div className="recommendations">
          <h4>Recommendations</h4>
          {behavior.recommendations.map((rec, i) => (
            <RecommendationCard key={i} rec={rec} />
          ))}
        </div>
      )}

      <style jsx>{`
        .behavior-section {
          background: white;
          border-radius: 16px;
          padding: 25px;
          margin-bottom: 30px;
          box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        }
        .behavior-section h3 {
          margin: 0 0 20px 0;
          padding-bottom: 15px;
          border-bottom: 2px solid #4facfe;
          color: #333;
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 10px;
        }
        .section-title {
          background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .engagement-badge {
          padding: 4px 12px;
          border-radius: 15px;
          font-size: 0.8em;
          text-transform: uppercase;
        }
        .behavior-summary {
          background: #f8f9fa;
          border-radius: 10px;
          padding: 15px;
          margin-bottom: 20px;
        }
        .behavior-summary p {
          margin: 0;
          color: #444;
          line-height: 1.6;
        }
        .behavior-hypothesis {
          margin-bottom: 20px;
          color: #555;
        }
        .behavior-hypothesis strong { color: #333; }
        .workflows {
          margin-bottom: 20px;
        }
        .workflows strong {
          display: block;
          margin-bottom: 10px;
          color: #333;
        }
        .workflow-tags {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .workflow-tag {
          background: #e8f0fe;
          color: #1967d2;
          padding: 4px 12px;
          border-radius: 15px;
          font-size: 0.9em;
        }
        .milestones {
          margin-bottom: 20px;
        }
        .milestones strong {
          display: block;
          margin-bottom: 10px;
          color: #333;
        }
        .milestone-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 10px;
        }
        .milestone-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          border-radius: 8px;
          font-size: 0.9em;
        }
        .milestone-item.completed {
          background: #d4edda;
          color: #155724;
        }
        .milestone-item.pending {
          background: #f8f9fa;
          color: #666;
        }
        .milestone-icon { font-weight: bold; }
        .milestone-count {
          color: #888;
          font-size: 0.85em;
        }
        .issues-summary {
          background: #fff3cd;
          border-radius: 10px;
          padding: 15px;
          margin-bottom: 20px;
        }
        .issues-summary strong {
          color: #856404;
          display: block;
          margin-bottom: 8px;
        }
        .issues-summary p {
          margin: 0;
          color: #856404;
          line-height: 1.5;
        }
        .recommendations {
          margin-top: 25px;
        }
        .recommendations h4 {
          margin: 0 0 15px 0;
          color: #333;
        }
      `}</style>
    </div>
  );
}
