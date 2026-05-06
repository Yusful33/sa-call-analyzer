"use client";

import type { AccountLast24hActivity, UserLast24hActivity, UserIssueEvent } from "@/lib/types";

interface Props {
  activity: AccountLast24hActivity;
}

function UserActivityCard({ user }: { user: UserLast24hActivity }) {
  const label = user.display_name || user.email || user.visitor_id || "Unknown user";
  const showEmail = user.email && user.display_name;

  return (
    <div className="user-card">
      <div className="user-header">
        <span className="user-name">{label}</span>
        {showEmail && <div className="user-email">{user.email}</div>}
      </div>
      {user.summary && <div className="user-summary">{user.summary}</div>}
      
      {user.fullstory_behavior_summary && (
        <div className="narrative-box">
          <div className="narrative-label">Session narrative (from FullStory data export)</div>
          <div className="narrative-text">{user.fullstory_behavior_summary}</div>
          <div className="narrative-note">
            Built from warehouse events (URLs, clicks, loads), not video. Use Recording for the replay.
          </div>
        </div>
      )}

      {user.top_features_24h.length > 0 && (
        <div className="activity-list">
          <strong className="list-label">Features</strong>
          <ul>
            {user.top_features_24h.slice(0, 5).map((f) => (
              <li key={f.id}>
                {f.name || f.id} <span className="count">({f.count})</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {user.top_pages_24h.length > 0 && (
        <div className="activity-list">
          <strong className="list-label">Pages</strong>
          <ul>
            {user.top_pages_24h.slice(0, 5).map((p) => (
              <li key={p.id}>
                {p.name || p.id} <span className="count">({p.count})</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {user.fullstory_issues_24h.length > 0 && (
        <div className="issues-section">
          <span className="issues-label">FullStory (friction)</span>
          {user.fullstory_issues_24h.slice(0, 4).map((issue, idx) => (
            <IssueItem key={idx} issue={issue} />
          ))}
        </div>
      )}

      <style jsx>{`
        .user-card {
          margin-bottom: 14px;
          padding: 12px;
          background: rgba(0, 0, 0, 0.22);
          border-radius: 10px;
          border-left: 3px solid #4facfe;
        }
        .user-header { margin-bottom: 6px; }
        .user-name { font-weight: 600; color: #fff; }
        .user-email { font-size: 0.8em; color: #8ab4f8; }
        .user-summary {
          font-size: 0.88em;
          color: #c8d0dc;
          line-height: 1.45;
          margin-top: 6px;
        }
        .narrative-box {
          margin-top: 10px;
          padding: 10px 12px;
          background: rgba(126, 214, 255, 0.08);
          border-radius: 8px;
          border: 1px solid rgba(126, 214, 255, 0.25);
        }
        .narrative-label {
          font-size: 0.72em;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: #7dd3fc;
          margin-bottom: 6px;
        }
        .narrative-text {
          font-size: 0.86em;
          color: #e2edf5;
          line-height: 1.55;
          white-space: pre-wrap;
        }
        .narrative-note {
          font-size: 0.68em;
          color: #7a8fa6;
          margin-top: 8px;
        }
        .activity-list {
          margin-top: 8px;
          font-size: 0.78em;
          color: #a8b0bc;
        }
        .list-label { color: #9ee0ff; }
        .activity-list ul {
          margin: 4px 0 0 18px;
          padding: 0;
        }
        .activity-list li {
          margin: 2px 0;
          color: #d0d8e4;
        }
        .count { opacity: 0.75; }
        .issues-section {
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        .issues-label {
          font-size: 0.75em;
          color: #f5b041;
        }
      `}</style>
    </div>
  );
}

function IssueItem({ issue }: { issue: UserIssueEvent }) {
  const context = issue.page_context || issue.issue_type || "";
  const hasRecording = issue.recording_url?.startsWith("https://");

  return (
    <div className="issue-item">
      <span className="issue-context">{context}</span>
      {hasRecording && (
        <a href={issue.recording_url!} target="_blank" rel="noopener noreferrer" className="recording-link">
          Recording
        </a>
      )}
      <style jsx>{`
        .issue-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 8px;
          margin-top: 4px;
          font-size: 0.78em;
          color: #e8c547;
        }
        .recording-link {
          color: #f5b041;
          font-size: 0.78em;
        }
      `}</style>
    </div>
  );
}

export function Last24hActivitySection({ activity }: Props) {
  const { active_users, account_summary, total_active_users_24h, pendo_total_events_24h } = activity;

  if (!active_users.length && !account_summary) {
    return null;
  }

  return (
    <div className="last-24h-section">
      <div className="section-label">Last 24 hours (UTC) — by user</div>
      {(total_active_users_24h != null || pendo_total_events_24h != null) && (
        <div className="meta-info">
          {total_active_users_24h} user(s) in this view · {pendo_total_events_24h || 0} Pendo events (24h)
        </div>
      )}
      {account_summary && <p className="account-summary">{account_summary}</p>}
      {active_users.length > 0 ? (
        active_users.map((user, idx) => <UserActivityCard key={user.visitor_id || idx} user={user} />)
      ) : (
        !account_summary && <p className="no-activity">No activity in this window.</p>
      )}

      <style jsx>{`
        .last-24h-section {
          margin-top: 20px;
          padding: 16px;
          background: rgba(255, 255, 255, 0.07);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .section-label {
          font-size: 0.72em;
          text-transform: uppercase;
          letter-spacing: 0.07em;
          color: #7dd3fc;
          margin-bottom: 10px;
        }
        .meta-info {
          font-size: 0.8em;
          color: #8fa3bf;
          margin-bottom: 10px;
        }
        .account-summary {
          margin: 0 0 12px 0;
          line-height: 1.5;
          color: #e8eef5;
          font-size: 0.95em;
        }
        .no-activity {
          margin: 0;
          color: #9aa5b8;
          font-size: 0.9em;
        }
      `}</style>
    </div>
  );
}
