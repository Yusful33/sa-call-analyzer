"use client";

import type { ShareQuery, ShareableTab } from "@/lib/shareableUrl";
import { buildStillnessShareUrl } from "@/lib/shareableUrl";
import { useToast } from "@/components/Toast";

export default function ShareToolbar({
  activeTab,
  hints,
}: {
  activeTab: ShareableTab;
  hints: Partial<ShareQuery>;
}) {
  const toast = useToast();

  async function copy() {
    const url = buildStillnessShareUrl({
      tab: activeTab,
      ...hints,
      account_name: hints.account_name,
      domain: hints.domain,
      sfdc_account_id: hints.sfdc_account_id,
      company_name: hints.company_name,
      company_domain: hints.company_domain,
      demo_account: hints.demo_account,
    });
    if (!url) return;
    try {
      await navigator.clipboard.writeText(url);
      toast.success("Share link copied to clipboard");
    } catch {
      toast.error("Could not copy link");
    }
  }

  return (
    <div style={{ marginTop: 12, display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
      <button type="button" className="btn-secondary" onClick={() => void copy()} style={{ fontSize: 14 }}>
        Copy share link
      </button>
      <span style={{ fontSize: 13, color: "rgba(255,255,255,0.75)" }}>
        Saves the current tab and form fields as a URL you can paste to teammates.
      </span>
    </div>
  );
}
