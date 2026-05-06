"use client";

import { useState } from "react";
import { apiPostBlob } from "@/lib/api";
import { buildPocDocDownloadFilename, downloadBlob } from "@/lib/helpers";
import type { ResolveAccountFn } from "@/lib/accountResolve";

type DocumentTemplateId = "pot" | "poc_saas" | "poc_vpc";

const TEMPLATES: { id: DocumentTemplateId; label: string; blurb: string }[] = [
  {
    id: "pot",
    label: "Proof of Technology (PoT)",
    blurb: "Lightweight technical fit check before a full PoC.",
  },
  {
    id: "poc_saas",
    label: "Proof of Concept — SaaS",
    blurb: "Full PoC against Arize SaaS. Most common path.",
  },
  {
    id: "poc_vpc",
    label: "Proof of Concept — VPC",
    blurb: "Customer-VPC deployment. Highest effort, used for regulated buyers.",
  },
];

export default function PocTab({
  onLoading,
  resolveAccount,
}: {
  onLoading: (msg: string) => void;
  resolveAccount: ResolveAccountFn;
}) {
  const [accountName, setAccountName] = useState("");
  const [template, setTemplate] = useState<DocumentTemplateId>("poc_saas");
  const [manualNotes, setManualNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit() {
    const an = accountName.trim();
    if (!an) {
      alert("Please enter a company / account name.");
      return;
    }

    const resolved = await resolveAccount({
      accountName: an,
      accountDomain: "",
      sfdcAccountId: "",
    });
    if (!resolved.proceed) return;
    const resolvedAccount = (resolved.accountName || an).trim();
    if (resolvedAccount !== an) setAccountName(resolvedAccount);

    setSubmitting(true);
    onLoading(
      `Pulling Salesforce, Gong, and Pendo data and filling the ${template.toUpperCase()} template for ${resolvedAccount}...`,
    );
    try {
      const { blob, filename } = await apiPostBlob(
        "/api/generate-poc-document",
        {
          account_name: resolvedAccount,
          document_template: template,
          manual_notes: manualNotes.trim() || null,
        },
        buildPocDocDownloadFilename(resolvedAccount, template),
      );
      downloadBlob(blob, filename);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      alert("Document generation failed: " + message);
    } finally {
      setSubmitting(false);
      onLoading("");
    }
  }

  const activeTemplate = TEMPLATES.find((t) => t.id === template) ?? TEMPLATES[1];

  return (
    <>
      <div className="input-section">
        <label htmlFor="pocAccountName">Company / Account name</label>
        <input
          type="text"
          id="pocAccountName"
          placeholder="e.g., Acme Corp, Stripe"
          autoComplete="off"
          value={accountName}
          onChange={(e) => setAccountName(e.target.value)}
        />
        <p className="help-text">
          Same BigQuery account resolution as the Prospect Overview tab.
        </p>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="pocDocTemplate">Document template</label>
        <select
          id="pocDocTemplate"
          value={template}
          onChange={(e) => setTemplate(e.target.value as DocumentTemplateId)}
          style={{
            width: "100%",
            padding: 12,
            border: "2px solid #e0e0e0",
            borderRadius: 8,
            fontSize: 14,
            background: "white",
            cursor: "pointer",
          }}
        >
          {TEMPLATES.map((t) => (
            <option key={t.id} value={t.id}>
              {t.label}
            </option>
          ))}
        </select>
        <p className="help-text">{activeTemplate.blurb}</p>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="pocManualNotes">Notes for the writer (optional)</label>
        <textarea
          id="pocManualNotes"
          rows={3}
          value={manualNotes}
          onChange={(e) => setManualNotes(e.target.value)}
          placeholder="Deal context, success criteria hints, or anything not in Gong / Pendo yet."
          style={{
            width: "100%",
            padding: 12,
            border: "2px solid #e0e0e0",
            borderRadius: 8,
            fontSize: 14,
            resize: "vertical",
          }}
        />
      </div>

      <p
        className="help-text"
        style={{
          marginTop: 15,
          padding: 12,
          background: "var(--arize-magenta-tint)",
          borderLeft: "4px solid var(--arize-magenta)",
          borderRadius: 8,
        }}
      >
        {"\u{1F4C4} "}
        <strong>PoC / PoT Document</strong> loads Salesforce, Gong, Pendo, and FullStory data
        from BigQuery, fills in the standard Word template (company name, stack choices, trial
        schedule), and downloads the populated <code>.docx</code> ready to share with the customer.
      </p>

      <div className="button-group">
        <button
          type="button"
          className="btn-primary"
          onClick={submit}
          disabled={submitting}
        >
          {submitting ? "Generating..." : "Generate & download Word"}
        </button>
      </div>
    </>
  );
}
