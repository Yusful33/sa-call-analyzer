"use client";

import { useState } from "react";
import { apiPostBlob } from "@/lib/api";
import { buildPocDocDownloadFilename, downloadBlob, escapeHtml } from "@/lib/helpers";
import type { ResolveAccountFn } from "@/lib/accountResolve";

export default function PocPotTab({
  onLoading,
  onResult,
  resolveAccount,
}: {
  onLoading: (msg: string) => void;
  onResult: (html: string) => void;
  resolveAccount: ResolveAccountFn;
}) {
  const [accountName, setAccountName] = useState("");
  const [template, setTemplate] = useState("poc_saas");
  const [domain, setDomain] = useState("");
  const [notes, setNotes] = useState("");

  const templateLabels: Record<string, string> = {
    pot: "Proof of Technology (PoT)",
    poc_saas: "Proof of Concept — SaaS",
    poc_vpc: "Proof of Concept — VPC",
  };

  async function submit() {
    if (!accountName.trim()) { alert("Please enter a company name."); return; }
    const resolved = await resolveAccount({
      accountName: accountName.trim(),
      accountDomain: domain.trim(),
      sfdcAccountId: "",
    });
    if (!resolved.proceed) return;
    const acc = (resolved.accountName || accountName).trim();
    const domResolved = (resolved.accountDomain ?? domain).trim();
    if (acc !== accountName.trim()) setAccountName(acc);
    onLoading("Generating Word document (BigQuery + AI)...");
    try {
      const body: Record<string, unknown> = {
        account_name: acc,
        document_template: template,
      };
      if (domResolved) body.domain = domResolved;
      if (notes.trim()) body.manual_notes = notes.trim();
      const hint = buildPocDocDownloadFilename(acc, template);
      const { blob, filename } = await apiPostBlob(
        "/api/generate-poc-document",
        body,
        hint
      );
      const fname = filename || "arize_ax_document.docx";
      downloadBlob(blob, fname);

      const sizeMb = (blob.size / (1024 * 1024)).toFixed(2);
      onResult(
        `<div style="padding:20px;background:#e8f5e9;border-radius:10px;border:1px solid #c8e6c9;">` +
        `<h2 style="margin-top:0;color:#2e7d32;">Document Generated Successfully</h2>` +
        `<p style="font-size:15px;margin-bottom:12px;">Your Word document has been downloaded.</p>` +
        `<div style="padding:14px;background:white;border-radius:8px;border:1px solid #e0e0e0;">` +
        `<p style="margin:0 0 8px;"><strong>File:</strong> ${escapeHtml(fname)}</p>` +
        `<p style="margin:0 0 8px;"><strong>Account:</strong> ${escapeHtml(acc)}</p>` +
        `<p style="margin:0 0 8px;"><strong>Template:</strong> ${escapeHtml(templateLabels[template] ?? template)}</p>` +
        `<p style="margin:0;"><strong>Size:</strong> ${sizeMb} MB</p>` +
        `</div>` +
        `<p style="color:#666;font-size:13px;margin-top:14px;margin-bottom:0;">Open the .docx file in Word, Google Docs, or any compatible editor to review and customize.</p>` +
        `</div>`
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      onResult(
        `<div style="padding:20px;color:#c62828;background:#ffebee;border-radius:10px;border:1px solid #ffcdd2;">` +
        `<h3 style="margin-top:0;">Document Generation Failed</h3>` +
        `<p>${escapeHtml(msg)}</p>` +
        `</div>`
      );
    }
  }

  return (
    <>
      {/* Trial types reference table */}
      <section className="pocpot-reference" aria-labelledby="trial-types-heading">
        <div className="trial-types-matrix" role="region">
          <h3 className="trial-types-title" id="trial-types-heading">Arize AX trial types</h3>
          <div className="trial-types-scroll">
            <table className="trial-types-table">
              <thead>
                <tr>
                  <th scope="col"></th>
                  <th scope="col" className="col-workshop">Technical Workshop (Demo)</th>
                  <th scope="col" className="col-pot">Proof of Technology (PoT)</th>
                  <th scope="colgroup" colSpan={2}>Proof of Concept (PoC)</th>
                </tr>
                <tr className="subhead">
                  <th scope="col"></th>
                  <th scope="col" className="col-workshop"></th>
                  <th scope="col" className="col-pot"></th>
                  <th scope="col" className="col-saas">SaaS</th>
                  <th scope="col" className="col-vpc">VPC</th>
                </tr>
              </thead>
              <tbody>
                <tr className="row-purpose"><th scope="row" className="row-label">Purpose</th><td className="data col-workshop">Demonstrate AX value in a demo.</td><td className="data col-pot">Workshops to map AX to your use cases.</td><td className="data col-saas">Prove AX fits your GenAI stack and hit success criteria.</td><td className="data col-vpc">Prove AX fits in your environment via success criteria.</td></tr>
                <tr className="row-deploy"><th scope="row" className="row-label">Deployment / data</th><td className="data col-workshop">SaaS demo</td><td className="data col-pot">SaaS / synthetic</td><td className="data col-saas">SaaS / customer data*</td><td className="data col-vpc">VPC / customer data</td></tr>
                <tr className="row-length"><th scope="row" className="row-label">Length of trial</th><td className="data col-workshop">1 day</td><td className="data col-pot">&lt; 1 week</td><td className="data col-saas">&lt; 2 weeks</td><td className="data col-vpc">&lt; 4 weeks</td></tr>
                <tr className="row-cost"><th scope="row" className="row-label">Cost</th><td className="data col-workshop">Free</td><td className="data col-pot">Free</td><td className="data col-saas">Free</td><td className="data col-vpc">TBD</td></tr>
                <tr className="row-urgency"><th scope="row" className="row-label">Customer urgency</th><td className="data col-workshop">Very high</td><td className="data col-pot">High</td><td className="data col-saas">Medium</td><td className="data col-vpc">Low</td></tr>
                <tr className="row-effort"><th scope="row" className="row-label">Level of effort</th><td className="data col-workshop">None</td><td className="data col-pot">Low</td><td className="data col-saas">High</td><td className="data col-vpc">Very high**</td></tr>
              </tbody>
            </table>
          </div>
          <div className="trial-types-footnotes">
            <div>* PII/PHI can be redacted within a SaaS PoC.</div>
            <div>** Requires infrastructure resources.</div>
          </div>
        </div>
      </section>

      <div className="input-section">
        <label htmlFor="pocAccountName">Company name</label>
        <input type="text" id="pocAccountName" placeholder="e.g., Acme Corp, Stripe" value={accountName} onChange={(e) => setAccountName(e.target.value)} />
        <p className="help-text">Used for BigQuery account resolution.</p>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="pocDocTemplate">Document type</label>
        <select id="pocDocTemplate" value={template} onChange={(e) => setTemplate(e.target.value)}>
          <option value="pot">Proof of Technology (PoT)</option>
          <option value="poc_saas">Proof of Concept — SaaS (PoC SaaS)</option>
          <option value="poc_vpc">Proof of Concept — VPC (PoC VPC)</option>
        </select>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="pocAccountDomain">Email domain (optional)</label>
        <input type="text" id="pocAccountDomain" placeholder="e.g., acme.com" value={domain} onChange={(e) => setDomain(e.target.value)} />
        <p className="help-text">When set with company name, <strong>both</strong> must match.</p>
      </div>
      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="pocManualNotes">Notes for the writer (optional)</label>
        <textarea id="pocManualNotes" rows={3} placeholder="Deal context, success criteria hints, or anything not in Gong/Pendo yet." value={notes} onChange={(e) => setNotes(e.target.value)} />
      </div>
      <p className="help-text" style={{ marginTop: 15, padding: 12, background: "#e8f5e9", borderRadius: 8 }}>
        {"📄 "}<strong>PoC / PoT Document</strong> loads data from BigQuery, then fills template placeholders in the Word file for download.
      </p>
      <div className="button-group">
        <button className="btn-primary" onClick={submit}>Generate &amp; download Word</button>
      </div>
    </>
  );
}
