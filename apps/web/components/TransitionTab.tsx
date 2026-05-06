"use client";

import { useMemo, useState } from "react";
import { apiPost } from "@/lib/api";
import { escapeHtml } from "@/lib/helpers";
import { useToast } from "@/components/Toast";
import type { ResolveAccountFn } from "@/lib/accountResolve";

type TransitionResponse = {
  account_name: string;
  markdown: string;
  model: string;
  data_sources: string[];
};

function renderInlineMd(text: string): string {
  let s = escapeHtml(text);
  // Markdown links: [label](url)
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_m, label: string, url: string) => {
    const safeUrl = url.replace(/"/g, "%22");
    return `<a href="${safeUrl}" target="_blank" rel="noopener">${label}</a>`;
  });
  // inline code: `code`
  s = s.replace(/`([^`]+)`/g, (_m, c: string) => `<code>${c}</code>`);
  // bold: **text**
  s = s.replace(/\*\*([^*]+)\*\*/g, (_m, c: string) => `<strong>${c}</strong>`);
  // italic: _text_ (avoid colliding with snake_case text)
  s = s.replace(
    /(^|[^a-zA-Z0-9_])_([^_]{1,200})_(?=$|[^a-zA-Z0-9_])/g,
    (_m, lead: string, c: string) => `${lead}<em>${c}</em>`,
  );
  // soft tags: (public knowledge), (inferred)
  s = s.replace(/\(public knowledge\)/gi, '<span class="trans-tag public">public</span>');
  s = s.replace(/\(inferred\)/gi, '<span class="trans-tag inferred">inferred</span>');
  return s;
}

function renderMarkdownToHtml(md: string): string {
  if (!md) return "";
  const lines = String(md).replace(/\r\n?/g, "\n").split("\n");
  const out: string[] = [];
  let listType: "ul" | "ol" | null = null;
  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };
  for (const rawLine of lines) {
    const line = rawLine ?? "";
    if (!line.trim()) {
      closeList();
      continue;
    }
    let m: RegExpMatchArray | null;
    if ((m = line.match(/^###\s+(.+)$/))) {
      closeList();
      out.push(`<h3>${renderInlineMd(m[1].trim())}</h3>`);
      continue;
    }
    if ((m = line.match(/^##\s+(.+)$/))) {
      closeList();
      out.push(`<h2>${renderInlineMd(m[1].trim())}</h2>`);
      continue;
    }
    if ((m = line.match(/^#\s+(.+)$/))) {
      closeList();
      out.push(`<h1>${renderInlineMd(m[1].trim())}</h1>`);
      continue;
    }
    if ((m = line.match(/^\s*[-*]\s+(.+)$/))) {
      if (listType !== "ul") {
        closeList();
        out.push("<ul>");
        listType = "ul";
      }
      out.push(`<li>${renderInlineMd(m[1].trim())}</li>`);
      continue;
    }
    if ((m = line.match(/^\s*\d+\.\s+(.+)$/))) {
      if (listType !== "ol") {
        closeList();
        out.push("<ol>");
        listType = "ol";
      }
      out.push(`<li>${renderInlineMd(m[1].trim())}</li>`);
      continue;
    }
    closeList();
    out.push(`<p>${renderInlineMd(line.trim())}</p>`);
  }
  closeList();
  return out.join("\n");
}

function slugify(s: string): string {
  return (
    String(s || "transition")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 60) || "transition"
  );
}

export default function TransitionTab({
  onLoading,
  resolveAccount,
}: {
  onLoading: (msg: string) => void;
  resolveAccount: ResolveAccountFn;
}) {
  const toast = useToast();
  const [accountName, setAccountName] = useState("");
  const [manualNotes, setManualNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<TransitionResponse | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const [copied, setCopied] = useState(false);

  const renderedHtml = useMemo(
    () => (result?.markdown ? renderMarkdownToHtml(result.markdown) : ""),
    [result?.markdown],
  );
  // React 19 regression (facebook/react#31660): dangerouslySetInnerHTML uses
  // object identity not string equality, so without memoizing the object the
  // div's innerHTML is reassigned on every parent rerender — clobbering any
  // user-driven DOM state (e.g. native <details>/<summary> toggles).
  const renderedInnerHtml = useMemo(
    () => ({ __html: renderedHtml }),
    [renderedHtml],
  );

  async function submit() {
    const an = accountName.trim();
    if (!an) {
      toast.warning("Please enter a customer / account name.");
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
    setResult(null);
    setShowRaw(false);
    onLoading(
      `Pulling BigQuery + Gong + Pendo data and asking the LLM to draft the KT doc for ${resolvedAccount}... This may take 30-60 seconds.`,
    );
    try {
      const data = await apiPost<TransitionResponse>("/api/transition-to-cs", {
        account_name: resolvedAccount,
        manual_notes: manualNotes.trim() || null,
      });
      setResult(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      toast.error("Knowledge Transfer generation failed: " + message);
    } finally {
      setSubmitting(false);
      onLoading("");
    }
  }

  async function copyMarkdown() {
    if (!result?.markdown) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(result.markdown);
      } else {
        const ta = document.createElement("textarea");
        ta.value = result.markdown;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      setCopied(true);
      toast.success("Markdown copied to clipboard");
      setTimeout(() => setCopied(false), 1600);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      toast.error("Could not copy markdown: " + message);
    }
  }

  function downloadMarkdown() {
    if (!result?.markdown) return;
    const blob = new Blob([result.markdown], {
      type: "text/markdown;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${slugify(result.account_name)}_kt.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      <div className="input-section">
        <label htmlFor="transAccountName">Customer / Account name</label>
        <input
          type="text"
          id="transAccountName"
          placeholder="e.g., Acme Corp, Stripe"
          autoComplete="off"
          value={accountName}
          onChange={(e) => setAccountName(e.target.value)}
        />
        <p className="help-text">
          Closed-won (or about-to-close) account. Same BigQuery resolution as the Prospect Overview tab.
        </p>
      </div>

      <div className="input-section" style={{ marginTop: 15 }}>
        <label htmlFor="transManualNotes">SA color / sticking points (optional)</label>
        <textarea
          id="transManualNotes"
          rows={4}
          placeholder="Politics, champions, things that aren't in Gong / Salesforce yet, asks the CSE should know about, links (POC scoping doc, product feature ticket), etc."
          value={manualNotes}
          onChange={(e) => setManualNotes(e.target.value)}
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
        {"\u{1F501} "}
        <strong>Transition to CS</strong> pulls Salesforce, Gong, Pendo, and FullStory from BigQuery, then asks the LLM
        to draft a Pre-Sales &rarr; Post-Sales <em>Internal Knowledge Transfer</em> document following the standard
        Arize KT template (Executive Summary, Stakeholders, Contract, Use Cases, Expansion, POC, Tech Stack, Asks,
        Risks).
      </p>

      <div className="button-group">
        <button
          type="button"
          className="btn-primary"
          onClick={submit}
          disabled={submitting}
          id="transGenerateBtn"
        >
          {submitting ? "Generating..." : "Generate Knowledge Transfer"}
        </button>
      </div>

      {result ? (
        <div className="trans-result" id="transResult">
          <div className="trans-result-toolbar">
            <div className="trans-result-meta">
              <span className="trans-meta-label" title={result.account_name}>
                {result.account_name || "—"}
              </span>
              <span className="trans-meta-pill">model: {result.model || "unknown"}</span>
              <span className="trans-meta-pill">
                sources: {result.data_sources?.length ? result.data_sources.join(", ") : "none"}
              </span>
            </div>
            <div className="trans-result-actions">
              <button
                type="button"
                className={`trans-action-btn${copied ? " copied" : ""}`}
                onClick={copyMarkdown}
              >
                {copied ? "Copied!" : "Copy markdown"}
              </button>
              <button type="button" className="trans-action-btn" onClick={downloadMarkdown}>
                Download .md
              </button>
              <button
                type="button"
                className="trans-action-btn"
                onClick={() => setShowRaw((v) => !v)}
              >
                {showRaw ? "View rendered" : "View raw"}
              </button>
            </div>
          </div>

          {showRaw ? (
            <pre className="trans-result-raw">{result.markdown}</pre>
          ) : (
            <div
              className="trans-result-rendered"
              dangerouslySetInnerHTML={renderedInnerHtml}
            />
          )}
        </div>
      ) : null}
    </>
  );
}
