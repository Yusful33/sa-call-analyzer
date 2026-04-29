export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function formatCurrency(v: number | null | undefined): string {
  if (v == null) return "N/A";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(v);
}

export function formatPercent(v: number | null | undefined): string {
  if (v == null) return "N/A";
  return `${Math.round(v * 100)}%`;
}

export function formatDate(v: string | null | undefined): string {
  if (!v) return "N/A";
  try {
    return new Date(v).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return v;
  }
}

export function formatMinutes(v: number | null | undefined): string {
  if (v == null) return "N/A";
  if (v < 60) return `${Math.round(v)}m`;
  const h = Math.floor(v / 60);
  const m = Math.round(v % 60);
  return `${h}h ${m}m`;
}

/** Matches server `poc_document_generator` export naming when Content-Disposition is unavailable (CORS). */
export function buildPocDocDownloadFilename(accountName: string, template: string): string {
  const typeLabels: Record<string, string> = {
    pot: "PoT",
    poc_saas: "PoC-SaaS",
    poc_vpc: "PoC-VPC",
  };
  const safe =
    (accountName || "Account")
      .replace(/[^\w\s-]/g, "")
      .trim()
      .replace(/\s+/g, "_")
      .slice(0, 80) || "Account";
  const type = typeLabels[template] ?? template;
  const d = new Date();
  const datePart = `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, "0")}${String(d.getUTCDate()).padStart(2, "0")}`;
  return `Arize_AX_${safe}_${type}_${datePart}.docx`;
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
