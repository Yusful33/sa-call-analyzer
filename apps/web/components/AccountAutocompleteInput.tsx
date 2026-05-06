"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiPost } from "@/lib/api";
import type { AccountSuggestionsResponse, AccountSuggestionMatch } from "@/lib/accountResolve";

type Props = {
  id: string;
  label: string;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  /** When set with account query, forwarded to `/api/account-suggestions` */
  domainHint?: string;
  helpText?: string;
  disabled?: boolean;
};

export default function AccountAutocompleteInput({
  id,
  label,
  placeholder,
  value,
  onChange,
  domainHint,
  helpText,
  disabled,
}: Props) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [matches, setMatches] = useState<AccountSuggestionMatch[]>([]);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const fetchSuggest = useCallback(
    async (q: string) => {
      if (q.trim().length < 2) {
        setMatches([]);
        return;
      }
      setLoading(true);
      try {
        const r = await apiPost<AccountSuggestionsResponse>("/api/account-suggestions", {
          account_name: q.trim(),
          domain: (domainHint || "").trim() || null,
        });
        const m = r.matches ?? [];
        setMatches(m.slice(0, 8));
        setOpen(m.length > 0);
      } catch {
        setMatches([]);
        setOpen(false);
      } finally {
        setLoading(false);
      }
    },
    [domainHint]
  );

  function handleInput(v: string) {
    onChange(v);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      void fetchSuggest(v);
    }, 280);
  }

  function pick(m: AccountSuggestionMatch) {
    onChange(m.name);
    setOpen(false);
    setMatches([]);
  }

  const listboxId = `${id}-suggestions`;

  return (
    <div className="input-section" ref={wrapRef}>
      <label htmlFor={id}>{label}</label>
      <div style={{ position: "relative" }}>
        <input
          type="text"
          id={id}
          role="combobox"
          autoComplete="off"
          placeholder={placeholder}
          value={value}
          disabled={disabled}
          onChange={(e) => handleInput(e.target.value)}
          onFocus={() => {
            if (matches.length) setOpen(true);
          }}
          aria-autocomplete="list"
          aria-expanded={open}
          aria-haspopup="listbox"
          aria-controls={open && matches.length > 0 ? listboxId : undefined}
        />
        {open && matches.length > 0 ? (
          <ul
            id={listboxId}
            role="listbox"
            style={{
              position: "absolute",
              zIndex: 30,
              left: 0,
              right: 0,
              top: "100%",
              marginTop: 4,
              maxHeight: 240,
              overflowY: "auto",
              background: "white",
              border: "1px solid var(--arize-border,#e5e7eb)",
              borderRadius: 8,
              boxShadow: "0 8px 24px rgba(15,17,23,0.12)",
              padding: 4,
              listStyle: "none",
            }}
          >
            {matches.map((m) => (
              <li key={m.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={false}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => pick(m)}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "8px 10px",
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    borderRadius: 6,
                    fontSize: 14,
                  }}
                >
                  <div style={{ fontWeight: 600 }}>{m.name}</div>
                  <div style={{ fontSize: 12, color: "#666" }}>
                    SFDC {m.id}
                    {m.website ? ` · ${m.website}` : ""}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        ) : null}
        {loading ? (
          <span style={{ position: "absolute", right: 12, top: 12, fontSize: 12, color: "#888" }}>…</span>
        ) : null}
      </div>
      {helpText ? <p className="help-text">{helpText}</p> : null}
    </div>
  );
}
