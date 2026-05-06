"use client";

import { useToast, type Toast, type ToastType } from "./ToastContext";

const ICONS: Record<ToastType, string> = {
  success: "✓",
  error: "✕",
  warning: "⚠",
  info: "ℹ",
};

const COLORS: Record<ToastType, { bg: string; border: string; text: string; icon: string }> = {
  success: { bg: "#d4edda", border: "#28a745", text: "#155724", icon: "#28a745" },
  error: { bg: "#f8d7da", border: "#dc3545", text: "#721c24", icon: "#dc3545" },
  warning: { bg: "#fff3cd", border: "#ffc107", text: "#856404", icon: "#856404" },
  info: { bg: "#d1ecf1", border: "#17a2b8", text: "#0c5460", icon: "#17a2b8" },
};

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const colors = COLORS[toast.type];

  return (
    <div
      className="toast-item"
      style={{
        background: colors.bg,
        borderLeft: `4px solid ${colors.border}`,
        color: colors.text,
      }}
    >
      <span className="toast-icon" style={{ color: colors.icon }}>
        {ICONS[toast.type]}
      </span>
      <span className="toast-message">{toast.message}</span>
      <button
        className="toast-close"
        onClick={onDismiss}
        aria-label="Dismiss"
        style={{ color: colors.text }}
      >
        ×
      </button>

      <style jsx>{`
        .toast-item {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          padding: 14px 16px;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          animation: slideIn 0.3s ease-out;
          min-width: 300px;
          max-width: 450px;
        }
        @keyframes slideIn {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        .toast-icon {
          font-size: 1.2em;
          font-weight: bold;
          flex-shrink: 0;
        }
        .toast-message {
          flex: 1;
          line-height: 1.4;
          word-break: break-word;
        }
        .toast-close {
          background: none;
          border: none;
          font-size: 1.5em;
          cursor: pointer;
          padding: 0;
          line-height: 1;
          opacity: 0.7;
          flex-shrink: 0;
        }
        .toast-close:hover {
          opacity: 1;
        }
      `}</style>
    </div>
  );
}

export function ToastContainer() {
  const { toasts, removeToast } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={() => removeToast(toast.id)} />
      ))}

      <style jsx>{`
        .toast-container {
          position: fixed;
          top: 20px;
          right: 20px;
          z-index: 9999;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
      `}</style>
    </div>
  );
}
