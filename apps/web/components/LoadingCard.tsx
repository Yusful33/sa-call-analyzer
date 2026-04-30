"use client";

export default function LoadingCard({
  visible,
  message,
}: {
  visible: boolean;
  message: string;
}) {
  if (!visible) return null;
  return (
    <div className="card loading">
      <div className="spinner" />
      <p>{message}</p>
      <p style={{ marginTop: 10, fontSize: "0.9em", color: "#666" }}>
        This may take 30-60 seconds
      </p>
    </div>
  );
}
