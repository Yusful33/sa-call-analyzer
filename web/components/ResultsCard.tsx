"use client";

export default function ResultsCard({
  visible,
  html,
}: {
  visible: boolean;
  html: string;
}) {
  if (!visible || !html) return null;
  return (
    <div
      className="card results show"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
