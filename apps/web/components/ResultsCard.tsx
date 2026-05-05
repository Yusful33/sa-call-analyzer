"use client";

import { useRef, useEffect, useCallback } from "react";
import { useAchievements, ResultsEndMessage } from "./UsageAchievements";

export default function ResultsCard({
  visible,
  html,
}: {
  visible: boolean;
  html: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { trackResultsScrolled } = useAchievements();

  const handleScroll = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const scrolledToBottom =
      container.scrollHeight - container.scrollTop <= container.clientHeight + 50;

    if (scrolledToBottom && html.length > 2000) {
      trackResultsScrolled();
    }
  }, [html.length, trackResultsScrolled]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !visible) return;

    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, [visible, handleScroll]);

  if (!visible || !html) return null;

  return (
    <div ref={containerRef} className="card results show results-scrollable">
      <div dangerouslySetInnerHTML={{ __html: html }} />
      <ResultsEndMessage />
    </div>
  );
}
