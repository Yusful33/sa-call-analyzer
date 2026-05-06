"use client";

import { useRef, useEffect, useCallback, useMemo } from "react";
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

  // React 19 regression (facebook/react#31660): dangerouslySetInnerHTML uses
  // object identity instead of string equality, so the inner <div>'s innerHTML
  // is reassigned on every parent rerender. That clobbers any DOM mutation a
  // user just made — most visibly, native <details>/<summary> toggles inside
  // the rendered HTML get reset the instant the user clicks. Memoizing the
  // object keeps the same reference across renders when html is unchanged.
  const innerHtml = useMemo(() => ({ __html: html }), [html]);

  if (!visible || !html) return null;

  return (
    <div ref={containerRef} className="card results show results-scrollable">
      <div dangerouslySetInnerHTML={innerHtml} />
      <ResultsEndMessage />
    </div>
  );
}
