"use client";

import { useRef, useEffect, useMemo } from "react";
import { useAchievements, ResultsEndMessage } from "./UsageAchievements";

export default function ResultsCard({
  visible,
  html,
}: {
  visible: boolean;
  html: string;
}) {
  const sentinelRef = useRef<HTMLDivElement>(null);
  const { trackResultsScrolled } = useAchievements();

  useEffect(() => {
    if (!visible || !html || html.length <= 2000) return;
    const el = sentinelRef.current;
    if (!el) return;

    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          trackResultsScrolled();
        }
      },
      { root: null, rootMargin: "0px 0px 80px 0px", threshold: 0 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [visible, html, html.length, trackResultsScrolled]);

  // React 19 regression (facebook/react#31660): dangerouslySetInnerHTML uses
  // object identity instead of string equality, so the inner <div>'s innerHTML
  // is reassigned on every parent rerender. That clobbers any DOM mutation a
  // user just made — most visibly, native <details>/<summary> toggles inside
  // the rendered HTML get reset the instant the user clicks. Memoizing the
  // object keeps the same reference across renders when html is unchanged.
  const innerHtml = useMemo(() => ({ __html: html }), [html]);

  if (!visible || !html) return null;

  return (
    <div className="card results show results-scrollable">
      <div dangerouslySetInnerHTML={innerHtml} />
      <div ref={sentinelRef} aria-hidden="true" style={{ height: 1, width: "100%", pointerEvents: "none" }} />
      <ResultsEndMessage />
    </div>
  );
}
