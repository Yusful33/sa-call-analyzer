"use client";

import { createContext, useContext, useState, useCallback, ReactNode, useEffect } from "react";

type Achievement = {
  id: string;
  emoji: string;
  title: string;
  message: string;
  timestamp: number;
};

type AchievementState = {
  prospectRuns: number;
  gongSuccesses: number;
  demoBuilds: number;
  tabsUsed: Set<string>;
  achievements: Achievement[];
  resultsScrolled: boolean;
};

type AchievementContextType = {
  trackProspectRun: () => void;
  trackGongSuccess: () => void;
  trackDemoBuild: () => void;
  trackTabUsed: (tabId: string) => void;
  trackResultsScrolled: () => void;
  state: AchievementState;
  activeAchievement: Achievement | null;
  dismissAchievement: () => void;
  showBottomMessage: boolean;
};

const AchievementContext = createContext<AchievementContextType | null>(null);

export function useAchievements() {
  const ctx = useContext(AchievementContext);
  if (!ctx) throw new Error("useAchievements must be used within AchievementProvider");
  return ctx;
}

const ACHIEVEMENTS = {
  powerUser: {
    id: "power-user",
    emoji: "🎯",
    title: "Power User!",
    message: "You've researched 3 prospects! You're on a roll.",
  },
  callAnalyst: {
    id: "call-analyst",
    emoji: "📞",
    title: "Call Analyst!",
    message: "Your first successful call analysis. Great insights await!",
  },
  demoBuilder: {
    id: "demo-builder",
    emoji: "🏗️",
    title: "Demo Architect!",
    message: "You built your first custom demo. Time to impress!",
  },
  explorer: {
    id: "explorer",
    emoji: "🗺️",
    title: "Explorer!",
    message: "You've explored 3+ different features. Curiosity unlocked!",
  },
  deepDiver: {
    id: "deep-diver",
    emoji: "🤿",
    title: "Deep Diver!",
    message: "You scrolled through all the results. Thoroughness pays off!",
  },
};

export function AchievementProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AchievementState>({
    prospectRuns: 0,
    gongSuccesses: 0,
    demoBuilds: 0,
    tabsUsed: new Set(),
    achievements: [],
    resultsScrolled: false,
  });
  const [activeAchievement, setActiveAchievement] = useState<Achievement | null>(null);
  const [showBottomMessage, setShowBottomMessage] = useState(false);

  const addAchievement = useCallback((achievement: Omit<Achievement, "timestamp">) => {
    const newAchievement = { ...achievement, timestamp: Date.now() };
    setState(s => ({
      ...s,
      achievements: [...s.achievements, newAchievement],
    }));
    setActiveAchievement(newAchievement);
  }, []);

  const hasAchievement = useCallback((id: string) => {
    return state.achievements.some(a => a.id === id);
  }, [state.achievements]);

  const trackProspectRun = useCallback(() => {
    setState(s => {
      const newCount = s.prospectRuns + 1;
      return { ...s, prospectRuns: newCount };
    });
  }, []);

  useEffect(() => {
    if (state.prospectRuns === 3 && !hasAchievement("power-user")) {
      addAchievement(ACHIEVEMENTS.powerUser);
    }
  }, [state.prospectRuns, hasAchievement, addAchievement]);

  const trackGongSuccess = useCallback(() => {
    setState(s => {
      const newCount = s.gongSuccesses + 1;
      return { ...s, gongSuccesses: newCount };
    });
  }, []);

  useEffect(() => {
    if (state.gongSuccesses === 1 && !hasAchievement("call-analyst")) {
      addAchievement(ACHIEVEMENTS.callAnalyst);
    }
  }, [state.gongSuccesses, hasAchievement, addAchievement]);

  const trackDemoBuild = useCallback(() => {
    setState(s => {
      const newCount = s.demoBuilds + 1;
      return { ...s, demoBuilds: newCount };
    });
  }, []);

  useEffect(() => {
    if (state.demoBuilds === 1 && !hasAchievement("demo-builder")) {
      addAchievement(ACHIEVEMENTS.demoBuilder);
    }
  }, [state.demoBuilds, hasAchievement, addAchievement]);

  const trackTabUsed = useCallback((tabId: string) => {
    setState(s => {
      const newTabs = new Set(s.tabsUsed);
      newTabs.add(tabId);
      return { ...s, tabsUsed: newTabs };
    });
  }, []);

  useEffect(() => {
    if (state.tabsUsed.size >= 3 && !hasAchievement("explorer")) {
      addAchievement(ACHIEVEMENTS.explorer);
    }
  }, [state.tabsUsed.size, hasAchievement, addAchievement]);

  const trackResultsScrolled = useCallback(() => {
    if (!state.resultsScrolled) {
      setState(s => ({ ...s, resultsScrolled: true }));
      setShowBottomMessage(true);
      if (!hasAchievement("deep-diver")) {
        addAchievement(ACHIEVEMENTS.deepDiver);
      }
    }
  }, [state.resultsScrolled, hasAchievement, addAchievement]);

  const dismissAchievement = useCallback(() => {
    setActiveAchievement(null);
  }, []);

  return (
    <AchievementContext.Provider
      value={{
        trackProspectRun,
        trackGongSuccess,
        trackDemoBuild,
        trackTabUsed,
        trackResultsScrolled,
        state,
        activeAchievement,
        dismissAchievement,
        showBottomMessage,
      }}
    >
      {children}
    </AchievementContext.Provider>
  );
}

export function AchievementToast() {
  const { activeAchievement, dismissAchievement } = useAchievements();

  useEffect(() => {
    if (activeAchievement) {
      const timer = setTimeout(dismissAchievement, 5000);
      return () => clearTimeout(timer);
    }
  }, [activeAchievement, dismissAchievement]);

  if (!activeAchievement) return null;

  return (
    <div className="achievement-toast" onClick={dismissAchievement}>
      <div className="achievement-toast-content">
        <span className="achievement-emoji">{activeAchievement.emoji}</span>
        <div className="achievement-text">
          <strong className="achievement-title">{activeAchievement.title}</strong>
          <p className="achievement-message">{activeAchievement.message}</p>
        </div>
      </div>
      <div className="achievement-confetti">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="confetti-piece"
            style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 0.5}s`,
              backgroundColor: ["#667eea", "#764ba2", "#ffd700", "#ff6b6b", "#6bcb77"][
                Math.floor(Math.random() * 5)
              ],
            }}
          />
        ))}
      </div>
    </div>
  );
}

export function AchievementBadges() {
  const { state } = useAchievements();

  if (state.achievements.length === 0) return null;

  return (
    <div className="achievement-badges">
      {state.achievements.map(a => (
        <span key={a.id} className="achievement-badge" title={a.title}>
          {a.emoji}
        </span>
      ))}
    </div>
  );
}

export function ResultsEndMessage() {
  const { showBottomMessage } = useAchievements();

  if (!showBottomMessage) return null;

  return (
    <div className="results-end-message">
      <span className="results-end-emoji">🎉</span>
      <p>You made it to the end! Thorough analysis FTW.</p>
    </div>
  );
}
