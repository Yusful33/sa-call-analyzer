"use client";

import { useEffect, useState, useCallback, useRef } from "react";

type EasterEggState = {
  konamiActive: boolean;
  innerPeaceMode: boolean;
  serenityActive: boolean;
};

export default function EasterEggs({
  onInnerPeaceToggle,
}: {
  onInnerPeaceToggle?: (active: boolean) => void;
}) {
  const [state, setState] = useState<EasterEggState>({
    konamiActive: false,
    innerPeaceMode: false,
    serenityActive: false,
  });
  
  const sKeyCount = useRef(0);
  const sKeyTimer = useRef<NodeJS.Timeout>();
  const typedKeys = useRef<string>("");
  
  const triggerKonami = useCallback(() => {
    setState(s => ({ ...s, konamiActive: true }));
    setTimeout(() => setState(s => ({ ...s, konamiActive: false })), 5000);
  }, []);
  
  const triggerSerenity = useCallback(() => {
    setState(s => ({ ...s, serenityActive: true }));
    setTimeout(() => setState(s => ({ ...s, serenityActive: false })), 4000);
  }, []);
  
  const toggleInnerPeace = useCallback(() => {
    setState(s => {
      const newState = !s.innerPeaceMode;
      onInnerPeaceToggle?.(newState);
      return { ...s, innerPeaceMode: newState };
    });
  }, [onInnerPeaceToggle]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      
      // "S" key 3 times triggers sparkle celebration
      if (key === "s") {
        sKeyCount.current++;
        if (sKeyTimer.current) clearTimeout(sKeyTimer.current);
        
        if (sKeyCount.current >= 3) {
          triggerKonami();
          sKeyCount.current = 0;
        } else {
          sKeyTimer.current = setTimeout(() => {
            sKeyCount.current = 0;
          }, 1500);
        }
      }
      
      // "zen" triggers lotus animation
      if (key.length === 1 && /[a-z]/.test(key)) {
        typedKeys.current += key;
        if (typedKeys.current.length > 10) {
          typedKeys.current = typedKeys.current.slice(-10);
        }
        if (typedKeys.current.includes("zen")) {
          triggerSerenity();
          typedKeys.current = "";
        }
      }
    };
    
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [triggerKonami, triggerSerenity]);

  return (
    <>
      {/* Konami Code Easter Egg - Sparkle Celebration */}
      {state.konamiActive && (
        <div className="easter-egg-konami">
          <div className="konami-message">
            <span className="konami-emoji">✨🧘‍♀️✨</span>
            <p className="konami-text">You found the secret!</p>
            <p className="konami-quote">"In stillness, all things become clear."</p>
          </div>
          {[...Array(30)].map((_, i) => (
            <div
              key={i}
              className="sparkle"
              style={{
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 2}s`,
                animationDuration: `${2 + Math.random() * 2}s`,
              }}
            />
          ))}
        </div>
      )}
      
      {/* Serenity Easter Egg - Floating Lotus */}
      {state.serenityActive && (
        <div className="easter-egg-serenity">
          {[...Array(12)].map((_, i) => (
            <div
              key={i}
              className="lotus"
              style={{
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 1.5}s`,
                fontSize: `${1.5 + Math.random() * 1.5}rem`,
              }}
            >
              🪷
            </div>
          ))}
          <div className="serenity-message">
            <span>🌸 Serenity Achieved 🌸</span>
          </div>
        </div>
      )}
      
      {/* Inner Peace Mode indicator */}
      {state.innerPeaceMode && (
        <div className="inner-peace-badge">
          🌈 Inner Peace Mode Active
        </div>
      )}
    </>
  );
}

export function useEmojiClick(threshold = 7) {
  const clickCount = useRef(0);
  const clickTimer = useRef<NodeJS.Timeout>();
  const [triggered, setTriggered] = useState(false);
  
  const handleClick = useCallback(() => {
    clickCount.current++;
    
    if (clickTimer.current) {
      clearTimeout(clickTimer.current);
    }
    
    if (clickCount.current >= threshold) {
      setTriggered(t => !t);
      clickCount.current = 0;
      return;
    }
    
    clickTimer.current = setTimeout(() => {
      clickCount.current = 0;
    }, 2000);
  }, [threshold]);
  
  return { handleClick, triggered };
}
