"use client";

import type { ShareableTab } from "@/lib/shareableUrl";

type StageRailProps = {
  activeTab: ShareableTab;
  onSelectTab: (tab: ShareableTab) => void;
};

function stageClassForTab(tab: ShareableTab): string {
  if (tab === "hypothesis") return "stage1";
  if (tab === "demo" || tab === "gong") return "stage2";
  if (tab === "pocpot") return "stage3";
  if (tab === "prospect") return "stage4";
  return "stage5";
}

export default function SalesStageRail({ activeTab, onSelectTab }: StageRailProps) {
  const activeStage = stageClassForTab(activeTab);

  return (
    <div className="stage-rail-wrapper">
      <div className="stage-rail-eyebrow">Sales Stage</div>
      <ol className="stage-rail" role="tablist" aria-label="Sales stage">
        <li className="stage-rail-item">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "hypothesis"}
            className={`stage-card stage-card-single${activeStage === "stage1" ? " active" : ""}`}
            data-stage="stage1"
            onClick={() => onSelectTab("hypothesis")}
          >
            <div className="stage-card-eyebrow">
              <span className="stage-card-number">Stage 1</span>
              <span className="stage-card-divider" aria-hidden="true">
                &bull;
              </span>
              <span className="stage-card-name">Engaged</span>
            </div>
            <div className="stage-card-tool-name">{"\u{1F52C} Hypothesis Research"}</div>
            <div className="stage-card-blurb">
              First touch. Pull AI/ML signals before the first call.
            </div>
          </button>
          <span className="stage-rail-arrow" aria-hidden="true">
            &rarr;
          </span>
        </li>

        <li className="stage-rail-item">
          <div
            className={`stage-card stage-card-multi${activeStage === "stage2" ? " active" : ""}`}
            data-stage="stage2"
          >
            <div className="stage-card-eyebrow">
              <span className="stage-card-number">Stage 2</span>
              <span className="stage-card-divider" aria-hidden="true">
                &bull;
              </span>
              <span className="stage-card-name">Qualification</span>
            </div>
            <div className="stage-card-tool-list" role="tablist" aria-label="Qualification tools">
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === "demo"}
                className={`stage-tool-button${activeTab === "demo" ? " active" : ""}`}
                onClick={() => onSelectTab("demo")}
              >
                {"\u{1F3AF} Custom Demo Builder"}
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === "gong"}
                className={`stage-tool-button${activeTab === "gong" ? " active" : ""}`}
                onClick={() => onSelectTab("gong")}
              >
                {"\u{1F4DE} Single Call Analysis"}
              </button>
            </div>
            <div className="stage-card-blurb">
              Validate fit. Build a tailored demo and dissect the discovery call.
            </div>
          </div>
          <span className="stage-rail-arrow" aria-hidden="true">
            &rarr;
          </span>
        </li>

        <li className="stage-rail-item">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "pocpot"}
            className={`stage-card stage-card-single${activeStage === "stage3" ? " active" : ""}`}
            data-stage="stage3"
            onClick={() => onSelectTab("pocpot")}
          >
            <div className="stage-card-eyebrow">
              <span className="stage-card-number">Stage 3</span>
              <span className="stage-card-divider" aria-hidden="true">
                &bull;
              </span>
              <span className="stage-card-name">Pre-PoC</span>
            </div>
            <div className="stage-card-tool-name">{"\u{1F4DD} PoC / PoT Document"}</div>
            <div className="stage-card-blurb">
              Stand up the PoC scope. Draft the PoT / PoC doc and align on success criteria.
            </div>
          </button>
          <span className="stage-rail-arrow" aria-hidden="true">
            &rarr;
          </span>
        </li>

        <li className="stage-rail-item">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "prospect"}
            className={`stage-card stage-card-single${activeStage === "stage4" ? " active" : ""}`}
            data-stage="stage4"
            onClick={() => onSelectTab("prospect")}
          >
            <div className="stage-card-eyebrow">
              <span className="stage-card-number">Stage 4</span>
              <span className="stage-card-divider" aria-hidden="true">
                &bull;
              </span>
              <span className="stage-card-name">PoC</span>
            </div>
            <div className="stage-card-tool-name">{"\u{1F4CA} Prospect Overview"}</div>
            <div className="stage-card-blurb">
              PoC in flight. 360&deg; view of the buyer and the deal as you push to close.
            </div>
          </button>
          <span className="stage-rail-arrow" aria-hidden="true">
            &rarr;
          </span>
        </li>

        <li className="stage-rail-item">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "transition"}
            className={`stage-card stage-card-single${activeStage === "stage5" ? " active" : ""}`}
            data-stage="stage5"
            onClick={() => onSelectTab("transition")}
          >
            <div className="stage-card-eyebrow">
              <span className="stage-card-number">Stage 5</span>
              <span className="stage-card-divider" aria-hidden="true">
                &bull;
              </span>
              <span className="stage-card-name">Closed Won</span>
            </div>
            <div className="stage-card-tool-name">{"\u{1F501} Transition to CS"}</div>
            <div className="stage-card-blurb">
              Ship a clean Knowledge Transfer doc into Customer Success.
            </div>
          </button>
        </li>
      </ol>
    </div>
  );
}

export function stageBodyCopy(tab: ShareableTab): { eyebrow: string; toolName: string } {
  const map: Record<ShareableTab, { eyebrow: string; toolName: string }> = {
    hypothesis: { eyebrow: "Stage 1 • Engaged", toolName: "\u{1F52C} Hypothesis Research" },
    prospect: { eyebrow: "Stage 4 • PoC", toolName: "\u{1F4CA} Prospect Overview" },
    demo: { eyebrow: "Stage 2 • Qualification", toolName: "\u{1F3AF} Custom Demo Builder" },
    gong: { eyebrow: "Stage 2 • Qualification", toolName: "\u{1F4DE} Single Call Analysis" },
    pocpot: { eyebrow: "Stage 3 • Pre-PoC", toolName: "\u{1F4DD} PoC / PoT Document" },
    transition: { eyebrow: "Stage 5 • Closed Won", toolName: "\u{1F501} Transition to CS" },
  };
  return map[tab];
}
