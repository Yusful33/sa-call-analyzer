from pydantic import BaseModel
from typing import Optional, List, Dict


class TranscriptLine(BaseModel):
    """A single line from the transcript"""
    timestamp: Optional[str] = None
    speaker: Optional[str] = None
    text: str


class AnalyzeRequest(BaseModel):
    """Request to analyze a transcript"""
    transcript: Optional[str] = None  # Manual transcript text
    gong_url: Optional[str] = None  # Gong call URL (alternative to transcript)
    sa_name: Optional[str] = None  # Manual override for SA identification

    def model_post_init(self, __context):
        """Validate that either transcript or gong_url is provided."""
        if not self.transcript and not self.gong_url:
            raise ValueError("Either 'transcript' or 'gong_url' must be provided")
        if self.transcript and self.gong_url:
            raise ValueError("Provide either 'transcript' or 'gong_url', not both")


class ActionableInsight(BaseModel):
    """A single actionable insight with specific recommendations"""
    category: str  # e.g., "Problem Identification", "Differentiation", etc.
    severity: str  # "critical", "important", "minor"
    timestamp: Optional[str] = None
    conversation_snippet: Optional[str] = None  # Brief excerpt from the actual conversation
    what_happened: str  # What the SA did/said
    why_it_matters: str  # Business impact
    better_approach: str  # Specific alternative
    example_phrasing: Optional[str] = None  # Exact words they could use


class CommandOfMessageScore(BaseModel):
    """Scoring for Command of the Message framework (optional fields)"""
    problem_identification: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    differentiation: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    proof_evidence: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    required_capabilities: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)


class SAPerformanceMetrics(BaseModel):
    """Additional SA-specific metrics (optional fields)"""
    technical_depth: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    discovery_quality: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    active_listening: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    value_articulation: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)


class AnalysisResult(BaseModel):
    """Complete analysis results"""
    sa_identified: str  # Who we identified as the SA
    sa_confidence: str  # "high", "medium", "low"
    call_summary: str
    overall_score: Optional[float] = 7.0  # 1-10 (optional, defaults to 7.0)
    command_scores: CommandOfMessageScore
    sa_metrics: SAPerformanceMetrics
    top_insights: List[ActionableInsight]
    strengths: List[str]
    improvement_areas: List[str]
    key_moments: List[Dict[str, str]]  # timestamp, description
