from pydantic import BaseModel
from typing import Optional, List, Dict


class TranscriptLine(BaseModel):
    """A single line from the transcript"""
    timestamp: Optional[str] = None
    speaker: Optional[str] = None
    text: str


class AnalyzeRequest(BaseModel):
    """Request to analyze a transcript"""
    transcript: str
    sa_name: Optional[str] = None  # Manual override for SA identification


class ActionableInsight(BaseModel):
    """A single actionable insight with specific recommendations"""
    category: str  # e.g., "Problem Identification", "Differentiation", etc.
    severity: str  # "critical", "important", "minor"
    timestamp: Optional[str] = None
    what_happened: str  # What the SA did/said
    why_it_matters: str  # Business impact
    better_approach: str  # Specific alternative
    example_phrasing: Optional[str] = None  # Exact words they could use


class CommandOfMessageScore(BaseModel):
    """Scoring for Command of the Message framework"""
    problem_identification: float  # 1-10 (can be decimal like 6.5)
    differentiation: float  # 1-10 (can be decimal like 6.5)
    proof_evidence: float  # 1-10 (can be decimal like 6.5)
    required_capabilities: float  # 1-10 (can be decimal like 6.5)


class SAPerformanceMetrics(BaseModel):
    """Additional SA-specific metrics"""
    technical_depth: float  # 1-10 (can be decimal like 6.5)
    discovery_quality: float  # 1-10 (can be decimal like 6.5)
    active_listening: float  # 1-10 (can be decimal like 6.5)
    value_articulation: float  # 1-10 (can be decimal like 6.5)


class AnalysisResult(BaseModel):
    """Complete analysis results"""
    sa_identified: str  # Who we identified as the SA
    sa_confidence: str  # "high", "medium", "low"
    call_summary: str
    overall_score: float  # 1-10
    command_scores: CommandOfMessageScore
    sa_metrics: SAPerformanceMetrics
    top_insights: List[ActionableInsight]
    strengths: List[str]
    improvement_areas: List[str]
    key_moments: List[Dict[str, str]]  # timestamp, description
