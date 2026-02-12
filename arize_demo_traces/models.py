from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, RootModel


class UseCaseEnum(str, Enum):
    rag_search = "retrieval-augmented-search"
    multiturn_chatbot = "multiturn-chatbot-with-tools"
    text_to_sql = "text-to-sql-bi-agent"
    content_moderation = "content-moderation-pipeline"
    document_qa = "document-qa-ingestion"
    personalization = "personalization-recommendations"
    voicebot = "voicebot-speech-analytics"
    image_understanding = "image-understanding-captioning"
    code_assistant = "code-assistant-code-search"
    anomaly_detection = "anomaly-detection-alerting"
    custom = "custom"


USE_CASE_LABELS = {
    UseCaseEnum.rag_search: "Retrieval-augmented search",
    UseCaseEnum.multiturn_chatbot: "Multiturn chatbot with tools",
    UseCaseEnum.text_to_sql: "Text-to-SQL / BI agent",
    UseCaseEnum.content_moderation: "Content moderation pipeline",
    UseCaseEnum.document_qa: "Document Q&A / ingestion",
    UseCaseEnum.personalization: "Personalization / recommendations",
    UseCaseEnum.voicebot: "Voicebot / speech analytics",
    UseCaseEnum.image_understanding: "Image understanding / captioning",
    UseCaseEnum.code_assistant: "Code assistant / code search",
    UseCaseEnum.anomaly_detection: "Anomaly detection / alerting",
    UseCaseEnum.custom: "Custom",
}


class TechStack(BaseModel):
    architecture: List[str] = Field(default_factory=list, description="e.g., single-agent, multi-agent")
    frameworks: List[str] = Field(
        default_factory=list,
        description="LangChain, Vercel AI SDK, Hugging Face smolagents, custom",
    )
    providers: List[str] = Field(
        default_factory=list,
        description="OpenAI, Anthropic, Amazon Bedrock, Vertex AI, Mistral, custom",
    )
    languages: List[str] = Field(default_factory=list, description="e.g., Python, Node, Java, other")
    extras: Optional[str] = Field(default=None, description="freeform notes")


class GenerateRequest(BaseModel):
    use_case: UseCaseEnum = Field(description="Selected use case")
    custom_use_case: Optional[str] = Field(default=None, description="If use_case=custom, describe it here")
    tech_stack: TechStack = Field(default_factory=TechStack)
    project_id: Optional[str] = Field(default=None, description="Optional Arize project name/id to tag spans")
    preview_only: bool = True
    send_to_arize: bool = False
    traces_per_variant: int = Field(default=1, ge=1, le=10)


class SpanPreview(BaseModel):
    span_id: str
    parent_span_id: Optional[str]
    name: str
    status: str
    start_time: str
    end_time: str
    attributes: dict
    events: list


class GenerateResponse(BaseModel):
    use_case: str
    spans: list
    sent_to_arize: bool = False
    message: Optional[str] = None
    arize_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"


class SpanPreviewList(RootModel[List[SpanPreview]]):
    """Wrapper to keep FastAPI response docs tidy."""
