import os
import time
from contextlib import contextmanager
from typing import Iterable, List

from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

from .models import SpanPreview


def _build_provider(service_name: str, project_id: str | None = None) -> TracerProvider:
    resource_attrs = {"service.name": service_name}
    if project_id:
        resource_attrs["openinference.project.name"] = project_id
    resource = Resource.create(resource_attrs)
    provider = TracerProvider(resource=resource)
    return provider


def _exporter(api_key: str, endpoint: str, space_id: str | None = None) -> OTLPSpanExporter:
    headers = {"api_key": api_key}
    if space_id:
        headers["space_id"] = space_id
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _emit_trace_tree(
    tracer: trace.Tracer,
    trace_spans: List[SpanPreview],
) -> int:
    """
    Emit a single trace tree with proper parent-child context propagation.
    trace_spans[0] is the root; subsequent spans are children.
    """
    exported = 0
    # Map from our preview span_id -> live OTel span, so children can
    # look up their parent's context.
    span_map = {}

    for span_preview in trace_spans:
        # Determine parent context
        parent_ctx = None
        if span_preview.parent_span_id and span_preview.parent_span_id in span_map:
            parent_span = span_map[span_preview.parent_span_id]
            parent_ctx = trace.set_span_in_context(parent_span)

        ctx = parent_ctx or context.get_current()
        otel_span = tracer.start_span(
            span_preview.name,
            context=ctx,
            attributes=span_preview.attributes,
        )
        span_map[span_preview.span_id] = otel_span

        for event in span_preview.events:
            otel_span.add_event(event["name"], attributes=event.get("attributes"))
        if span_preview.status == "ERROR":
            otel_span.set_status(Status(status_code=StatusCode.ERROR))

        exported += 1

    # End spans in reverse order (children first, then root)
    for span_preview in reversed(trace_spans):
        if span_preview.span_id in span_map:
            span_map[span_preview.span_id].end()

    return exported


def export_traces(
    traces: List[List[SpanPreview]],
    api_key: str,
    endpoint: str,
    space_id: str | None = None,
    project_id: str | None = None,
) -> int:
    """
    Export multiple trace trees to Arize via OTLP/HTTP.
    Each trace is a list of SpanPreview with proper parent_span_id references.
    Returns total count of exported spans.
    """
    provider = _build_provider(service_name="arize-demo-trace-service", project_id=project_id)
    exporter = _exporter(api_key, endpoint, space_id)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)

    total_exported = 0
    for trace_spans in traces:
        total_exported += _emit_trace_tree(tracer, trace_spans)

    provider.shutdown()
    return total_exported


# Keep backward compat alias
def export_spans(
    spans: Iterable[SpanPreview],
    api_key: str,
    endpoint: str,
    space_id: str | None = None,
    project_id: str | None = None,
) -> int:
    """Legacy flat export - wraps all spans as one trace."""
    return export_traces(
        [list(spans)],
        api_key=api_key,
        endpoint=endpoint,
        space_id=space_id,
        project_id=project_id,
    )


def resolve_endpoint() -> str:
    return os.getenv("ARIZE_OTLP_ENDPOINT", "https://otlp.arize.com/v1/traces")
