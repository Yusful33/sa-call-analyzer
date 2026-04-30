"""
Fixed Span Processor - Properly integrates with the export chain.

The key fix: Instead of adding a standalone processor that replaces the exporter,
we need to either:
1. Create a composite processor with both cleaning and exporting, OR
2. Access the internal processor chain and insert ourselves at the right position

Solution: We'll modify spans BEFORE the BatchSpanProcessor exports them.
"""

import json
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan


class CleaningSpanProcessor(SpanProcessor):
    """
    Span processor that cleans attributes before export.

    This processor should be added to the tracer provider ALONGSIDE
    the existing BatchSpanProcessor, not replacing it.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.processed_count = 0
        self.error_count = 0
        print(f"✅ CleaningSpanProcessor initialized (enabled: {enabled})")

    def on_start(self, span, parent_context=None):
        """Nothing to do on start."""
        pass

    def on_end(self, span: ReadableSpan):
        """
        Clean span attributes BEFORE export.

        Critical: This is called before BatchSpanProcessor.on_end(),
        so our modifications will be included in the export.
        """
        if not self.enabled:
            return

        try:
            # Only process spans we care about
            if not self._should_process(span):
                return

            # Get attributes safely
            attrs = self._get_mutable_attributes(span)
            if not attrs:
                return

            # Get span kind
            span_kind = str(attrs.get("openinference.span.kind", ""))

            # Process based on span kind
            if span_kind == "llm":
                self._clean_llm_span(attrs, span.name)
            elif span_kind == "agent":
                self._clean_agent_span(attrs, span.name)

            self.processed_count += 1

        except Exception as e:
            self.error_count += 1
            print(f"⚠️  Processor error on span {getattr(span, 'name', 'unknown')}: {e}")
            # Don't re-raise - let the span export anyway

    def _should_process(self, span: ReadableSpan) -> bool:
        """Only process specific spans to minimize risk."""
        if not hasattr(span, 'name') or not span.name:
            return False
        if span.name.startswith("otel."):
            return False
        return True

    def _get_mutable_attributes(self, span: ReadableSpan):
        """Get attributes in a way we can modify."""
        try:
            if hasattr(span, '_attributes'):
                attrs = span._attributes
                if hasattr(attrs, '__setitem__'):
                    return attrs
            if hasattr(span, 'attributes'):
                attrs = span.attributes
                if hasattr(attrs, '__setitem__'):
                    return attrs
        except Exception as e:
            print(f"⚠️  Could not get attributes: {e}")
        return None

    def _clean_llm_span(self, attrs: dict, span_name: str):
        """Clean LLM span output - extract JSON from verbose text."""
        try:
            if "output.value" not in attrs:
                return

            original = attrs.get("output.value")
            if not isinstance(original, str) or len(original) < 100:
                return

            # Try to extract JSON
            cleaned = self._extract_json(original)

            # Only update if we actually cleaned something
            if cleaned and cleaned != original and len(cleaned) < len(original):
                try:
                    json.loads(cleaned) if isinstance(cleaned, str) else None
                    attrs["output.value"] = cleaned
                    attrs["arize.cleaned"] = True
                except:
                    pass

        except Exception as e:
            print(f"⚠️  Error cleaning LLM span {span_name}: {e}")

    def _clean_agent_span(self, attrs: dict, span_name: str):
        """Clean agent span - truncate large inputs/outputs."""
        try:
            if "input.value" in attrs:
                original = attrs.get("input.value")
                if isinstance(original, str) and len(original) > 5000:
                    attrs["input.value"] = original[:5000] + "\n... [truncated]"
                    attrs["arize.input_truncated"] = True

            if "output.value" in attrs:
                original = attrs.get("output.value")
                if isinstance(original, str) and len(original) > 10000:
                    attrs["output.value"] = original[:10000] + "\n... [truncated]"
                    attrs["arize.output_truncated"] = True

        except Exception as e:
            print(f"⚠️  Error cleaning agent span {span_name}: {e}")

    def _extract_json(self, text: str) -> str:
        """Extract clean JSON from verbose text."""
        if not text or len(text) < 10:
            return text

        try:
            start = text.find('{')
            end = text.rfind('}')

            if start >= 0 and end > start:
                potential_json = text[start:end + 1]
                parsed = json.loads(potential_json)

                # If it has a "raw" key, extract just that
                if isinstance(parsed, dict) and "raw" in parsed:
                    raw_data = parsed["raw"]
                    if raw_data:
                        return json.dumps(raw_data, indent=2)

                return potential_json

        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"⚠️  JSON extraction error: {e}")

        return text

    def shutdown(self):
        """Cleanup."""
        print(f"CleaningSpanProcessor stats: processed={self.processed_count}, errors={self.error_count}")

    def force_flush(self, timeout_millis: int = 30000):
        """No-op for processor."""
        return True
