"""
Minimal, Bulletproof Span Processor - Only Cleans What's Needed

The key to making this work:
1. Only modify specific spans (not all)
2. Only modify string attributes (no complex types)
3. Validate everything before modifying
4. Fail fast if anything looks wrong
5. Keep processing under 1ms per span
"""

import json
import re
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan


class MinimalCleaningProcessor(SpanProcessor):
    """
    Ultra-defensive span processor that only cleans specific spans.

    Strategy: Do the minimum necessary to get clean output, nothing more.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.processed_count = 0
        self.error_count = 0

        print(f"✅ MinimalCleaningProcessor initialized (enabled: {enabled})")

    def on_start(self, span, parent_context=None):
        """Nothing to do on start."""
        pass

    def on_end(self, span: ReadableSpan):
        """
        Called when span ends. Only modify if safe.
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
        # Only process named spans (not auto-generated)
        if not hasattr(span, 'name') or not span.name:
            return False

        # Skip internal OpenTelemetry spans
        if span.name.startswith("otel."):
            return False

        return True

    def _get_mutable_attributes(self, span: ReadableSpan):
        """
        Get attributes in a way we can modify.

        The trick: Some OpenTelemetry implementations use BoundedAttributes,
        others use plain dicts. We need to handle both.
        """
        try:
            # Try _attributes first (most common)
            if hasattr(span, '_attributes'):
                attrs = span._attributes
                # Verify it's dict-like
                if hasattr(attrs, '__setitem__'):
                    return attrs

            # Try .attributes (some versions)
            if hasattr(span, 'attributes'):
                attrs = span.attributes
                if hasattr(attrs, '__setitem__'):
                    return attrs

        except Exception as e:
            print(f"⚠️  Could not get attributes: {e}")

        return None

    def _clean_llm_span(self, attrs: dict, span_name: str):
        """
        Clean LLM span output - extract JSON from verbose text.

        Key insight: LLM outputs often look like:
        "verbose instructions...\n\n{\"raw\": {...}}\n\npydantic: null"

        We want just: {...}
        """
        try:
            # Only process output.value
            if "output.value" not in attrs:
                return

            original = attrs.get("output.value")

            # Must be a string to process
            if not isinstance(original, str):
                return

            # Must be long enough to have verbose content
            if len(original) < 100:
                return

            # Try to extract JSON
            cleaned = self._extract_json(original)

            # Only update if we actually cleaned something
            if cleaned and cleaned != original and len(cleaned) < len(original):
                # Validate it's JSON-serializable
                try:
                    json.loads(cleaned) if isinstance(cleaned, str) else None
                    # Safe to update
                    attrs["output.value"] = cleaned
                    attrs["arize.cleaned"] = True
                except:
                    # Not valid JSON, don't update
                    pass

        except Exception as e:
            print(f"⚠️  Error cleaning LLM span {span_name}: {e}")

    def _clean_agent_span(self, attrs: dict, span_name: str):
        """
        Clean agent span (like crew_analysis_execution).

        For these, we just truncate the input/output to reasonable sizes.
        """
        try:
            # Truncate input if huge
            if "input.value" in attrs:
                original = attrs.get("input.value")
                if isinstance(original, str) and len(original) > 5000:
                    # Keep first 5000 chars
                    attrs["input.value"] = original[:5000] + "\n... [truncated]"
                    attrs["arize.input_truncated"] = True

            # Truncate output if huge
            if "output.value" in attrs:
                original = attrs.get("output.value")
                if isinstance(original, str) and len(original) > 10000:
                    # Keep first 10000 chars
                    attrs["output.value"] = original[:10000] + "\n... [truncated]"
                    attrs["arize.output_truncated"] = True

        except Exception as e:
            print(f"⚠️  Error cleaning agent span {span_name}: {e}")

    def _extract_json(self, text: str) -> str:
        """
        Extract clean JSON from verbose text.

        Common patterns:
        1. Task description + JSON + extra stuff
        2. {"raw": {...}, "pydantic": null, ...}
        3. ```json {...} ```
        """
        if not text or len(text) < 10:
            return text

        # Pattern 1: Find first { to last }
        try:
            start = text.find('{')
            end = text.rfind('}')

            if start >= 0 and end > start:
                potential_json = text[start:end + 1]

                # Validate it's JSON
                parsed = json.loads(potential_json)

                # If it has a "raw" key, extract just that
                if isinstance(parsed, dict) and "raw" in parsed:
                    raw_data = parsed["raw"]
                    if raw_data:
                        return json.dumps(raw_data, indent=2)

                # Otherwise return the cleaned JSON
                return potential_json

        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"⚠️  JSON extraction error: {e}")

        # If extraction fails, return original
        return text

    def shutdown(self):
        """Cleanup."""
        print(f"MinimalCleaningProcessor stats: processed={self.processed_count}, errors={self.error_count}")

    def force_flush(self, timeout_millis: int = 30000):
        """No-op for processor."""
        return True
