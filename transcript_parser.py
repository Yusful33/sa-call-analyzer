import re
from typing import List, Tuple, Optional
from models import TranscriptLine
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Initialize tracer for transcript parser
tracer = trace.get_tracer("transcript-parser")


class TranscriptParser:
    """
    Parses call transcripts in various formats.
    Handles both labeled and unlabeled transcripts.
    """

    @staticmethod
    def parse(transcript: str) -> Tuple[List[TranscriptLine], bool]:
        """
        Parse transcript and return list of transcript lines.

        Returns:
            Tuple of (parsed_lines, has_speaker_labels)
        """
        with tracer.start_as_current_span(
            "parse_transcript_lines",
            attributes={
                "transcript.input_length": len(transcript),
                "transcript.raw_line_count": len(transcript.strip().split('\n')),
                # OpenInference input
                "input.value": transcript,
                "input.mime_type": "text/plain",
                # OpenInference span kind - this is a chain/transformation
                "openinference.span.kind": "chain",
            }
        ) as span:
            try:
                lines = transcript.strip().split('\n')
                parsed_lines = []
                has_labels = False
                timestamp_matches = 0
                full_pattern_matches = 0
                plain_text_lines = 0

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Try to parse format: "0:16 | Speaker"
                    timestamp_speaker_pattern = r'^(\d+:\d+)\s*\|\s*(.+)$'
                    match = re.match(timestamp_speaker_pattern, line)

                    if match:
                        # This is a timestamp + speaker line
                        timestamp = match.group(1)
                        speaker = match.group(2).strip()
                        has_labels = True
                        timestamp_matches += 1
                        # Create entry with empty text (will be filled by next line)
                        parsed_lines.append(TranscriptLine(
                            timestamp=timestamp,
                            speaker=speaker,
                            text=""
                        ))
                        continue

                    # Try to parse format: "Timestamp | Speaker: text" (single line)
                    full_pattern = r'^(\d+:\d+)\s*\|\s*([^:]+):\s*(.+)$'
                    match = re.match(full_pattern, line)

                    if match:
                        timestamp = match.group(1)
                        speaker = match.group(2).strip()
                        text = match.group(3).strip()
                        has_labels = True
                        full_pattern_matches += 1
                        parsed_lines.append(TranscriptLine(
                            timestamp=timestamp,
                            speaker=speaker,
                            text=text
                        ))
                        continue

                    # Check if this looks like dialogue text (comes after a speaker line)
                    if parsed_lines and has_labels:
                        # Append this as text to the most recent speaker
                        if parsed_lines[-1].text:
                            parsed_lines[-1].text += " " + line
                        else:
                            parsed_lines[-1].text = line
                    else:
                        # No speaker labels detected, treat as plain text
                        plain_text_lines += 1
                        parsed_lines.append(TranscriptLine(
                            timestamp=None,
                            speaker=None,
                            text=line
                        ))

                span.set_attribute("transcript.parsed_line_count", len(parsed_lines))
                span.set_attribute("transcript.has_labels", has_labels)
                span.set_attribute("transcript.timestamp_matches", timestamp_matches)
                span.set_attribute("transcript.full_pattern_matches", full_pattern_matches)
                span.set_attribute("transcript.plain_text_lines", plain_text_lines)
                # OpenInference output - serialize parsed lines
                import json
                span.set_attribute("output.value", json.dumps({
                    "parsed_lines": [{"timestamp": line.timestamp, "speaker": line.speaker, "text": line.text} for line in parsed_lines],
                    "has_labels": has_labels
                }))
                span.set_attribute("output.mime_type", "application/json")
                span.set_status(Status(StatusCode.OK))

                return parsed_lines, has_labels
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    @staticmethod
    def format_for_analysis(parsed_lines: List[TranscriptLine]) -> str:
        """
        Format parsed lines back into a clean string for LLM analysis.
        """
        with tracer.start_as_current_span(
            "format_transcript_for_llm",
            attributes={
                "transcript.input_line_count": len(parsed_lines),
                # OpenInference input
                "input.value": str([{"timestamp": line.timestamp, "speaker": line.speaker, "text": line.text} for line in parsed_lines]),
                "input.mime_type": "application/json",
                # OpenInference span kind - this is a chain/transformation
                "openinference.span.kind": "chain",
            }
        ) as span:
            try:
                result = []
                lines_with_timestamp = 0
                lines_with_speaker = 0

                for line in parsed_lines:
                    parts = []
                    if line.timestamp:
                        parts.append(f"[{line.timestamp}]")
                        lines_with_timestamp += 1
                    if line.speaker:
                        parts.append(f"{line.speaker}:")
                        lines_with_speaker += 1
                    parts.append(line.text)
                    result.append(" ".join(parts))

                formatted = "\n".join(result)

                span.set_attribute("transcript.output_length", len(formatted))
                span.set_attribute("transcript.output_line_count", len(result))
                span.set_attribute("transcript.lines_with_timestamp", lines_with_timestamp)
                span.set_attribute("transcript.lines_with_speaker", lines_with_speaker)
                # OpenInference output
                span.set_attribute("output.value", formatted)
                span.set_attribute("output.mime_type", "text/plain")
                span.set_status(Status(StatusCode.OK))

                return formatted
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    @staticmethod
    def extract_speakers(parsed_lines: List[TranscriptLine]) -> List[str]:
        """
        Extract unique speakers from parsed transcript.
        """
        with tracer.start_as_current_span(
            "extract_speakers",
            attributes={
                "transcript.input_line_count": len(parsed_lines),
                # OpenInference span kind - this is a chain/transformation
                "openinference.span.kind": "chain",
            }
        ) as span:
            try:
                speakers = set()
                for line in parsed_lines:
                    if line.speaker:
                        speakers.add(line.speaker)

                speaker_list = sorted(list(speakers))

                span.set_attribute("transcript.speaker_count", len(speaker_list))
                if speaker_list:
                    span.set_attribute("transcript.speaker_names", ", ".join(speaker_list))
                span.set_status(Status(StatusCode.OK))

                return speaker_list
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
