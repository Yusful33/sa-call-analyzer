import re
from typing import List, Tuple, Optional
from models import TranscriptLine


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
        lines = transcript.strip().split('\n')
        parsed_lines = []
        has_labels = False

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
                # Next line should be the actual text
                continue

            # Try to parse format: "Timestamp | Speaker: text" (single line)
            full_pattern = r'^(\d+:\d+)\s*\|\s*([^:]+):\s*(.+)$'
            match = re.match(full_pattern, line)

            if match:
                timestamp = match.group(1)
                speaker = match.group(2).strip()
                text = match.group(3).strip()
                has_labels = True
                parsed_lines.append(TranscriptLine(
                    timestamp=timestamp,
                    speaker=speaker,
                    text=text
                ))
                continue

            # Check if this looks like dialogue text (comes after a speaker line)
            if parsed_lines and has_labels:
                # Add this as text to the most recent speaker
                parsed_lines[-1].text = line
            else:
                # No speaker labels detected, treat as plain text
                parsed_lines.append(TranscriptLine(
                    timestamp=None,
                    speaker=None,
                    text=line
                ))

        return parsed_lines, has_labels

    @staticmethod
    def format_for_analysis(parsed_lines: List[TranscriptLine]) -> str:
        """
        Format parsed lines back into a clean string for LLM analysis.
        """
        result = []
        for line in parsed_lines:
            parts = []
            if line.timestamp:
                parts.append(f"[{line.timestamp}]")
            if line.speaker:
                parts.append(f"{line.speaker}:")
            parts.append(line.text)
            result.append(" ".join(parts))
        return "\n".join(result)

    @staticmethod
    def extract_speakers(parsed_lines: List[TranscriptLine]) -> List[str]:
        """
        Extract unique speakers from parsed transcript.
        """
        speakers = set()
        for line in parsed_lines:
            if line.speaker:
                speakers.add(line.speaker)
        return sorted(list(speakers))
