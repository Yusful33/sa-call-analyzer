"""
Client for interacting with the Gong MCP HTTP server.

The MCP server now exposes HTTP endpoints that wrap the MCP JSON-RPC protocol internally.
"""
import json
import os
import base64
import requests
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class GongMCPClient:
    """Client for the Gong MCP HTTP server."""

    def __init__(self, base_url: str = None):
        """
        Initialize Gong MCP client.

        Args:
            base_url: Base URL of the Gong MCP HTTP server.
                     Defaults to GONG_MCP_URL env var, or "http://gong-mcp:8080" if not set.
        """
        self.base_url = base_url or os.getenv("GONG_MCP_URL", "http://gong-mcp:8080")
        self.tracer = trace.get_tracer("gong-mcp-client")

    def _make_request(self, endpoint: str, payload: Dict) -> Dict:
        """
        Make HTTP request to the Gong MCP server.

        Args:
            endpoint: API endpoint (e.g., "/transcript", "/calls")
            payload: Request payload

        Returns:
            Response dictionary

        Raises:
            RuntimeError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        with self.tracer.start_as_current_span(
            f"gong_mcp_{endpoint.strip('/')}",
            attributes={
                "http.url": url,
                "http.method": "POST",
                "input.value": json.dumps(payload),
                "input.mime_type": "application/json",
                "openinference.span.kind": "tool",
            }
        ) as span:
            try:
                response = requests.post(url, json=payload, timeout=60)
                
                span.set_attribute("http.status_code", response.status_code)
                
                if response.status_code != 200:
                    error_msg = f"Gong MCP error {response.status_code}: {response.text}"
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    raise RuntimeError(error_msg)
                
                result = response.json()
                span.set_attribute("output.value", json.dumps(result))
                span.set_attribute("output.mime_type", "application/json")
                span.set_status(Status(StatusCode.OK))
                
                return result
                
            except requests.RequestException as e:
                error_msg = f"HTTP request failed: {str(e)}"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                span.record_exception(e)
                raise RuntimeError(error_msg)

    @staticmethod
    def extract_call_id_from_url(gong_url: str) -> str:
        """
        Extract call ID from a Gong URL.

        Supports:
        - Regular: https://app.gong.io/call?id=7782342274025937895
        - Embedded: https://subdomain.app.gong.io/embedded-call?call-id=12345

        Args:
            gong_url: Gong call URL

        Returns:
            Call ID string

        Raises:
            ValueError: If URL format is invalid or call ID not found
        """
        try:
            parsed = urlparse(gong_url)
            params = parse_qs(parsed.query)

            # Check for regular format: ?id=...
            if "id" in params:
                return params["id"][0]

            # Check for embedded format: ?call-id=...
            if "call-id" in params:
                return params["call-id"][0]

            raise ValueError("No call ID found in URL")

        except Exception as e:
            raise ValueError(f"Invalid Gong URL format: {e}")

    def list_calls(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        """
        List Gong calls with optional date filtering.

        Args:
            from_date: ISO format start date (e.g., "2024-03-01T00:00:00Z")
            to_date: ISO format end date (e.g., "2024-03-31T23:59:59Z")

        Returns:
            List of call dictionaries

        Raises:
            RuntimeError: If request fails
        """
        payload = {}
        if from_date:
            payload["from_date"] = from_date
        if to_date:
            payload["to_date"] = to_date

        response = self._make_request("/calls", payload)
        return response.get("calls", [])

    def get_transcript(self, call_id: str) -> Dict:
        """
        Retrieve transcript for a specific call ID.

        Args:
            call_id: Gong call ID

        Returns:
            Transcript data dictionary

        Raises:
            RuntimeError: If request fails
        """
        return self._make_request("/transcript", {"call_id": call_id})

    def format_transcript_for_analysis(self, transcript_data: Dict, speaker_names: Dict = None) -> str:
        """
        Format Gong transcript data into readable format for analysis.

        Args:
            transcript_data: Raw transcript data from Gong MCP
            speaker_names: Optional mapping of speaker IDs to name info (from get_speaker_names)

        Returns:
            Formatted transcript string with timestamps and speakers
        """
        with self.tracer.start_as_current_span(
            "format_transcript",
            attributes={
                "transcript.raw_size": len(json.dumps(transcript_data)),
                "input.value": json.dumps(transcript_data),
                "input.mime_type": "application/json",
                "openinference.span.kind": "chain",
            }
        ) as span:
            try:
                if not transcript_data:
                    raise ValueError("Invalid transcript data: Response is empty")

                # Handle different response formats
                # Gong API format: {"callTranscripts": [{"transcript": [...]}]}
                # MCP format: {"transcripts": [{"sentences": [...], "speakerId": "..."}]}

                if "callTranscripts" in transcript_data:
                    # Direct Gong API format - needs flattening
                    call_transcripts = transcript_data["callTranscripts"]
                    if not call_transcripts:
                        raise ValueError("No transcripts found in callTranscripts")

                    transcript_turns = call_transcripts[0].get("transcript", [])

                    # Flatten speaker turns into individual sentences
                    transcript_sentences = []
                    for turn in transcript_turns:
                        speaker_id = turn.get("speakerId", "Unknown")
                        sentences = turn.get("sentences", [])
                        for sentence in sentences:
                            transcript_sentences.append({
                                "speakerId": speaker_id,
                                "start": sentence.get("start", 0),
                                "end": sentence.get("end", 0),
                                "text": sentence.get("text", "")
                            })
                elif "transcripts" in transcript_data:
                    # MCP server format - flatten transcripts into sentences
                    transcripts = transcript_data["transcripts"]
                    if not transcripts:
                        raise ValueError("No transcripts found in transcripts")
                    transcript_sentences = []
                    for t in transcripts:
                        for sentence in t.get("sentences", []):
                            transcript_sentences.append({
                                "speakerId": t.get("speakerId", "Unknown"),
                                "start": sentence.get("start", 0),
                                "text": sentence.get("text", "")
                            })
                else:
                    available_keys = list(transcript_data.keys()) if isinstance(transcript_data, dict) else "not a dict"
                    raise ValueError(
                        f"Invalid transcript data: Missing 'callTranscripts' or 'transcripts' field. "
                        f"Available fields: {available_keys}."
                    )

                transcript = transcript_sentences
                if not transcript:
                    raise ValueError("Transcript is empty")

                # Format into readable transcript
                lines = []
                current_speaker = None
                current_text = []
                last_timestamp = None

                for sentence in transcript:
                    speaker_id = sentence.get("speakerId", "Unknown")
                    text = sentence.get("text", "").strip()
                    start_time_ms = sentence.get("start", 0)

                    if not text:
                        continue

                    timestamp = self._format_timestamp(start_time_ms)

                    if speaker_id != current_speaker:
                        if current_speaker and current_text:
                            # Use actual name if available, otherwise fall back to Speaker ID
                            speaker_info = speaker_names.get(current_speaker, {}) if speaker_names else {}
                            speaker_label = speaker_info.get("name") or f"Speaker {current_speaker}"
                            lines.append(f"{last_timestamp} | {speaker_label}")
                            lines.append(" ".join(current_text))
                            lines.append("")

                        current_speaker = speaker_id
                        current_text = [text]
                        last_timestamp = timestamp
                    else:
                        current_text.append(text)

                # Add final speaker's text
                if current_speaker and current_text:
                    # Use actual name if available, otherwise fall back to Speaker ID
                    speaker_info = speaker_names.get(current_speaker, {}) if speaker_names else {}
                    speaker_label = speaker_info.get("name") or f"Speaker {current_speaker}"
                    lines.append(f"{last_timestamp} | {speaker_label}")
                    lines.append(" ".join(current_text))

                result = "\n".join(lines)

                if not result.strip():
                    raise ValueError("Transcript formatting resulted in empty content.")

                # Count unique speakers
                unique_speakers = set()
                for line in lines:
                    if " | Speaker " in line:
                        speaker = line.split(" | Speaker ")[1]
                        unique_speakers.add(speaker)

                span.set_attribute("transcript.formatted_length", len(result))
                span.set_attribute("transcript.sentence_count", len(transcript))
                span.set_attribute("transcript.speaker_count", len(unique_speakers))
                span.set_attribute("output.value", result)
                span.set_attribute("output.mime_type", "text/plain")
                span.set_status(Status(StatusCode.OK))

                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    @staticmethod
    def _format_timestamp(milliseconds: int) -> str:
        """Convert milliseconds to MM:SS format."""
        seconds = milliseconds // 1000
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}"

    def get_formatted_transcript_from_url(self, gong_url: str) -> str:
        """
        Fetch and format transcript from a Gong URL (convenience method).

        Args:
            gong_url: Gong call URL

        Returns:
            Formatted transcript string ready for analysis

        Raises:
            ValueError: If URL is invalid
            RuntimeError: If request fails
        """
        with self.tracer.start_as_current_span(
            "fetch_gong_transcript",
            attributes={
                "gong.url": gong_url,
                "input.value": gong_url,
                "input.mime_type": "text/plain",
                "openinference.span.kind": "agent"
            }
        ) as span:
            try:
                call_id = self.extract_call_id_from_url(gong_url)
                span.set_attribute("gong.call_id", call_id)

                transcript_data = self.get_transcript(call_id)
                formatted = self.format_transcript_for_analysis(transcript_data)

                span.set_attribute("transcript.length", len(formatted))
                span.set_attribute("output.value", formatted)
                span.set_attribute("output.mime_type", "text/plain")
                span.set_status(Status(StatusCode.OK))

                return formatted
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def get_speaker_names(self, call_id: str) -> Dict[str, Dict]:
        """
        Get speaker ID to name mapping for a call using Gong API.

        Note: This calls the Gong API directly for speaker info.

        Args:
            call_id: Gong call ID

        Returns:
            Dictionary mapping speaker IDs to participant info
        """
        with self.tracer.start_as_current_span(
            "get_speaker_names",
            attributes={
                "gong.call_id": call_id,
                "openinference.span.kind": "tool",
            }
        ) as span:
            try:
                # Get Gong credentials from environment
                access_key = os.getenv("GONG_ACCESS_KEY")
                secret_key = os.getenv("GONG_SECRET_KEY") or os.getenv("GONG_ACCESS_SECRET")

                if not access_key or not secret_key:
                    raise ValueError("GONG_ACCESS_KEY and GONG_SECRET_KEY must be set")

                # Create Basic Auth header
                credentials = f"{access_key}:{secret_key}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers = {
                    "Authorization": f"Basic {encoded}",
                    "Content-Type": "application/json"
                }

                # Call Gong API for extensive call details
                url = "https://api.gong.io/v2/calls/extensive"
                payload = {
                    "filter": {
                        "callIds": [call_id]
                    },
                    "contentSelector": {
                        "exposedFields": {
                            "parties": True
                        }
                    }
                }

                response = requests.post(url, headers=headers, json=payload, timeout=30)

                if response.status_code != 200:
                    error_msg = f"Gong API error {response.status_code}: {response.text}"
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    raise RuntimeError(error_msg)

                data = response.json()

                # Extract speaker mappings
                speaker_map = {}
                if "calls" in data and len(data["calls"]) > 0:
                    parties = data["calls"][0].get("parties", [])

                    for party in parties:
                        speaker_id = party.get("speakerId")
                        if speaker_id:
                            speaker_map[speaker_id] = {
                                "name": party.get("name", "Unknown"),
                                "email": party.get("emailAddress", ""),
                                "title": party.get("title", ""),
                                "affiliation": party.get("affiliation", "Unknown")
                            }

                span.set_attribute("speaker.count", len(speaker_map))
                span.set_status(Status(StatusCode.OK))

                return speaker_map

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def analyze_speaker_statistics(self, transcript_data: Dict) -> Dict[str, Dict]:
        """
        Analyze speaker statistics from transcript data.

        Args:
            transcript_data: Raw transcript data from Gong MCP

        Returns:
            Dictionary mapping speaker IDs to their statistics
        """
        with self.tracer.start_as_current_span(
            "analyze_speaker_statistics",
            attributes={"openinference.span.kind": "chain"}
        ) as span:
            try:
                speaker_stats = {}

                # Extract transcript data
                if "callTranscripts" in transcript_data:
                    transcript_turns = transcript_data["callTranscripts"][0].get("transcript", [])
                elif "transcripts" in transcript_data:
                    transcript_turns = []
                    for t in transcript_data["transcripts"]:
                        speaker_id = t.get("speakerId", "Unknown")
                        for sentence in t.get("sentences", []):
                            transcript_turns.append({
                                "speakerId": speaker_id,
                                "sentences": [sentence]
                            })
                else:
                    raise ValueError("Invalid transcript data format")

                # Analyze each turn
                for idx, turn in enumerate(transcript_turns):
                    speaker_id = turn.get("speakerId", "Unknown")
                    sentences = turn.get("sentences", [])

                    text = " ".join(s.get("text", "") for s in sentences)
                    word_count = len(text.split())
                    sentence_count = len(sentences)

                    if speaker_id not in speaker_stats:
                        speaker_stats[speaker_id] = {
                            "turn_count": 0,
                            "sentence_count": 0,
                            "word_count": 0,
                            "topics": set(),
                            "turns": []
                        }

                    speaker_stats[speaker_id]["turn_count"] += 1
                    speaker_stats[speaker_id]["sentence_count"] += sentence_count
                    speaker_stats[speaker_id]["word_count"] += word_count
                    speaker_stats[speaker_id]["turns"].append({
                        "index": idx,
                        "text": text,
                        "word_count": word_count
                    })

                # Calculate derived metrics
                for speaker_id, stats in speaker_stats.items():
                    stats["avg_words_per_turn"] = (
                        stats["word_count"] / stats["turn_count"]
                        if stats["turn_count"] > 0 else 0
                    )
                    score = (
                        (stats["word_count"] / 100) * 0.4 +
                        (stats["avg_words_per_turn"] / 10) * 0.3 +
                        (stats["turn_count"] / 10) * 0.3
                    )
                    stats["sa_likelihood_score"] = round(score, 2)

                span.set_attribute("speaker.count", len(speaker_stats))
                span.set_status(Status(StatusCode.OK))
                return speaker_stats

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def create_hybrid_sample_for_sa_identification(
        self,
        transcript_data: Dict,
        call_id: Optional[str] = None,
        max_speakers: int = 3
    ) -> str:
        """
        Create a hybrid sample optimized for SA identification.

        Args:
            transcript_data: Raw transcript data from Gong MCP
            call_id: Optional call ID to fetch speaker names
            max_speakers: Maximum number of speakers to include

        Returns:
            Formatted sample string optimized for SA identification
        """
        with self.tracer.start_as_current_span(
            "create_hybrid_sample",
            attributes={
                "openinference.span.kind": "chain",
                "max_speakers": max_speakers,
                "has_call_id": bool(call_id)
            }
        ) as span:
            try:
                speaker_stats = self.analyze_speaker_statistics(transcript_data)

                speaker_names = {}
                if call_id:
                    try:
                        speaker_names = self.get_speaker_names(call_id)
                    except Exception as e:
                        print(f"Could not fetch speaker names: {e}")

                sorted_speakers = sorted(
                    speaker_stats.items(),
                    key=lambda x: x[1]["sa_likelihood_score"],
                    reverse=True
                )

                top_speakers = sorted_speakers[:max_speakers]

                sample_parts = []
                sample_parts.append("SPEAKER ANALYSIS FOR SA IDENTIFICATION")
                sample_parts.append("=" * 80)
                sample_parts.append("")

                for speaker_id, stats in top_speakers:
                    turns = stats["turns"]
                    speaker_info = speaker_names.get(speaker_id, {})
                    speaker_name = speaker_info.get("name")
                    speaker_title = speaker_info.get("title")

                    if speaker_name:
                        header = f"--- {speaker_name}"
                        if speaker_title:
                            header += f" ({speaker_title})"
                        header += " ---"
                        sample_parts.append(header)
                        sample_parts.append(f"Speaker ID: {speaker_id}")
                    else:
                        sample_parts.append(f"--- Speaker {speaker_id} ---")

                    sample_parts.append(
                        f"Statistics: {stats['turn_count']} turns, "
                        f"{stats['word_count']} total words, "
                        f"{stats['avg_words_per_turn']:.1f} avg words/turn, "
                        f"SA likelihood score: {stats['sa_likelihood_score']}"
                    )
                    sample_parts.append("")

                    # Hybrid sampling
                    sample_turns = []
                    if turns:
                        sample_turns.append(("first", turns[0]))

                    sorted_by_content = sorted(turns, key=lambda t: t["word_count"], reverse=True)
                    for turn in sorted_by_content[:3]:
                        sample_turns.append(("content", turn))

                    if len(turns) > 1:
                        sample_turns.append(("last", turns[-1]))

                    seen_indices = set()
                    unique_sample = []
                    for label, turn in sample_turns:
                        if turn["index"] not in seen_indices:
                            seen_indices.add(turn["index"])
                            unique_sample.append((label, turn))

                    unique_sample.sort(key=lambda x: x[1]["index"])

                    sample_parts.append(f"Sample ({len(unique_sample)} turns):")
                    for label, turn in unique_sample:
                        text = turn["text"]
                        if len(text) > 300:
                            text = text[:300] + "..."

                        marker = ""
                        if label == "first":
                            marker = " [FIRST]"
                        elif label == "last":
                            marker = " [LAST]"
                        elif label == "content":
                            marker = f" [TOP CONTENT - {turn['word_count']} words]"

                        sample_parts.append(f"  Turn {turn['index']}{marker}: {text}")

                    sample_parts.append("")
                    sample_parts.append("-" * 80)
                    sample_parts.append("")

                result = "\n".join(sample_parts)

                span.set_attribute("sample.length", len(result))
                span.set_attribute("sample.speaker_count", len(top_speakers))
                span.set_attribute("output.value", result)
                span.set_attribute("output.mime_type", "text/plain")
                span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
