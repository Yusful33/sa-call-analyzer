"""
Client for interacting with the Gong MCP HTTP server.

The MCP server now exposes HTTP endpoints that wrap the MCP JSON-RPC protocol internally.
"""
import json
import os
import base64
import time
import requests
from typing import List, Dict, Optional, Generator
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# #region agent log
LOG_PATH = "/Users/yusufcattaneo/Projects/.cursor/debug.log"
def _debug_log(location, message, data, hypothesis_id=None):
    try:
        import time
        log_entry = {
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "hypothesisId": hypothesis_id or "unknown"
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except: pass
# #endregion


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
        # TTL cache for transcripts and call info (5 minutes default)
        self._transcript_cache: Dict[str, Dict] = {}
        self._transcript_cache_times: Dict[str, float] = {}
        self._call_info_cache: Dict[str, Dict] = {}
        self._call_info_cache_times: Dict[str, float] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

    def _is_cache_valid(self, cache_times: Dict[str, float], key: str) -> bool:
        """Check if a cached entry is still valid."""
        if key not in cache_times:
            return False
        return (time.time() - cache_times[key]) < self._cache_ttl_seconds

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
                response = requests.post(url, json=payload, timeout=300)
                
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

        print(f"📞 Calling list_calls with payload: {payload}")
        response = self._make_request("/calls", payload)
        print(f"📞 API response keys: {list(response.keys())}")
        print(f"📞 Response type: {type(response)}")
        print(f"📞 Full response (first 1000 chars): {str(response)[:1000]}")
        
        # Try multiple possible response structures
        calls = []
        if isinstance(response, dict):
            # Try different possible keys
            if "calls" in response:
                calls = response.get("calls", [])
            elif "callIds" in response:
                # If we only get IDs, we'd need to fetch details separately
                call_ids = response.get("callIds", [])
                print(f"📞 Got callIds list with {len(call_ids)} IDs")
                calls = [{"id": cid} for cid in call_ids]
            elif isinstance(response, list):
                calls = response
        elif isinstance(response, list):
            calls = response
            
        print(f"📞 Found {len(calls)} calls after parsing")
        if len(calls) > 0:
            print(f"📞 First call sample: {calls[0]}")
        return calls

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
        # Check cache first
        if call_id in self._transcript_cache and self._is_cache_valid(self._transcript_cache_times, call_id):
            return self._transcript_cache[call_id]

        result = self._make_request("/transcript", {"call_id": call_id})

        # Cache the result
        self._transcript_cache[call_id] = result
        self._transcript_cache_times[call_id] = time.time()

        return result

    def get_transcripts_parallel(
        self,
        call_ids: List[str],
        max_workers: int = 5
    ) -> Dict[str, Dict]:
        """
        Fetch transcripts for multiple calls in parallel.

        Args:
            call_ids: List of Gong call IDs
            max_workers: Maximum number of parallel workers (default 5)

        Returns:
            Dictionary mapping call_id -> transcript_data
        """
        with self.tracer.start_as_current_span(
            "get_transcripts_parallel",
            attributes={
                "call_ids.count": len(call_ids),
                "max_workers": max_workers,
                "openinference.span.kind": "tool",
            }
        ) as span:
            results = {}
            calls_to_fetch = []

            # Check cache first
            for call_id in call_ids:
                if call_id in self._transcript_cache and self._is_cache_valid(self._transcript_cache_times, call_id):
                    results[call_id] = self._transcript_cache[call_id]
                else:
                    calls_to_fetch.append(call_id)

            span.set_attribute("calls.cached", len(call_ids) - len(calls_to_fetch))
            span.set_attribute("calls.to_fetch", len(calls_to_fetch))

            if not calls_to_fetch:
                span.set_status(Status(StatusCode.OK))
                return results

            # Fetch remaining transcripts in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_call_id = {
                    executor.submit(self._make_request, "/transcript", {"call_id": cid}): cid
                    for cid in calls_to_fetch
                }

                for future in as_completed(future_to_call_id):
                    call_id = future_to_call_id[future]
                    try:
                        transcript_data = future.result()
                        results[call_id] = transcript_data
                        # Cache the result
                        self._transcript_cache[call_id] = transcript_data
                        self._transcript_cache_times[call_id] = time.time()
                    except Exception as e:
                        span.add_event("transcript_fetch_failed", {
                            "call_id": call_id,
                            "error": str(e)
                        })
                        # Store None for failed fetches
                        results[call_id] = None

            span.set_attribute("calls.fetched", len([r for r in results.values() if r is not None]))
            span.set_status(Status(StatusCode.OK))
            return results

    def clear_cache(self):
        """Clear all cached data."""
        self._transcript_cache.clear()
        self._transcript_cache_times.clear()
        self._call_info_cache.clear()
        self._call_info_cache_times.clear()

    def get_call_info(self, call_id: str) -> Dict:
        """
        Get call metadata including the call date.

        Args:
            call_id: Gong call ID

        Returns:
            Call info dictionary with 'scheduled', 'title', 'duration', etc.

        Raises:
            RuntimeError: If request fails
        """
        # Check cache first
        if call_id in self._call_info_cache and self._is_cache_valid(self._call_info_cache_times, call_id):
            return self._call_info_cache[call_id]

        with self.tracer.start_as_current_span(
            "get_call_info",
            attributes={
                "gong.call_id": call_id,
                "openinference.span.kind": "tool",
            }
        ) as span:
            try:
                response = self._make_request("/call-info", {"call_id": call_id})
                span.set_attribute("call.has_scheduled", "scheduled" in response)
                # #region agent log
                _debug_log("gong_mcp_client.py:173", "get_call_info response", {
                    "call_id": call_id,
                    "response_keys": list(response.keys())[:15],
                    "has_parties": "parties" in response,
                    "response_sample": str(response)[:500]
                }, "H2")
                # #endregion

                # Cache the result
                self._call_info_cache[call_id] = response
                self._call_info_cache_times[call_id] = time.time()

                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                # If call-info endpoint doesn't exist, return empty dict
                span.set_status(Status(StatusCode.ERROR, str(e)))
                print(f"⚠️ Could not get call info: {e}")
                return {}

    def get_call_date(self, call_id: str) -> str:
        """
        Get the formatted call date for a call.

        Args:
            call_id: Gong call ID

        Returns:
            Formatted date string (e.g., "December 10, 2025") or empty string if not available
        """
        try:
            call_info = self.get_call_info(call_id)
            
            # Try to get 'scheduled' field (ISO format timestamp)
            scheduled = call_info.get("scheduled") or call_info.get("started") or call_info.get("date")
            
            if scheduled:
                from datetime import datetime
                # Parse ISO format date
                if isinstance(scheduled, str):
                    # Try parsing ISO format
                    try:
                        dt = datetime.fromisoformat(scheduled.replace('Z', '+00:00'))
                        return dt.strftime("%B %d, %Y")
                    except ValueError:
                        # Try other formats
                        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                            try:
                                dt = datetime.strptime(scheduled[:len(fmt.replace('%', ''))+4], fmt)
                                return dt.strftime("%B %d, %Y")
                            except ValueError:
                                continue
                elif isinstance(scheduled, (int, float)):
                    # Unix timestamp (milliseconds)
                    dt = datetime.fromtimestamp(scheduled / 1000 if scheduled > 1e12 else scheduled)
                    return dt.strftime("%B %d, %Y")
            
            return ""
        except Exception as e:
            print(f"⚠️ Could not extract call date: {e}")
            return ""

    def format_transcript_for_analysis(self, transcript_data: Dict) -> str:
        """
        Format Gong transcript data into readable format for analysis.

        Args:
            transcript_data: Raw transcript data from Gong MCP

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
                            lines.append(f"{last_timestamp} | Speaker {current_speaker}")
                            lines.append(" ".join(current_text))
                            lines.append("")

                        current_speaker = speaker_id
                        current_text = [text]
                        last_timestamp = timestamp
                    else:
                        current_text.append(text)

                # Add final speaker's text
                if current_speaker and current_text:
                    lines.append(f"{last_timestamp} | Speaker {current_speaker}")
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

    @staticmethod
    def normalize_prospect_name(name: str) -> str:
        """
        Normalize a prospect name for matching.
        
        Removes common prefixes/suffixes, lowercases, and trims whitespace.
        Handles variations like "John A. Smith" vs "John Smith" or "Gong LLC" vs "Gong".
        
        Args:
            name: Prospect name to normalize
            
        Returns:
            Normalized name string
        """
        if not name:
            return ""
        
        # Lowercase and trim
        normalized = name.lower().strip()
        
        # Remove common prefixes
        prefixes = ["mr.", "mrs.", "ms.", "dr.", "prof."]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Remove common suffixes (company types)
        suffixes = ["llc", "inc.", "inc", "corp.", "corp", "ltd.", "ltd", "limited", 
                   "corporation", "company", "co.", "co"]
        # Remove suffixes with optional comma/period before them
        for suffix in suffixes:
            # Handle "Gong LLC", "Gong, Inc.", "Gong Inc" etc.
            normalized = normalized.replace(f", {suffix}", "")
            normalized = normalized.replace(f" {suffix}", "")
            normalized = normalized.replace(f".{suffix}", "")
            normalized = normalized.replace(f". {suffix}", "")
        
        # Remove middle initials/names (e.g., "John A. Smith" -> "john smith")
        # This is a simple approach - remove single letters/initials
        import re
        # Remove single letter words followed by period (middle initials)
        normalized = re.sub(r'\b[a-z]\.\s*', '', normalized)
        # Remove standalone single letters (middle initials without period)
        normalized = re.sub(r'\b[a-z]\b', '', normalized)
        # Clean up extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized.strip()

    def extract_account_names(self, call_info: Dict) -> List[str]:
        """
        Extract ONLY account/company names from call metadata (not participant names).
        This matches Gong's "Account name" filter behavior.
        
        Args:
            call_info: Call info dictionary from get_call_info()
            
        Returns:
            List of account/company names only
        """
        account_names = []
        
        # #region agent log
        _debug_log("gong_mcp_client.py:769", "Extracting account names only", {
            "call_info_keys": list(call_info.keys())[:15],
            "has_accountName": "accountName" in call_info,
            "has_account": "account" in call_info,
            "has_parties": "parties" in call_info,
            "full_call_info_sample": str(call_info)[:800]  # Larger sample to see account fields
        }, "H2,H3")
        # #endregion
        
        # PRIORITY 1: Extract account name at call level (this is what Gong UI filters by)
        account_name = call_info.get("accountName")
        if account_name and account_name.strip():
            account_names.append(account_name.strip())
            # #region agent log
            _debug_log("gong_mcp_client.py:806", "Found account name at call level", {
                "account_name": account_name
            }, "H2")
            # #endregion
        
        # PRIORITY 2: Extract account name from nested account object
        account_obj = call_info.get("account")
        if isinstance(account_obj, dict):
            account_name_from_obj = account_obj.get("name")
            if account_name_from_obj and account_name_from_obj.strip():
                if account_name_from_obj.strip() not in account_names:
                    account_names.append(account_name_from_obj.strip())
                    # #region agent log
                    _debug_log("gong_mcp_client.py:815", "Found account name from account object", {
                        "account_name": account_name_from_obj
                    }, "H2")
                    # #endregion
        
        # PRIORITY 3: Extract company names from parties (fallback - but NOT participant names)
        parties = call_info.get("parties", [])
        if isinstance(parties, list):
            for party in parties:
                if isinstance(party, dict):
                    # Extract ONLY company/account name from party (NOT participant name)
                    company_name = party.get("companyName") or party.get("company") or party.get("affiliation")
                    if company_name and company_name.strip():
                        if company_name.strip() not in account_names:
                            account_names.append(company_name.strip())
                            # #region agent log
                            _debug_log("gong_mcp_client.py:825", "Found company name from party", {
                                "company_name": company_name
                            }, "H2")
                            # #endregion
                    # NOTE: We intentionally skip party.get("name") - that's a participant name, not account name
        
        # #region agent log
        _debug_log("gong_mcp_client.py:810", "Account names extracted", {
            "count": len(account_names),
            "account_names": account_names
        }, "H3")
        # #endregion
        return account_names

    def get_calls_by_prospect_name(
        self,
        prospect_name: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        fuzzy_threshold: float = 0.85
    ) -> List[Dict]:
        """
        Get all calls where the prospect name matches any participant.
        
        Uses fuzzy matching to handle name variations like "John Smith" vs "John A. Smith"
        or "Gong" vs "Gong LLC".
        
        Args:
            prospect_name: Name to search for (e.g., "John Smith" or "Gong")
            from_date: ISO format start date (e.g., "2024-03-01T00:00:00Z")
            to_date: ISO format end date (e.g., "2024-03-31T23:59:59Z")
            fuzzy_threshold: Similarity threshold for matching (0-1, default 0.85)
            
        Returns:
            List of call dictionaries where prospect name matches any participant
            
        Raises:
            RuntimeError: If request fails
            ImportError: If rapidfuzz is not installed
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            raise ImportError(
                "rapidfuzz library is required for prospect name matching. "
                "Install it with: pip install rapidfuzz"
            )
        
        with self.tracer.start_as_current_span(
            "get_calls_by_prospect_name",
            attributes={
                "prospect.name": prospect_name,
                "fuzzy.threshold": fuzzy_threshold,
                "openinference.span.kind": "tool",
            }
        ) as span:
            try:
                # Normalize the search name
                normalized_search = self.normalize_prospect_name(prospect_name)
                span.set_attribute("prospect.name.normalized", normalized_search)
                # #region agent log
                _debug_log("gong_mcp_client.py:812", "Normalized prospect name", {
                    "original": prospect_name,
                    "normalized": normalized_search
                }, "H4")
                # #endregion
                
                # Fetch all calls (or calls in date range)
                all_calls = self.list_calls(from_date=from_date, to_date=to_date)
                span.set_attribute("calls.total", len(all_calls))
                # #region agent log
                _debug_log("gong_mcp_client.py:817", "Fetched calls from Gong", {
                    "total_calls": len(all_calls),
                    "from_date": from_date,
                    "to_date": to_date,
                    "first_call_id": all_calls[0].get("id") if all_calls else None
                }, "H1")
                # #endregion
                
                matching_calls = []
                matched_names = set()
                
                # Debug: Print summary of what we're searching
                print(f"🔍 Searching for prospect: '{prospect_name}' (normalized: '{normalized_search}')")
                print(f"📞 Total calls to check: {len(all_calls)}")
                
                # For each call, check if prospect name matches any account/company name
                # NOTE: list_calls already returns extensive call data including parties,
                # so we use that directly instead of making separate get_call_info requests
                for idx, call in enumerate(all_calls):
                    call_id = call.get("id")
                    if not call_id:
                        continue

                    # Use call data directly - list_calls already fetches extensive details
                    try:
                        call_info = call  # Data already has parties, accountName, title, etc.

                        # Debug: Print call info structure for first few calls
                        if idx < 3:
                            print(f"\n📋 Call {idx+1} (ID: {call_id[:20]}...):")
                            print(f"   Keys in call: {list(call.keys())[:10]}")
                            print(f"   accountName: {call.get('accountName')}")
                            print(f"   title: {call.get('title')}")
                            if call.get('parties'):
                                print(f"   parties count: {len(call.get('parties', []))}")
                                if len(call.get('parties', [])) > 0:
                                    first_party = call.get('parties', [])[0]
                                    if isinstance(first_party, dict):
                                        print(f"   first party keys: {list(first_party.keys())[:10]}")

                        names_to_match = self.extract_account_names(call_info)

                        # Also check call title for potential matches (fallback when account names not available)
                        call_title = call_info.get("title") or call.get("title") or ""
                        if call_title and call_title not in names_to_match:
                            names_to_match.append(call_title)

                        # Extract email domains from participants (key for matching accounts like "Extend" -> @extend.com)
                        parties = call_info.get("parties", [])
                        if isinstance(parties, list):
                            for party in parties:
                                if isinstance(party, dict):
                                    email = party.get("emailAddress") or party.get("email") or ""
                                    if email and "@" in email:
                                        # Extract domain without TLD (e.g., "extend" from "user@extend.com")
                                        domain_part = email.split("@")[1].split(".")[0]
                                        if domain_part and domain_part not in names_to_match:
                                            names_to_match.append(domain_part)
                                        # Also add full domain (e.g., "extend.com")
                                        full_domain = email.split("@")[1]
                                        if full_domain and full_domain not in names_to_match:
                                            names_to_match.append(full_domain)

                        # #region agent log
                        _debug_log("gong_mcp_client.py:832", "Extracted names", {
                            "call_id": call_id,
                            "name_count": len(names_to_match),
                            "names": names_to_match[:5]  # First 5 only
                        }, "H3")
                        # #endregion

                        # Debug: Print extracted names
                        if names_to_match:
                            print(f"   ✅ Extracted account names (+ title): {names_to_match[:3]}")
                        else:
                            print(f"   ⚠️  No account names extracted from this call")
                        
                        # Check if any account/company name matches
                        for name_to_match in names_to_match:
                            normalized_name = self.normalize_prospect_name(name_to_match)
                            # #region agent log
                            _debug_log("gong_mcp_client.py:836", "Normalized name", {
                                "original": name_to_match,
                                "normalized": normalized_name,
                                "search_normalized": normalized_search
                            }, "H4")
                            # #endregion

                            # Check for substring match first (e.g., "arize" in "gong <> arize")
                            # Only count as substring match if:
                            # 1. The search term appears in the name (e.g., "extend" in "Arize + Extend")
                            # 2. The name appears in search AND name is at least 4 chars (avoid "x" matching "extend")
                            is_search_in_name = normalized_search in normalized_name.lower()
                            is_name_in_search = len(normalized_name) >= 4 and normalized_name.lower() in normalized_search
                            is_substring_match = is_search_in_name or is_name_in_search

                            # Calculate fuzzy similarity
                            similarity = fuzz.ratio(normalized_search, normalized_name) / 100.0

                            # Only use partial ratio if both strings are reasonably long
                            # This avoids "yext" matching "extend" due to shared "ext"
                            if len(normalized_name) >= 4 and len(normalized_search) >= 4:
                                partial_similarity = fuzz.partial_ratio(normalized_search, normalized_name) / 100.0
                                # Require higher partial match threshold to reduce false positives
                                partial_similarity = partial_similarity if partial_similarity >= 0.9 else 0.0
                            else:
                                partial_similarity = 0.0

                            # Use the best match: direct substring, fuzzy ratio, or high partial ratio
                            effective_similarity = max(similarity, partial_similarity, 1.0 if is_substring_match else 0.0)

                            print(f"      Comparing '{name_to_match}' (normalized: '{normalized_name}') -> similarity: {similarity:.2f}, partial: {partial_similarity:.2f}, substring: {is_substring_match} (threshold: {fuzzy_threshold})")
                            # #region agent log
                            _debug_log("gong_mcp_client.py:838", "Fuzzy match result", {
                                "name": name_to_match,
                                "normalized_name": normalized_name,
                                "normalized_search": normalized_search,
                                "similarity": similarity,
                                "partial_similarity": partial_similarity,
                                "is_substring": is_substring_match,
                                "effective_similarity": effective_similarity,
                                "threshold": fuzzy_threshold,
                                "match": effective_similarity >= fuzzy_threshold
                            }, "H5")
                            # #endregion

                            if effective_similarity >= fuzzy_threshold:
                                matching_calls.append(call)
                                matched_names.add(name_to_match)
                                span.add_event("match_found", {
                                    "call_id": call_id,
                                    "matched_name": name_to_match,
                                    "similarity": similarity
                                })
                                break  # Found a match, no need to check other names
                    except Exception as e:
                        # Skip calls where we can't get info
                        span.add_event("call_skipped", {
                            "call_id": call_id,
                            "reason": str(e)
                        })
                        continue
                
                # Debug: Print final summary
                print(f"\n📊 Matching Summary:")
                print(f"   Total calls checked: {len(all_calls)}")
                print(f"   Matching calls found: {len(matching_calls)}")
                if matched_names:
                    print(f"   Matched account names: {sorted(matched_names)}")
                else:
                    print(f"   ⚠️  No account names matched!")
                
                span.set_attribute("calls.matched", len(matching_calls))
                span.set_attribute("accounts.matched", ", ".join(sorted(matched_names)))
                span.set_status(Status(StatusCode.OK))
                # #region agent log
                _debug_log("gong_mcp_client.py:860", "Final matching results", {
                    "total_calls_checked": len(all_calls),
                    "matching_calls": len(matching_calls),
                    "matched_names": list(matched_names),
                    "normalized_search": normalized_search
                }, "H1,H2,H3,H5")
                # #endregion
                
                return matching_calls

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def get_calls_by_prospect_name_with_progress(
        self,
        prospect_name: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        fuzzy_threshold: float = 0.85
    ) -> Generator[Dict, None, None]:
        """
        Get all calls where the prospect name matches any participant, with progress events.

        This is a generator version of get_calls_by_prospect_name that yields progress events
        during the search process, enabling real-time progress updates via SSE.

        Yields:
            Progress event dicts with stage, message, and optional metadata
            Final event has stage="complete" with "result" containing the matching calls
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            yield {
                "stage": "error",
                "message": "rapidfuzz library is required for prospect name matching."
            }
            return

        with self.tracer.start_as_current_span(
            "get_calls_by_prospect_name_with_progress",
            attributes={
                "prospect.name": prospect_name,
                "fuzzy.threshold": fuzzy_threshold,
                "openinference.span.kind": "tool",
            }
        ) as span:
            try:
                # Normalize the search name
                normalized_search = self.normalize_prospect_name(prospect_name)
                span.set_attribute("prospect.name.normalized", normalized_search)

                # Stage 1: Querying Gong API
                yield {
                    "stage": "querying_api",
                    "message": "Querying Gong API for calls..."
                }

                # Fetch all calls (or calls in date range)
                all_calls = self.list_calls(from_date=from_date, to_date=to_date)
                span.set_attribute("calls.total", len(all_calls))

                # Stage 2: Retrieved calls, now filtering
                yield {
                    "stage": "filtering",
                    "message": f"Retrieved {len(all_calls)} calls, filtering by account name...",
                    "total_calls_to_check": len(all_calls)
                }

                matching_calls = []
                matched_names = set()

                # For each call, check if prospect name matches any account/company name
                for idx, call in enumerate(all_calls):
                    # Yield progress every 10 calls
                    if idx % 10 == 0 and len(all_calls) > 10:
                        yield {
                            "stage": "matching",
                            "message": f"Checking call {idx+1}/{len(all_calls)}...",
                            "checked": idx,
                            "total": len(all_calls)
                        }

                    call_id = call.get("id")
                    if not call_id:
                        continue

                    # Use call data directly - list_calls already fetches extensive details
                    try:
                        call_info = call

                        names_to_match = self.extract_account_names(call_info)

                        # Also check call title for potential matches
                        call_title = call_info.get("title") or call.get("title") or ""
                        if call_title and call_title not in names_to_match:
                            names_to_match.append(call_title)

                        # Extract email domains from participants
                        parties = call_info.get("parties", [])
                        if isinstance(parties, list):
                            for party in parties:
                                if isinstance(party, dict):
                                    email = party.get("emailAddress") or party.get("email") or ""
                                    if email and "@" in email:
                                        domain_part = email.split("@")[1].split(".")[0]
                                        if domain_part and domain_part not in names_to_match:
                                            names_to_match.append(domain_part)
                                        full_domain = email.split("@")[1]
                                        if full_domain and full_domain not in names_to_match:
                                            names_to_match.append(full_domain)

                        # Check if any account/company name matches
                        for name_to_match in names_to_match:
                            normalized_name = self.normalize_prospect_name(name_to_match)

                            # Check for substring match
                            is_search_in_name = normalized_search in normalized_name.lower()
                            is_name_in_search = len(normalized_name) >= 4 and normalized_name.lower() in normalized_search
                            is_substring_match = is_search_in_name or is_name_in_search

                            # Calculate fuzzy similarity
                            similarity = fuzz.ratio(normalized_search, normalized_name) / 100.0

                            # Only use partial ratio if both strings are reasonably long
                            if len(normalized_name) >= 4 and len(normalized_search) >= 4:
                                partial_similarity = fuzz.partial_ratio(normalized_search, normalized_name) / 100.0
                                partial_similarity = partial_similarity if partial_similarity >= 0.9 else 0.0
                            else:
                                partial_similarity = 0.0

                            effective_similarity = max(similarity, partial_similarity, 1.0 if is_substring_match else 0.0)

                            if effective_similarity >= fuzzy_threshold:
                                matching_calls.append(call)
                                matched_names.add(name_to_match)
                                span.add_event("match_found", {
                                    "call_id": call_id,
                                    "matched_name": name_to_match,
                                    "similarity": similarity
                                })
                                break
                    except Exception as e:
                        span.add_event("call_skipped", {
                            "call_id": call_id,
                            "reason": str(e)
                        })
                        continue

                span.set_attribute("calls.matched", len(matching_calls))
                span.set_attribute("accounts.matched", ", ".join(sorted(matched_names)))
                span.set_status(Status(StatusCode.OK))

                # Final stage: Complete with results
                yield {
                    "stage": "complete",
                    "message": f"Found {len(matching_calls)} matching calls",
                    "result": matching_calls,
                    "matched_names": list(matched_names)
                }

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                yield {
                    "stage": "error",
                    "message": f"Search failed: {str(e)}"
                }
