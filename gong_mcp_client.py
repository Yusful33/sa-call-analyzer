"""
Client for interacting with the Gong MCP server running in Docker.

The MCP server communicates via JSON-RPC over stdio.
"""
import json
import subprocess
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class GongMCPClient:
    """Client for the Gong MCP server running in Docker."""

    def __init__(self, container_name: str = "bold_mestorf"):
        """
        Initialize Gong MCP client.

        Args:
            container_name: Name of the Docker container running the Gong MCP server
        """
        self.container_name = container_name
        self.request_id = 0
        self.tracer = trace.get_tracer("gong-mcp-client")

    def _send_mcp_request(self, method: str, params: Dict) -> Dict:
        """
        Send a JSON-RPC request to the MCP server via docker exec.

        Args:
            method: MCP tool name
            params: Tool parameters

        Returns:
            Response dictionary

        Raises:
            RuntimeError: If MCP request fails
        """
        with self.tracer.start_as_current_span(
            f"mcp_tool_call_{method}",
            attributes={
                "mcp.tool": method,
                "mcp.container": self.container_name,
                "mcp.params": json.dumps(params),
                # OpenInference input
                "input.value": json.dumps({"method": method, "params": params}),
                "input.mime_type": "application/json",
                # OpenInference span kind - this is a tool call
                "openinference.span.kind": "tool",
            }
        ) as span:
            try:
                self.request_id += 1

                # MCP protocol requires initialization first
                init_request = {
                    "jsonrpc": "2.0",
                    "id": self.request_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "sa-call-analyzer",
                            "version": "1.0.0"
                        }
                    }
                }

                # Then the actual tool call
                self.request_id += 1
                tool_request = {
                    "jsonrpc": "2.0",
                    "id": self.request_id,
                    "method": "tools/call",
                    "params": {
                        "name": method,
                        "arguments": params
                    }
                }

                # Combine requests (newline-delimited JSON)
                requests_json = json.dumps(init_request) + "\n" + json.dumps(tool_request) + "\n"

                # Send via docker exec to the MCP server
                result = subprocess.run(
                    [
                        "docker", "exec", "-i", self.container_name,
                        "node", "/app/dist/index.js"
                    ],
                    input=requests_json,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                span.set_attribute("mcp.returncode", result.returncode)

                if result.returncode != 0:
                    span.set_status(Status(StatusCode.ERROR, f"MCP server error: {result.stderr}"))
                    raise RuntimeError(f"MCP server error: {result.stderr}")

                # Parse response (MCP server returns newline-delimited JSON)
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        response = json.loads(line)
                        # Look for the tool call response (not the initialize response)
                        if response.get("id") == self.request_id:
                            if "result" in response:
                                # OpenInference output
                                span.set_attribute("output.value", json.dumps(response["result"]))
                                span.set_attribute("output.mime_type", "application/json")
                                span.set_status(Status(StatusCode.OK))
                                return response["result"]
                            elif "error" in response:
                                error_msg = f"MCP error: {response['error']}"
                                span.set_status(Status(StatusCode.ERROR, error_msg))
                                raise RuntimeError(error_msg)
                    except json.JSONDecodeError:
                        continue

                error_msg = f"No valid tool response found in output: {result.stdout}"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                raise RuntimeError(error_msg)

            except subprocess.TimeoutExpired:
                error_msg = "MCP server request timed out"
                span.set_status(Status(StatusCode.ERROR, error_msg))
                raise RuntimeError(error_msg)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

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
            RuntimeError: If MCP request fails
        """
        params = {}
        if from_date:
            params["fromDateTime"] = from_date
        if to_date:
            params["toDateTime"] = to_date

        response = self._send_mcp_request("list_calls", params)

        # MCP server wraps response in content array with type "text"
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "{}")
                import json
                response = json.loads(text_content)

        return response.get("calls", [])

    def get_transcript(self, call_id: str) -> Dict:
        """
        Retrieve transcript for a specific call ID.

        Args:
            call_id: Gong call ID

        Returns:
            Transcript data dictionary

        Raises:
            RuntimeError: If MCP request fails
        """
        response = self._send_mcp_request("retrieve_transcripts", {"callIds": [call_id]})

        # MCP server wraps response in content array with type "text"
        # Extract the actual JSON from the text field
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "{}")
                import json
                response = json.loads(text_content)

        return response

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
                # OpenInference input
                "input.value": json.dumps(transcript_data),
                "input.mime_type": "application/json",
                # OpenInference span kind - this is a chain/transformation
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
                    # Structure: callTranscripts[0].transcript[i].sentences[j]
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
                    # Convert MCP format to Gong API format
                    transcript_sentences = []
                    for t in transcripts:
                        for sentence in t.get("sentences", []):
                            transcript_sentences.append({
                                "speakerId": t.get("speakerId", "Unknown"),
                                "start": sentence.get("start", 0),
                                "text": sentence.get("text", "")
                            })
                else:
                    # Log what we actually received for debugging
                    available_keys = list(transcript_data.keys()) if isinstance(transcript_data, dict) else "not a dict"
                    raise ValueError(
                        f"Invalid transcript data: Missing 'callTranscripts' or 'transcripts' field. "
                        f"Available fields: {available_keys}. "
                        f"Response type: {type(transcript_data).__name__}. "
                        f"Raw response: {json.dumps(transcript_data)[:500]}..."
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

                    # Skip sentences with no text
                    if not text:
                        continue

                    # Convert milliseconds to MM:SS format
                    timestamp = self._format_timestamp(start_time_ms)

                    if speaker_id != current_speaker:
                        # New speaker, save previous speaker's text
                        if current_speaker and current_text:
                            lines.append(f"{last_timestamp} | Speaker {current_speaker}")
                            lines.append(" ".join(current_text))
                            lines.append("")  # Empty line between speakers

                        # Start new speaker
                        current_speaker = speaker_id
                        current_text = [text]
                        last_timestamp = timestamp
                    else:
                        # Same speaker, accumulate text
                        current_text.append(text)

                # Add final speaker's text
                if current_speaker and current_text:
                    lines.append(f"{last_timestamp} | Speaker {current_speaker}")
                    lines.append(" ".join(current_text))

                result = "\n".join(lines)

                # If result is empty or only has speaker headers with no text, raise error
                if not result.strip() or result.count("|") == result.count("\n"):
                    raise ValueError(
                        "Transcript formatting resulted in empty content. "
                        "The Gong transcript may be empty or sentences have no text. "
                        f"Number of sentences processed: {len(transcript)}"
                    )

                # Count unique speakers
                unique_speakers = set()
                for line in lines:
                    if " | Speaker " in line:
                        speaker = line.split(" | Speaker ")[1]
                        unique_speakers.add(speaker)

                span.set_attribute("transcript.formatted_length", len(result))
                span.set_attribute("transcript.sentence_count", len(transcript))
                span.set_attribute("transcript.speaker_count", len(unique_speakers))
                # OpenInference output
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
            RuntimeError: If MCP request fails
        """
        with self.tracer.start_as_current_span(
            "fetch_gong_transcript",
            attributes={
                "gong.url": gong_url,
                # OpenInference input
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
                # OpenInference output
                span.set_attribute("output.value", formatted)
                span.set_attribute("output.mime_type", "text/plain")
                span.set_status(Status(StatusCode.OK))

                return formatted
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
