"""
HTTP wrapper for the Gong MCP server.
Translates HTTP requests to MCP JSON-RPC protocol and communicates with the MCP server via stdio.
"""
import os
import json
import subprocess
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Gong MCP HTTP Wrapper", version="1.0.0")

# Path to the MCP server - can be configured via env var
MCP_SERVER_CMD = os.getenv("MCP_SERVER_CMD", "node /app/dist/index.js")


class MCPClient:
    """Client for communicating with MCP server via stdio."""
    
    def __init__(self):
        self.request_id = 0
    
    def call_tool(self, method: str, params: Dict[str, Any]) -> Dict:
        """
        Send a JSON-RPC request to the MCP server.
        
        Args:
            method: MCP tool name (e.g., "list_calls", "retrieve_transcripts")
            params: Tool parameters
            
        Returns:
            Response dictionary from MCP server
        """
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
                    "name": "gong-http-wrapper",
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
        
        # Run MCP server and send requests via stdin
        result = subprocess.run(
            MCP_SERVER_CMD.split(),
            input=requests_json,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"MCP server error: {result.stderr}")
        
        # Parse response (newline-delimited JSON)
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
            try:
                response = json.loads(line)
                # Look for the tool call response (not the initialize response)
                if response.get("id") == self.request_id:
                    if "result" in response:
                        return response["result"]
                    elif "error" in response:
                        raise RuntimeError(f"MCP error: {response['error']}")
            except json.JSONDecodeError:
                continue
        
        raise RuntimeError(f"No valid response from MCP server: {result.stdout}")


# Global MCP client
mcp_client = MCPClient()


class TranscriptRequest(BaseModel):
    call_id: str


class ListCallsRequest(BaseModel):
    from_date: Optional[str] = None
    to_date: Optional[str] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/transcript")
async def get_transcript(request: TranscriptRequest):
    """
    Retrieve transcript for a specific call ID via MCP protocol.
    
    Args:
        request: TranscriptRequest with call_id
        
    Returns:
        Transcript data from Gong via MCP
    """
    try:
        result = mcp_client.call_tool("retrieve_transcripts", {"callIds": [request.call_id]})
        
        # MCP server wraps response in content array
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "{}")
                return json.loads(text_content)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calls")
async def list_calls(request: ListCallsRequest):
    """
    List Gong calls with optional date filtering via MCP protocol.
    
    Args:
        request: ListCallsRequest with optional date filters
        
    Returns:
        List of calls from Gong via MCP
    """
    try:
        params = {}
        if request.from_date:
            params["fromDateTime"] = request.from_date
        if request.to_date:
            params["toDateTime"] = request.to_date
        
        result = mcp_client.call_tool("list_calls", params)
        
        # MCP server wraps response in content array
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "{}")
                return json.loads(text_content)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

