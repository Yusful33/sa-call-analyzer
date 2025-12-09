# Gong Integration Guide

This guide explains how to use the Gong API integration to automatically fetch call transcripts.

## Prerequisites

1. **Gong MCP Server running in Docker** ✅ (You have this)
2. **Gong API credentials** in `.env` file ✅ (Already configured)

## How It Works

The app can now accept Gong call URLs instead of manual transcript text. When you provide a Gong URL:

1. The app extracts the call ID from the URL
2. Communicates with your local Gong MCP Docker container
3. Fetches the full transcript with timestamps and speakers
4. Analyzes it with the CrewAI multi-agent system

## Usage

### Option 1: Using Gong URL (New!)

Send a POST request to `/api/analyze` with a Gong URL:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "gong_url": "https://app.gong.io/call?id=YOUR_CALL_ID"
  }'
```

### Option 2: Manual Transcript (Original)

You can still paste transcripts manually:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "0:16 | John\nHello everyone..."
  }'
```

## Supported Gong URL Formats

The integration supports both regular and embedded Gong URLs:

- **Regular**: `https://app.gong.io/call?id=7782342274025937895`
- **Embedded**: `https://subdomain.app.gong.io/embedded-call?call-id=12345`

## Configuration

Your `.env` file should have:

```env
# Gong API credentials (already configured)
GONG_ACCESS_KEY=GJP4SAOFBN4HUQTI3OGLTKJQLDD23T6G
GONG_SECRET_KEY=eyJhbGci...
```

## Verifying the Integration

Test the Gong MCP client:

```bash
uv run python test_gong_mcp.py
```

You should see:
```
✅ GongMCPClient initialized
✅ Found X calls
✅ Gong MCP client is working!
```

## Troubleshooting

### "Gong MCP client not available"

Check that your Gong MCP Docker container is running:

```bash
docker ps | grep gong
```

Should show:
```
bold_mestorf      gong-mcp      Up X minutes
```

### "Failed to fetch transcript from Gong"

1. Verify the Gong URL format is correct
2. Ensure the call ID exists in your Gong account
3. Check that Gong API credentials are valid

### Restart the Gong MCP container

```bash
docker restart bold_mestorf
```

## Architecture

```
Your FastAPI App
    ↓ (Python subprocess)
Gong MCP Docker Container (bold_mestorf)
    ↓ (HTTP API)
Gong API (https://api.gong.io)
```

The integration uses JSON-RPC protocol to communicate with the MCP server via `docker exec`.

## Files Added

- `gong_mcp_client.py` - Python client for Gong MCP server
- `test_gong_mcp.py` - Integration test script
- `models.py` - Updated to support `gong_url` field
- `main.py` - Updated `/api/analyze` endpoint to handle Gong URLs
