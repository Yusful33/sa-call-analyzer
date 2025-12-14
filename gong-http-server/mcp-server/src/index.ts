#!/usr/bin/env node
/**
 * Custom Gong MCP Server
 * 
 * Provides MCP tools for interacting with the Gong API:
 * - list_calls: List calls with optional date filters
 * - retrieve_transcripts: Get transcript for specific call IDs
 * - get_call_info: Get detailed call metadata by ID (NEW)
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

// Gong API configuration
const GONG_API_BASE = "https://api.gong.io/v2";

// Get credentials from environment
function getCredentials(): string {
  const accessKey = process.env.GONG_ACCESS_KEY;
  const secretKey = process.env.GONG_ACCESS_SECRET || process.env.GONG_SECRET_KEY;
  
  if (!accessKey || !secretKey) {
    throw new Error("GONG_ACCESS_KEY and GONG_ACCESS_SECRET/GONG_SECRET_KEY must be set");
  }
  
  return Buffer.from(`${accessKey}:${secretKey}`).toString("base64");
}

// Make authenticated request to Gong API
async function gongRequest(endpoint: string, method: string = "GET", body?: object): Promise<any> {
  const credentials = getCredentials();
  
  const options: RequestInit = {
    method,
    headers: {
      "Authorization": `Basic ${credentials}`,
      "Content-Type": "application/json",
    },
  };
  
  if (body) {
    options.body = JSON.stringify(body);
  }
  
  const response = await fetch(`${GONG_API_BASE}${endpoint}`, options);
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gong API error ${response.status}: ${errorText}`);
  }
  
  return response.json();
}

// Tool input schemas
const ListCallsSchema = z.object({
  fromDateTime: z.string().optional().describe("Start date in ISO format (e.g., 2024-01-01T00:00:00Z)"),
  toDateTime: z.string().optional().describe("End date in ISO format"),
});

const RetrieveTranscriptsSchema = z.object({
  callIds: z.array(z.string()).describe("Array of call IDs to retrieve transcripts for"),
});

const GetCallInfoSchema = z.object({
  callId: z.string().describe("The Gong call ID to get metadata for"),
});

// Create MCP server
const server = new Server(
  {
    name: "gong-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list_calls",
        description: "List Gong calls with optional date filtering. Returns call IDs, titles, and basic metadata.",
        inputSchema: {
          type: "object",
          properties: {
            fromDateTime: {
              type: "string",
              description: "Start date in ISO format (e.g., 2024-01-01T00:00:00Z)",
            },
            toDateTime: {
              type: "string",
              description: "End date in ISO format",
            },
          },
        },
      },
      {
        name: "retrieve_transcripts",
        description: "Retrieve transcripts for specific Gong calls by their IDs.",
        inputSchema: {
          type: "object",
          properties: {
            callIds: {
              type: "array",
              items: { type: "string" },
              description: "Array of call IDs to retrieve transcripts for",
            },
          },
          required: ["callIds"],
        },
      },
      {
        name: "get_call_info",
        description: "Get detailed metadata for a specific call including scheduled date, title, duration, URL, and participants.",
        inputSchema: {
          type: "object",
          properties: {
            callId: {
              type: "string",
              description: "The Gong call ID to get metadata for",
            },
          },
          required: ["callId"],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "list_calls": {
        const parsed = ListCallsSchema.parse(args);
        const body: any = { filter: {} };
        
        if (parsed.fromDateTime) {
          body.filter.fromDateTime = parsed.fromDateTime;
        }
        if (parsed.toDateTime) {
          body.filter.toDateTime = parsed.toDateTime;
        }
        
        const result = await gongRequest("/calls", "POST", body);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case "retrieve_transcripts": {
        const parsed = RetrieveTranscriptsSchema.parse(args);
        
        const body = {
          filter: {
            callIds: parsed.callIds,
          },
        };
        
        const result = await gongRequest("/calls/transcript", "POST", body);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case "get_call_info": {
        const parsed = GetCallInfoSchema.parse(args);
        
        const body = {
          filter: {
            callIds: [parsed.callId],
          },
        };
        
        const result = await gongRequest("/calls/extensive", "POST", body);
        
        // Extract and format call metadata
        const calls = result.calls || [];
        if (calls.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ error: "Call not found" }),
              },
            ],
          };
        }
        
        const call = calls[0];
        const meta = call.metaData || call;
        
        // Return structured call info
        const callInfo = {
          id: meta.id,
          title: meta.title,
          scheduled: meta.scheduled,
          started: meta.started,
          duration: meta.duration,
          url: meta.url,
          direction: meta.direction,
          scope: meta.scope,
          media: meta.media,
          language: meta.language,
          workspaceId: meta.workspaceId,
          parties: call.parties || [],
        };
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(callInfo, null, 2),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ error: errorMessage }),
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Gong MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
