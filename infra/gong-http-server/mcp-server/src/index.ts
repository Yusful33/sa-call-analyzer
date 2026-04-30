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
        
        // If no dates provided, default to last 90 days
        let fromDateTime = parsed.fromDateTime;
        let toDateTime = parsed.toDateTime;
        
        if (!fromDateTime || !toDateTime) {
          const now = new Date();
          const ninetyDaysAgo = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);
          
          if (!fromDateTime) {
            fromDateTime = ninetyDaysAgo.toISOString();
          }
          if (!toDateTime) {
            toDateTime = now.toISOString();
          }
        }
        
        // Strategy: Use /calls endpoint with GET to list calls, then get full details with /calls/extensive
        let callIds: string[] = [];

        try {
          // Step 1: Get call IDs using the /calls endpoint (GET with query parameters)
          // Gong API v2 /calls uses GET method with fromDateTime and toDateTime as query params
          // IMPORTANT: Gong API returns max 100 records per page - must handle pagination via cursor
          let cursor: string | undefined = undefined;

          do {
            const queryParams = new URLSearchParams();
            queryParams.set("fromDateTime", fromDateTime);
            queryParams.set("toDateTime", toDateTime);
            if (cursor) {
              queryParams.set("cursor", cursor);
            }

            const listResult = await gongRequest(`/calls?${queryParams.toString()}`, "GET");

            // Extract call IDs from this page
            // Response format: { calls: [...], records: { totalRecords, currentPageSize, currentPageNumber, cursor? } }
            if (listResult.calls && Array.isArray(listResult.calls)) {
              const pageIds = listResult.calls.map((call: any) => call.id).filter(Boolean);
              callIds.push(...pageIds);
            }

            // Check for next page - cursor exists when there are more results
            cursor = listResult.records?.cursor;

            console.error(`Gong API: Fetched ${callIds.length} calls so far (page cursor: ${cursor ? 'yes' : 'no'})`);

          } while (cursor);

          console.error(`Gong API: Total calls fetched: ${callIds.length}`);

        } catch (listError: any) {
          console.error("Error listing calls:", listError);
          const errorMessage = listError instanceof Error ? listError.message : String(listError);
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  calls: [],
                  error: errorMessage,
                  attemptedEndpoint: "/calls (GET)"
                }),
              },
            ],
          };
        }
        
        if (callIds.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ calls: [], total: 0 }),
              },
            ],
          };
        }
        
        // Step 2: Get full call details using /calls/extensive (same endpoint as get_call_info)
        // This gives us account names and all metadata
        // Process in batches to avoid hitting API limits
        const batchSize = 50;
        const allCalls: any[] = [];
        
        for (let i = 0; i < callIds.length; i += batchSize) {
          const batch = callIds.slice(i, i + batchSize);
          
          try {
            const extensiveBody = {
              filter: {
                callIds: batch
              },
              contentSelector: {
                exposedFields: {
                  parties: true,
                  content: {
                    structure: true
                  }
                }
              }
            };

            const extensiveResult = await gongRequest("/calls/extensive", "POST", extensiveBody);
            
            if (extensiveResult.calls && Array.isArray(extensiveResult.calls)) {
              // Format calls similar to get_call_info
              const formattedCalls = extensiveResult.calls.map((call: any) => {
                const meta = call.metaData || call;
                const accountName = 
                  call.accountName || 
                  call.account?.name || 
                  meta.accountName ||
                  meta.account?.name ||
                  null;
                
                return {
                  id: meta.id,
                  title: meta.title,
                  scheduled: meta.scheduled,
                  started: meta.started,
                  duration: meta.duration,
                  url: meta.url,
                  direction: meta.direction,
                  accountName: accountName,
                  parties: call.parties || [],
                };
              });
              
              allCalls.push(...formattedCalls);
            }
          } catch (extensiveError: any) {
            console.error(`Error getting extensive call details for batch:`, extensiveError);
            // Continue with other batches
          }
        }
        
        const result = {
          calls: allCalls,
          total: allCalls.length
        };
        
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
          contentSelector: {
            exposedFields: {
              parties: true,
              content: {
                structure: true
              }
            }
          }
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
        
        // Extract account name from various possible fields
        // Gong API may store account name in different places
        const accountName = 
          call.accountName || 
          call.account?.name || 
          call.accountName || 
          meta.accountName ||
          meta.account?.name ||
          null;
        
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
          accountName: accountName,
          // Include full call object for debugging
          _raw: call,
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

