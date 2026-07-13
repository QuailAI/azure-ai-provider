import { describe, expect, it } from "vitest";
import type { LanguageModelV4Prompt } from "@ai-sdk/provider";
import { convertToAzureChatMessages } from "./convert-to-azure-messages";
import { createAzure } from "./azure-ai-provider";
import { prepareTools } from "./azure-prepare-tools";
import { mapAzureFinishReason } from "./map-azure-finish-reason";

describe("AI SDK v7 provider contract", () => {
  it("exposes V4 language and embedding models", () => {
    const provider = createAzure({
      endpoint: "https://example.test/models/",
      apiKey: "test-key",
    });

    expect(provider.specificationVersion).toBe("v4");
    expect(provider("chat-model").specificationVersion).toBe("v4");
    expect(provider.embeddingModel("embedding-model").specificationVersion).toBe(
      "v4"
    );
  });

  it("maps raw Azure finish reasons to structured V4 reasons", () => {
    expect(mapAzureFinishReason("tool_calls")).toEqual({
      unified: "tool-calls",
      raw: "tool_calls",
    });
    expect(mapAzureFinishReason("future_reason")).toEqual({
      unified: "other",
      raw: "future_reason",
    });
  });

  it("converts V4 tool calls and results to Azure messages", () => {
    const prompt: LanguageModelV4Prompt = [
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "call-1",
            toolName: "weather",
            input: { city: "Boston" },
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call-1",
            toolName: "weather",
            output: { type: "json", value: { temperature: 20 } },
          },
        ],
      },
    ];

    expect(convertToAzureChatMessages(prompt)).toEqual([
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call-1",
            type: "function",
            function: {
              name: "weather",
              arguments: '{"city":"Boston"}',
            },
          },
        ],
      },
      {
        role: "tool",
        content: '{"temperature":20}',
        tool_call_id: "call-1",
      },
    ]);
  });

  it("converts function tools and named tool choice", () => {
    const result = prepareTools(
      [
        {
          type: "function",
          name: "weather",
          description: "Get weather",
          inputSchema: { type: "object", properties: {} },
        },
      ],
      { type: "tool", toolName: "weather" }
    );

    expect(result.tools?.[0]).toMatchObject({
      type: "function",
      function: { name: "weather" },
    });
    expect(result.tool_choice).toEqual({
      type: "function",
      function: { name: "weather" },
    });
    expect(result.toolWarnings).toEqual([]);
  });

  it.each(["auto", "required", "none"] as const)(
    "passes through the %s tool choice",
    (choice) => {
      const result = prepareTools(
        [
          {
            type: "function",
            name: "weather",
            inputSchema: { type: "object", properties: {} },
          },
        ],
        { type: choice }
      );

      expect(result.tool_choice).toBe(choice);
      expect(result.toolWarnings).toEqual([]);
    }
  );
});
