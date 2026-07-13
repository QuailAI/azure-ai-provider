import { createAzure } from "./azure-ai-provider";
import { generateText, streamText } from "ai";
import { Readable } from "node:stream";
import { describe, it, expect, vi } from "vitest";
import { config } from "dotenv";
import type {
  LanguageModelV4Prompt,
  LanguageModelV4FunctionTool,
  LanguageModelV4Text,
} from "@ai-sdk/provider";
import { APICallError } from "@ai-sdk/provider";

config();

const hasCredentials = process.env.RUN_AZURE_INTEGRATION === "true" &&
  process.env.AZURE_API_KEY && process.env.AZURE_API_ENDPOINT;

function mockClient(
  model: ReturnType<ReturnType<typeof createAzure>["languageModel"]>,
  options: {
    response?: unknown;
    sse?: unknown[];
    streamStatus?: string;
  }
) {
  const post = vi.fn(() => {
    const request = Promise.resolve(
      options.response == null
        ? options.response
        : {
            headers: { "x-request-id": "request-1" },
            request: {
              url: "https://example.test/chat/completions",
              method: "POST",
            },
            ...options.response,
          }
    );
    return Object.assign(request, {
      asNodeStream: async () => ({
        status: options.streamStatus ?? "200",
        headers: { "x-request-id": "stream-request-1" },
        body: Readable.from(
          (options.sse ?? []).map(
            (event) =>
              Buffer.from(
                `data: ${
                  event === "[DONE]" ? event : JSON.stringify(event)
                }\n\n`
              )
          )
        ),
      }),
    });
  });

  Object.assign(model, {
    client: { path: vi.fn(() => ({ post })) },
  });
  return post;
}

describe("AzureChatLanguageModel", () => {
  const testPrompt: LanguageModelV4Prompt = [
    { role: "user", content: [{ type: "text", text: "Say hello" }] },
  ];

  it("generates through the AI SDK v7 public API", async () => {
    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "test-key",
    }).languageModel("chat-model");
    const post = mockClient(model, {
      response: {
        status: "200",
        body: {
          id: "response-1",
          model: "chat-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Hello from Azure" },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 4,
            completion_tokens: 3,
            total_tokens: 7,
          },
        },
      },
    });

    const result = await generateText({ model, prompt: "Say hello" });

    expect(result.text).toBe("Hello from Azure");
    expect(result.finishReason).toBe("stop");
    expect(result.usage.inputTokens).toBe(4);
    expect(result.usage.outputTokens).toBe(3);
    expect(post).toHaveBeenCalledWith(expect.objectContaining({
      body: expect.objectContaining({
        model: "chat-model",
        messages: [{ role: "user", content: "Say hello" }],
      }),
    }));
  });

  it("maps request settings and V4 warnings", async () => {
    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "test-key",
    }).languageModel("chat-model");
    const post = mockClient(model, {
      response: {
        status: "200",
        body: {
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "ok" },
              finish_reason: "length",
            },
          ],
        },
      },
    });

    const abortController = new AbortController();
    const result = await model.doGenerate({
      prompt: testPrompt,
      temperature: 0.2,
      maxOutputTokens: 100,
      stopSequences: ["stop"],
      topP: 0.9,
      topK: 20,
      presencePenalty: 0.1,
      frequencyPenalty: 0.2,
      seed: 1,
      responseFormat: {
        type: "json",
        name: "answer",
        description: "A structured answer",
        schema: {
          type: "object",
          properties: { answer: { type: "string" } },
          required: ["answer"],
        },
      },
      abortSignal: abortController.signal,
      headers: { "x-call-header": "call-value" },
    });

    expect(result.finishReason).toEqual({ unified: "length", raw: "length" });
    expect(result.warnings).toEqual([
      { type: "unsupported", feature: "topK" },
    ]);
    expect(result.response?.headers).toEqual({ "x-request-id": "request-1" });
    expect(post).toHaveBeenCalledWith({
      body: expect.objectContaining({
        temperature: 0.2,
        max_tokens: 100,
        stop: ["stop"],
        top_p: 0.9,
        presence_penalty: 0.1,
        frequency_penalty: 0.2,
        seed: 1,
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "answer",
            description: "A structured answer",
            schema: {
              type: "object",
              properties: { answer: { type: "string" } },
              required: ["answer"],
            },
          },
        },
      }),
      abortSignal: abortController.signal,
      headers: { "x-call-header": "call-value" },
    });
  });

  it("streams text, usage, and a choice-level finish reason", async () => {
    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "test-key",
    }).languageModel("chat-model");
    mockClient(model, {
      sse: [
        { choices: [{ index: 0, delta: { content: "Hello" } }] },
        {
          choices: [{ index: 0, delta: { content: " world" } }],
          usage: {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6,
          },
        },
        { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
        "[DONE]",
      ],
    });

    const result = await model.doStream({ prompt: testPrompt });
    const parts = [];
    for await (const part of result.stream) parts.push(part);

    expect(parts.map((part) => part.type)).toEqual([
      "stream-start",
      "text-start",
      "text-delta",
      "text-delta",
      "text-end",
      "finish",
    ]);
    expect(
      parts.filter((part) => part.type === "text-delta").map((part) => part.delta)
    ).toEqual(["Hello", " world"]);
    expect(parts.at(-1)).toMatchObject({
      type: "finish",
      finishReason: { unified: "stop", raw: "stop" },
      usage: {
        inputTokens: { total: 4 },
        outputTokens: { total: 2 },
      },
    });
  });

  it("streams through the AI SDK v7 public API", async () => {
    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "test-key",
    }).languageModel("chat-model");
    mockClient(model, {
      sse: [
        { choices: [{ index: 0, delta: { content: "Hello" } }] },
        { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
        "[DONE]",
      ],
    });

    const result = streamText({ model, prompt: "Say hello" });

    expect(await result.text).toBe("Hello");
    expect(await result.finishReason).toBe("stop");
  });

  it("assembles interleaved streamed tool calls by index", async () => {
    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "test-key",
    }).languageModel("chat-model");
    mockClient(model, {
      sse: [
        {
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: "call-a",
                    function: { name: "weather", arguments: '{"city":"' },
                  },
                  {
                    index: 1,
                    id: "call-b",
                    function: { name: "time", arguments: '{"zone":"' },
                  },
                ],
              },
            },
          ],
        },
        {
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  { index: 1, function: { arguments: 'UTC"}' } },
                  { index: 0, function: { arguments: 'Boston"}' } },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
        },
        "[DONE]",
      ],
    });

    const result = await model.doStream({ prompt: testPrompt });
    const parts = [];
    for await (const part of result.stream) parts.push(part);
    const calls = parts.filter((part) => part.type === "tool-call");

    expect(calls).toEqual([
      {
        type: "tool-call",
        toolCallId: "call-a",
        toolName: "weather",
        input: '{"city":"Boston"}',
      },
      {
        type: "tool-call",
        toolCallId: "call-b",
        toolName: "time",
        input: '{"zone":"UTC"}',
      },
    ]);
    expect(parts.at(-1)).toMatchObject({
      type: "finish",
      finishReason: { unified: "tool-calls", raw: "tool_calls" },
    });
  });

  it("rejects unsuccessful streaming responses", async () => {
    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "test-key",
    }).languageModel("chat-model");
    mockClient(model, { streamStatus: "401" });

    await expect(model.doStream({ prompt: testPrompt })).rejects.toThrow(
      "Failed to get chat completions: 401"
    );
  });

  it("returns a structured, retryable API error", async () => {
    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "test-key",
    }).languageModel("chat-model");
    mockClient(model, {
      response: {
        status: "429",
        body: {
          error: { code: "TooManyRequests", message: "Rate limited" },
        },
      },
    });

    const error = await model
      .doGenerate({ prompt: testPrompt })
      .then(() => undefined, (cause) => cause);

    expect(APICallError.isInstance(error)).toBe(true);
    expect(error).toMatchObject({
      message: "Rate limited",
      statusCode: 429,
      isRetryable: true,
    });
  });

  (hasCredentials ? it : it.skip)(
    "should generate text successfully",
    async () => {
      const provider = createAzure({
        endpoint: process.env.AZURE_API_ENDPOINT,
        apiKey: process.env.AZURE_API_KEY,
      });

      const model = provider.languageModel("Llama-3.3-70B-Instruct");
      const result = await model.doGenerate({
        prompt: testPrompt,
        temperature: 0,
        maxOutputTokens: 64,
      });

      const text = result.content.find(
        (c): c is LanguageModelV4Text => c.type === "text"
      )?.text;
      expect(text).toBeTruthy();
      expect(result.usage).toBeDefined();
      expect(result.finishReason).toBeDefined();
      console.log(result);
    }
  );

  (hasCredentials ? it : it.skip)(
    "should handle streaming responses",
    async () => {
      const provider = createAzure({
        endpoint: process.env.AZURE_API_ENDPOINT,
        apiKey: process.env.AZURE_API_KEY,
      });

      const model = provider.languageModel("Llama-3.3-70B-Instruct");
      const stream = await model.doStream({
        prompt: testPrompt,
        temperature: 0,
        maxOutputTokens: 64,
      });

      const chunks: string[] = [];
      for await (const chunk of stream.stream) {
        if (chunk.type === "text-delta") {
          chunks.push(chunk.delta);
        }
      }

      expect(chunks.join("")).toBeTruthy();
      console.log(chunks.join(""));
    }
  );

  (hasCredentials ? it : it.skip)("should handle tools correctly", async () => {
    const provider = createAzure({
      endpoint: process.env.AZURE_API_ENDPOINT,
      apiKey: process.env.AZURE_API_KEY,
    });

    const model = provider.languageModel("Llama-3.3-70B-Instruct");
    const tools: LanguageModelV4FunctionTool[] = [
      {
        type: "function",
        name: "getCurrentTime",
        description: "Get the current time",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
    ];

    const result = await model.doGenerate({
      prompt: testPrompt,
      tools,
      temperature: 0,
      maxOutputTokens: 64,
    });

    const text = result.content.find(
      (c): c is LanguageModelV4Text => c.type === "text"
    )?.text;
    expect(text).toBeTruthy();
    expect(result.warnings).toBeDefined();
    console.log(result);
  });
});
