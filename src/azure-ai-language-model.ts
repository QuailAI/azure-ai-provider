import {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4StreamPart,
  LanguageModelV4Content,
  LanguageModelV4Usage,
  SharedV4Warning,
  APICallError,
} from "@ai-sdk/provider";
import { AzureChatModelId, AzureChatSettings } from "./azure-ai-settings";
import { mapAzureFinishReason } from "./map-azure-finish-reason";
import { convertToAzureChatMessages } from "./convert-to-azure-messages";
import { prepareTools } from "./azure-prepare-tools";
import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";
import { createSseStream } from "@azure/core-sse";

type AzureChatConfig = {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string | undefined>;
};

function normalizeCallHeaders(headers: Record<string, string | undefined> | undefined) {
  if (!headers) return undefined;
  return Object.fromEntries(
    Object.entries(headers).filter((entry): entry is [string, string] => entry[1] != null)
  );
}

export class AzureChatLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4" as const;

  readonly modelId: AzureChatModelId;
  readonly settings: AzureChatSettings;

  private readonly config: AzureChatConfig;
  private readonly client: ReturnType<typeof ModelClient>;

  constructor(
    modelId: AzureChatModelId,
    settings: AzureChatSettings,
    config: AzureChatConfig
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;

    const apiKey = config.headers()["api-key"];
    if (!apiKey) {
      throw new Error("Azure API key is required");
    }

    this.client = ModelClient(config.baseURL, new AzureKeyCredential(apiKey));
  }

  get provider(): string {
    return this.config.provider;
  }

  private getArgs({
    prompt,
    temperature,
    maxOutputTokens,
    stopSequences,
    tools,
    toolChoice,
    topP,
    topK,
    presencePenalty,
    frequencyPenalty,
    responseFormat,
    seed,
  }: LanguageModelV4CallOptions) {
    const warnings: SharedV4Warning[] = [];

    for (const [setting, value] of Object.entries({
      topK,
    })) {
      if (value !== undefined) {
        warnings.push({ type: "unsupported", feature: setting });
      }
    }

    const messages = convertToAzureChatMessages(prompt);

    const baseArgs = {
      messages,
      model: this.modelId,
      temperature,
      max_tokens: maxOutputTokens,
      stop: stopSequences,
      top_p: topP,
      presence_penalty: presencePenalty,
      frequency_penalty: frequencyPenalty,
      seed,
      response_format:
        responseFormat?.type === "json"
          ? responseFormat.schema
            ? {
                type: "json_schema" as const,
                json_schema: {
                  name: responseFormat.name ?? "response",
                  description: responseFormat.description,
                  schema: responseFormat.schema,
                },
              }
            : { type: "json_object" as const }
          : undefined,
    };

    const {
      tools: azureTools,
      tool_choice,
      toolWarnings,
    } = prepareTools(tools, toolChoice);

    return {
      args: {
        ...baseArgs,
        tools: azureTools,
        tool_choice,
        temperature: temperature ?? baseArgs.temperature,
      },
      warnings: [...warnings, ...toolWarnings],
    };
  }

  async doGenerate(options: LanguageModelV4CallOptions) {
    const { args, warnings } = this.getArgs(options);

    const response = await this.client.path("/chat/completions").post({
      body: args,
      abortSignal: options.abortSignal,
      headers: normalizeCallHeaders(options.headers),
    });

    if (isUnexpected(response)) {
      throw new APICallError({
        message: response.body.error.message,
        url: response.request.url,
        requestBodyValues: args,
        statusCode: Number(response.status),
        responseHeaders: response.headers,
        responseBody: JSON.stringify(response.body),
        data: response.body.error,
      });
    }

    const choice = response.body.choices[0];
    if (!choice) {
      throw new Error("Azure AI response did not contain a completion choice");
    }
    const content: LanguageModelV4Content[] = [];
    if (choice.message.content) {
      content.push({ type: "text", text: choice.message.content });
    }
    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        content.push({
          type: "tool-call",
          toolCallId: toolCall.id,
          toolName: toolCall.function.name,
          input: toolCall.function.arguments,
        });
      }
    }

    return {
      content,
      finishReason: mapAzureFinishReason(choice.finish_reason ?? "unknown"),
      usage: {
        inputTokens: {
          total: response.body.usage?.prompt_tokens,
          noCache: undefined,
          cacheRead: undefined,
          cacheWrite: undefined,
        },
        outputTokens: {
          total: response.body.usage?.completion_tokens,
          text: undefined,
          reasoning: undefined,
        },
      },
      request: { body: args },
      response: {
        id: response.body.id,
        timestamp:
          response.body.created == null
            ? undefined
            : new Date(response.body.created * 1000),
        modelId: response.body.model,
        headers: response.headers,
        body: response.body,
      },
      warnings,
    };
  }

  async doStream(options: LanguageModelV4CallOptions) {
    const { args, warnings } = this.getArgs(options);
    const body = { ...args, stream: true };

    const response = await this.client
      .path("/chat/completions")
      .post({
        body,
        abortSignal: options.abortSignal,
        headers: normalizeCallHeaders(options.headers),
      })
      .asNodeStream();

    if (!response.body || response.status !== "200") {
      throw new Error(`Failed to get chat completions: ${response.status}`);
    }

    const stream = createSseStream(response.body);
    let finishReason = mapAzureFinishReason(undefined);
    let usage: LanguageModelV4Usage = {
      inputTokens: {
        total: undefined,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: undefined,
        text: undefined,
        reasoning: undefined,
      },
    };

    const toolCalls = new Map<
      string,
      {
        name: string;
        input: string;
        id: string;
        hasStarted: boolean;
      }
    >();
    const toolCallIdsByIndex = new Map<number, string>();
    let lastToolCallId: string | undefined;

    // State for text accumulation to support reasoning middleware
    let currentTextId: string | undefined;
    let hasStartedText = false;

    return {
      stream: stream.pipeThrough(
        new TransformStream<{ data: string }, LanguageModelV4StreamPart>({
          start(controller) {
            controller.enqueue({ type: "stream-start", warnings });
          },
          transform(chunk, controller) {
            if (chunk.data === "[DONE]") {
              // Finish any ongoing text
              if (hasStartedText && currentTextId) {
                controller.enqueue({
                  type: "text-end",
                  id: currentTextId,
                });
              }

              // Send any remaining tool calls as complete
              for (const toolCall of toolCalls.values()) {
                if (toolCall.hasStarted) {
                  controller.enqueue({
                    type: "tool-input-end",
                    id: toolCall.id,
                  });

                  controller.enqueue({
                    type: "tool-call",
                    toolCallId: toolCall.id,
                    toolName: toolCall.name,
                    input: toolCall.input,
                  });
                }
              }

              controller.enqueue({
                type: "finish",
                finishReason,
                usage,
              });
              return;
            }

            const data = JSON.parse(chunk.data);

            if (data.usage) {
              usage = {
                inputTokens: {
                  total: data.usage.prompt_tokens,
                  noCache: undefined,
                  cacheRead: undefined,
                  cacheWrite: undefined,
                },
                outputTokens: {
                  total: data.usage.completion_tokens,
                  text: undefined,
                  reasoning: undefined,
                },
              };
            }

            const choice = data.choices?.[0];
            if (!choice) return;

            if (choice.delta?.tool_calls) {
              for (const toolCall of choice.delta.tool_calls) {
                const incomingId = toolCall.id;
                if (incomingId) {
                  lastToolCallId = incomingId;
                  if (toolCall.index != null) {
                    toolCallIdsByIndex.set(toolCall.index, incomingId);
                  }
                }
                let toolCallId =
                  incomingId ??
                  (toolCall.index == null
                    ? undefined
                    : toolCallIdsByIndex.get(toolCall.index)) ??
                  lastToolCallId;
                if (!toolCallId) {
                  toolCallId = `tool-${toolCalls.size + 1}`;
                }

                if (!toolCalls.has(toolCallId)) {
                  toolCalls.set(toolCallId, {
                    name: toolCall.function?.name || "",
                    input: "",
                    id: toolCallId,
                    hasStarted: false,
                  });
                }

                const existing = toolCalls.get(toolCallId)!;

                if (toolCall.function?.name && !existing.hasStarted) {
                  existing.name = toolCall.function.name;
                  existing.hasStarted = true;

                  controller.enqueue({
                    type: "tool-input-start",
                    id: toolCallId,
                    toolName: toolCall.function.name,
                  });
                }

                if (toolCall.function?.arguments) {
                  existing.input += toolCall.function.arguments;

                  controller.enqueue({
                    type: "tool-input-delta",
                    id: toolCallId,
                    delta: toolCall.function.arguments,
                  });
                }
              }
            }

            if (choice.delta?.content) {
              // Start text if not already started
              if (!hasStartedText) {
                currentTextId = `text-${Date.now()}`;
                hasStartedText = true;
                controller.enqueue({
                  type: "text-start",
                  id: currentTextId,
                });
              }

              // Emit text delta
              controller.enqueue({
                type: "text-delta",
                id: currentTextId!,
                delta: choice.delta.content,
              });
            }

            const rawFinishReason =
              choice.finish_reason ?? choice.delta?.finish_reason;
            if (rawFinishReason) {
              finishReason = mapAzureFinishReason(rawFinishReason);
            }
          },
        })
      ),
      request: { body },
      response: { headers: response.headers },
    };
  }

  get supportedUrls() {
    return {} as Record<string, RegExp[]>;
  }
}
