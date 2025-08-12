import {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2CallWarning,
  LanguageModelV2StreamPart,
  LanguageModelV2Content,
} from "@ai-sdk/provider";
import { AzureChatModelId, AzureChatSettings } from "./azure-ai-settings";
import { mapAzureFinishReason } from "./map-azure-finish-reason";
import { convertToAzureChatMessages } from "./convert-to-azure-messages";
import { prepareToolsV2 as prepareTools } from "./azure-prepare-tools";
import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";
import { createSseStream } from "@azure/core-sse";

type AzureChatConfig = {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string | undefined>;
};

export class AzureChatLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2" as const;

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
  }: LanguageModelV2CallOptions) {
    const warnings: LanguageModelV2CallWarning[] = [];

    const messages = convertToAzureChatMessages(prompt);

    const baseArgs = {
      messages,
      model: this.modelId,
      temperature,
      max_tokens: maxOutputTokens,
      stop: stopSequences,
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

  async doGenerate(options: LanguageModelV2CallOptions) {
    const { args, warnings } = this.getArgs(options);

    const response = await this.client.path("/chat/completions").post({
      body: args,
    });

    if (isUnexpected(response)) {
      throw response.body.error;
    }

    const choice = response.body.choices[0];
    const content: LanguageModelV2Content[] = [];
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
        inputTokens: response.body.usage?.prompt_tokens,
        outputTokens: response.body.usage?.completion_tokens,
        totalTokens: response.body.usage?.total_tokens,
      },
      request: { body: args },
      response: { body: response.body },
      warnings,
    };
  }

  async doStream(options: LanguageModelV2CallOptions) {
    const { args, warnings } = this.getArgs(options);
    const body = { ...args, stream: true };

    const response = await this.client
      .path("/chat/completions")
      .post({
        body,
      })
      .asNodeStream();

    if (!response.body || response.status !== "200") {
      throw new Error(`Failed to get chat completions: ${response.status}`);
    }

    const stream = createSseStream(response.body);
    let finishReason: ReturnType<typeof mapAzureFinishReason> = "unknown";
    let usage: {
      inputTokens: number | undefined;
      outputTokens: number | undefined;
      totalTokens: number | undefined;
    } = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
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
    let lastToolCallId: string | undefined;

    return {
      stream: stream.pipeThrough(
        new TransformStream<{ data: string }, LanguageModelV2StreamPart>({
          start(controller) {
            controller.enqueue({ type: "stream-start", warnings });
          },
          transform(chunk, controller) {
            if (chunk.data === "[DONE]") {
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
                inputTokens: data.usage.prompt_tokens,
                outputTokens: data.usage.completion_tokens,
                totalTokens: data.usage.total_tokens,
              };
            }

            const choice = data.choices?.[0];
            if (!choice) return;

            if (choice.delta?.tool_calls) {
              for (const toolCall of choice.delta.tool_calls) {
                const incomingId = toolCall.id;
                if (incomingId) lastToolCallId = incomingId;
                let toolCallId = incomingId ?? lastToolCallId;
                if (!toolCallId) {
                  toolCallId = `tool-${toolCalls.size + 1}`;
                }

                if (!toolCalls.has(toolCallId)) {
                  toolCalls.set(toolCallId, {
                    name: toolCall.function?.name || "",
                    input: toolCall.function?.arguments || "",
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
              const textId = `text-${Date.now()}`;
              controller.enqueue({
                type: "text-start",
                id: textId,
              });
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: choice.delta.content,
              });
              controller.enqueue({
                type: "text-end",
                id: textId,
              });
            }

            if (choice.delta?.finish_reason) {
              finishReason = mapAzureFinishReason(choice.delta.finish_reason);
            }
          },
        })
      ),
      request: { body },
    };
  }

  get supportedUrls() {
    return {} as Record<string, RegExp[]>;
  }
}
