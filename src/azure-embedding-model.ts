import {
  EmbeddingModelV4,
  EmbeddingModelV4CallOptions,
  EmbeddingModelV4Result,
} from "@ai-sdk/provider";
import {
  AzureEmbeddingModelId,
  AzureEmbeddingSettings,
} from "./azure-embedding-settings";

type AzureEmbeddingConfig = {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string | undefined>;
};

export class AzureEmbeddingModel implements EmbeddingModelV4 {
  readonly specificationVersion = "v4" as const;
  readonly maxEmbeddingsPerCall: number | undefined;
  readonly supportsParallelCalls = true;

  readonly modelId: AzureEmbeddingModelId;
  readonly settings: AzureEmbeddingSettings;

  private readonly config: AzureEmbeddingConfig;

  constructor(
    modelId: AzureEmbeddingModelId,
    settings: AzureEmbeddingSettings,
    config: AzureEmbeddingConfig
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
    this.maxEmbeddingsPerCall = settings.maxEmbeddingsPerCall;
  }

  get provider(): string {
    return this.config.provider;
  }

  async doEmbed({
    values,
    abortSignal,
    headers,
  }: EmbeddingModelV4CallOptions): Promise<EmbeddingModelV4Result> {
    const response = await fetch(`${this.config.baseURL}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...this.config.headers(),
        ...headers,
      },
      body: JSON.stringify({
        input: values,
        model: this.modelId,
      }),
      signal: abortSignal,
    });

    if (!response.ok) {
      throw new Error(`Failed to get embeddings: ${response.statusText}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
      usage?: { total_tokens?: number };
    };
    return {
      embeddings: data.data.map((item) => item.embedding),
      usage:
        data.usage?.total_tokens == null
          ? undefined
          : { tokens: data.usage.total_tokens },
      response: {
        headers: Object.fromEntries(response.headers.entries()),
        body: data,
      },
      warnings: [],
    };
  }
}
