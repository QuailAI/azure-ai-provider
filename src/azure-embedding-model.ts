import { EmbeddingModelV2 } from "@ai-sdk/provider";
import {
  AzureEmbeddingModelId,
  AzureEmbeddingSettings,
} from "./azure-embedding-settings";

type AzureEmbeddingConfig = {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string | undefined>;
};

export class AzureEmbeddingModel implements EmbeddingModelV2<string> {
  readonly specificationVersion = "v2" as const;
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

  async doEmbed({ values, abortSignal, headers }: any) {
    const response = await fetch(
      `${this.config.baseURL}/deployments/${this.modelId}/embeddings`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...this.config.headers(),
          ...headers,
        },
        body: JSON.stringify({
          input: values,
        }),
        signal: abortSignal,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to get embeddings: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      embeddings: data.data.map(
        (item: { embedding: number[] }) => item.embedding
      ),
      usage: {
        tokens: data.usage?.total_tokens,
      },
      rawResponse: {
        headers: Object.fromEntries(response.headers.entries()),
      },
    } as any;
  }
}
