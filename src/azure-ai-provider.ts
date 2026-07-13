import {
  EmbeddingModelV4,
  LanguageModelV4,
  ProviderV4,
} from "@ai-sdk/provider";
import { loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils";
import { AzureChatLanguageModel } from "./azure-ai-language-model";
import { AzureChatModelId, AzureChatSettings } from "./azure-ai-settings";
import { AzureEmbeddingModel } from "./azure-embedding-model";
import {
  AzureEmbeddingModelId,
  AzureEmbeddingSettings,
} from "./azure-embedding-settings";

export interface AzureProvider extends ProviderV4 {
  (modelId: AzureChatModelId, settings?: AzureChatSettings): LanguageModelV4;

  /**
   * Creates a model for text generation.
   */
  languageModel(
    modelId: AzureChatModelId,
    settings?: AzureChatSettings
  ): LanguageModelV4;

  /**
   * Creates a model for text generation.
   */
  chat(
    modelId: AzureChatModelId,
    settings?: AzureChatSettings
  ): LanguageModelV4;

  /**
   * @deprecated Use `embeddingModel()` instead.
   */
  embedding(
    modelId: AzureEmbeddingModelId,
    settings?: AzureEmbeddingSettings
  ): EmbeddingModelV4;

  /**
   * @deprecated Use `embeddingModel()` instead.
   */
  textEmbedding(
    modelId: AzureEmbeddingModelId,
    settings?: AzureEmbeddingSettings
  ): EmbeddingModelV4;

  embeddingModel(
    modelId: AzureEmbeddingModelId,
    settings?: AzureEmbeddingSettings
  ): EmbeddingModelV4;

  textEmbeddingModel: (
    modelId: AzureEmbeddingModelId,
    settings?: AzureEmbeddingSettings
  ) => EmbeddingModelV4;
}

export interface AzureProviderSettings {
  /**
   * The Azure AI endpoint (e.g., "https://<resource>.services.ai.azure.com/models")
   */
  endpoint?: string;

  /**
   * The API key for Azure OpenAI service
   */
  apiKey?: string;

  /**
   * The API version to use
   * @default "2024-02-15-preview"
   */
  apiVersion?: string;

  /**
   * Custom headers to include in the requests
   */
  headers?: Record<string, string>;
}

export function createAzure(
  options: AzureProviderSettings = {}
): AzureProvider {
  const baseURL =
    withoutTrailingSlash(options.endpoint) ??
    "https://<resource>.services.ai.azure.com/models";

  const getHeaders = () => ({
    "api-key": loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: "AZURE_API_KEY",
      description: "Azure AI",
    }),
    "api-version": options.apiVersion ?? "2024-02-15-preview",
    ...options.headers,
  });

  const createChatModel = (
    modelId: AzureChatModelId,
    settings: AzureChatSettings = {}
  ) =>
    new AzureChatLanguageModel(modelId, settings, {
      provider: "azure.ai",
      baseURL,
      headers: getHeaders,
    });

  const createEmbeddingModel = (
    modelId: AzureEmbeddingModelId,
    settings: AzureEmbeddingSettings = {}
  ) =>
    new AzureEmbeddingModel(modelId, settings, {
      provider: "azure.embedding",
      baseURL,
      headers: getHeaders,
    });

  const provider = function (
    modelId: AzureChatModelId,
    settings?: AzureChatSettings
  ) {
    if (new.target) {
      throw new Error(
        "The Azure model function cannot be called with the new keyword."
      );
    }

    return createChatModel(modelId, settings);
  };

  provider.languageModel = createChatModel;
  provider.specificationVersion = "v4" as const;
  provider.chat = createChatModel;
  provider.embedding = createEmbeddingModel;
  provider.embeddingModel = createEmbeddingModel;
  provider.textEmbedding = createEmbeddingModel;
  provider.textEmbeddingModel = createEmbeddingModel;
  provider.imageModel = function imageModel() {
    throw new Error("Image model is not implemented for Azure provider.");
  } as AzureProvider["imageModel"];

  return provider;
}

/**
 * Default Azure AI provider instance
 */
export const azureProvider = createAzure();
