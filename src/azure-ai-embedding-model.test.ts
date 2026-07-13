import { describe, it, expect, vi, afterEach } from "vitest";
import { createAzure } from "./azure-ai-provider";

describe("AzureEmbeddingModel", () => {
  afterEach(() => vi.restoreAllMocks());

  it("implements the V4 embedding contract", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [{ embedding: [0.1, 0.2] }, { embedding: [0.3, 0.4] }],
          usage: { total_tokens: 7 },
        }),
        { status: 200, headers: { "x-request-id": "request-1" } }
      )
    );

    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "test-key",
    }).embeddingModel("embedding-model");
    const result = await model.doEmbed({ values: ["one", "two"] });

    expect(model.specificationVersion).toBe("v4");
    expect(result.embeddings).toEqual([[0.1, 0.2], [0.3, 0.4]]);
    expect(result.usage).toEqual({ tokens: 7 });
    expect(result.warnings).toEqual([]);
    expect(result.response?.headers?.["x-request-id"]).toBe("request-1");
    expect(fetch).toHaveBeenCalledWith(
      "https://example.test/models/embeddings",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          input: ["one", "two"],
          model: "embedding-model",
        }),
      })
    );
  });

  it("surfaces unsuccessful embedding responses", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(null, { status: 401, statusText: "Unauthorized" })
    );
    const model = createAzure({
      endpoint: "https://example.test/models",
      apiKey: "bad-key",
    }).embeddingModel("embedding-model");

    await expect(model.doEmbed({ values: ["one"] })).rejects.toThrow(
      "Failed to get embeddings: Unauthorized"
    );
  });
});
