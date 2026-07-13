import type { LanguageModelV4FinishReason } from "@ai-sdk/provider";

export function mapAzureFinishReason(
  finishReason: string | null | undefined
): LanguageModelV4FinishReason {
  let unified: LanguageModelV4FinishReason["unified"];
  switch (finishReason) {
    case "stop":
      unified = "stop";
      break;
    case "length":
    case "model_length":
      unified = "length";
      break;
    case "tool_calls":
      unified = "tool-calls";
      break;
    case "content_filter":
      unified = "content-filter";
      break;
    case "error":
      unified = "error";
      break;
    default:
      unified = "other";
  }

  return { unified, raw: finishReason ?? undefined };
}
