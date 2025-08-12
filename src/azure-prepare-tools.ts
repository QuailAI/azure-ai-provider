import {
  LanguageModelV2CallWarning,
  LanguageModelV2FunctionTool,
  LanguageModelV2ProviderDefinedTool,
} from "@ai-sdk/provider";
import { ChatCompletionsToolDefinition } from "@azure-rest/ai-inference";

type ToolChoiceV2 =
  | "auto"
  | "none"
  | "required"
  | { type: "tool"; toolName: string };

export function prepareToolsV2(
  toolsInput:
    | Array<LanguageModelV2FunctionTool | LanguageModelV2ProviderDefinedTool>
    | undefined,
  toolChoice:
    | { type: "auto" }
    | { type: "none" }
    | { type: "required" }
    | { type: "tool"; toolName: string }
    | undefined
): {
  tools: ChatCompletionsToolDefinition[] | undefined;
  tool_choice?: string | { type: "function"; function: { name: string } };
  toolWarnings: LanguageModelV2CallWarning[];
} {
  const toolWarnings: LanguageModelV2CallWarning[] = [];

  const tools = toolsInput?.length ? toolsInput : undefined;
  if (!tools) {
    return { tools: undefined, tool_choice: undefined, toolWarnings };
  }

  const azureTools: ChatCompletionsToolDefinition[] = [];
  for (const tool of tools) {
    if (tool.type !== "function") {
      toolWarnings.push({ type: "unsupported-tool", tool });
      continue;
    }
    const fn = tool as LanguageModelV2FunctionTool;
    azureTools.push({
      type: "function",
      function: {
        name: fn.name,
        description: fn.description ?? undefined,
        parameters: fn.inputSchema,
      },
    });
  }

  let tool_choice:
    | string
    | { type: "function"; function: { name: string } }
    | undefined = undefined;
  if (toolChoice?.type === "auto") {
    tool_choice = "auto";
  } else if (toolChoice?.type === "required") {
    toolWarnings.push({
      type: "unsupported-setting",
      setting: "toolChoice",
      details: "required toolChoice not supported",
    });
    tool_choice = "auto";
  } else if (toolChoice?.type === "none") {
    if (azureTools.length > 0) {
      toolWarnings.push({
        type: "unsupported-setting",
        setting: "toolChoice",
        details: "none toolChoice not supported when tools are provided",
      });
    }
    // Prefer to omit tools downstream if none.
  } else if (toolChoice?.type === "tool") {
    tool_choice = {
      type: "function",
      function: { name: toolChoice.toolName },
    };
  }

  return {
    tools: azureTools.length ? azureTools : undefined,
    tool_choice,
    toolWarnings,
  };
}

// Temporary alias to keep existing imports compiling during migration.
export const prepareTools = prepareToolsV2;
