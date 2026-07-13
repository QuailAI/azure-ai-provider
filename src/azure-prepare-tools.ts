import {
  LanguageModelV4FunctionTool,
  LanguageModelV4ProviderTool,
  SharedV4Warning,
} from "@ai-sdk/provider";
import { ChatCompletionsToolDefinition } from "@azure-rest/ai-inference";

export function prepareTools(
  toolsInput:
    | Array<LanguageModelV4FunctionTool | LanguageModelV4ProviderTool>
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
  toolWarnings: SharedV4Warning[];
} {
  const toolWarnings: SharedV4Warning[] = [];

  const tools = toolsInput?.length ? toolsInput : undefined;
  if (!tools) {
    return { tools: undefined, tool_choice: undefined, toolWarnings };
  }

  const azureTools: ChatCompletionsToolDefinition[] = [];
  for (const tool of tools) {
    if (tool.type !== "function") {
      toolWarnings.push({
        type: "unsupported",
        feature: `provider-defined tool ${tool.name}`,
      });
      continue;
    }
    const fn = tool as LanguageModelV4FunctionTool;
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
    tool_choice = "required";
  } else if (toolChoice?.type === "none") {
    tool_choice = "none";
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
