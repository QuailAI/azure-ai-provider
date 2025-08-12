import {
  LanguageModelV2Prompt,
  UnsupportedFunctionalityError,
  LanguageModelV2TextPart,
  LanguageModelV2FilePart,
  LanguageModelV2ToolCallPart,
  LanguageModelV2ToolResultPart,
} from "@ai-sdk/provider";
import {
  ChatRequestMessage,
  ChatMessageContentItem,
} from "@azure-rest/ai-inference";

export function convertToAzureChatMessages(
  prompt: LanguageModelV2Prompt
): ChatRequestMessage[] {
  const messages: ChatRequestMessage[] = [];
  for (const message of prompt) {
    if (message.role === "system") {
      // System: text only
      messages.push({ role: "system", content: message.content });
    } else if (message.role === "user") {
      if (message.content.length === 1 && message.content[0].type === "text") {
        messages.push({
          role: message.role,
          content: (message.content[0] as LanguageModelV2TextPart).text,
        });
        continue;
      }

      const contentItems = message.content.map((part, index) => {
        switch (part.type) {
          case "text": {
            return {
              type: "text",
              text: (part as LanguageModelV2TextPart).text,
            } as ChatMessageContentItem;
          }
          case "file": {
            const filePart = part as LanguageModelV2FilePart;

            if (filePart.data instanceof URL) {
              if (filePart.mediaType.startsWith("image/")) {
                return {
                  type: "image_url",
                  image_url: { url: filePart.data.toString(), detail: "auto" },
                } as ChatMessageContentItem;
              }
              throw new UnsupportedFunctionalityError({
                functionality: `'File content parts with URL data for ${filePart.mediaType}' functionality not supported.`,
              });
            }

            const asBase64 = (data: string | Uint8Array | URL) => {
              if (data instanceof URL) {
                throw new UnsupportedFunctionalityError({
                  functionality: "URL data in file parts",
                });
              }
              return typeof data === "string"
                ? data
                : Buffer.from(data).toString("base64");
            };

            switch (filePart.mediaType) {
              case "audio/wav":
                return {
                  type: "input_audio",
                  input_audio: { data: asBase64(filePart.data), format: "wav" },
                } as ChatMessageContentItem;
              case "audio/mp3":
              case "audio/mpeg":
                return {
                  type: "input_audio",
                  input_audio: { data: asBase64(filePart.data), format: "mp3" },
                } as ChatMessageContentItem;
              case "application/pdf": {
                const partName = `part-${index}.pdf`;
                return {
                  type: "file",
                  file: {
                    filename: partName,
                    file_data: `data:application/pdf;base64,${asBase64(
                      filePart.data
                    )}`,
                  },
                } as ChatMessageContentItem;
              }
              default: {
                if (filePart.mediaType.startsWith("image/")) {
                  const base64 = asBase64(filePart.data);
                  return {
                    type: "image_url",
                    image_url: {
                      url: `data:${filePart.mediaType};base64,${base64}`,
                      detail: "auto",
                    },
                  } as ChatMessageContentItem;
                }
                throw new UnsupportedFunctionalityError({
                  functionality: `File content part type ${filePart.mediaType} in user messages`,
                });
              }
            }
          }
          default: {
            const exhaustiveCheck: never = part;
            throw new UnsupportedFunctionalityError({
              functionality: `Unsupported user content part type`,
            });
          }
        }
      });

      messages.push({ role: "user", content: contentItems });
    } else if (message.role === "assistant") {
      // Collect assistant text and tool-calls
      let assistantText = "";
      const toolCalls: Array<{
        id: string;
        type: "function";
        function: { name: string; arguments: string };
      }> = [];

      for (const part of message.content) {
        if (part.type === "text") {
          assistantText += (part as LanguageModelV2TextPart).text;
        } else if (part.type === "tool-call") {
          const p = part as LanguageModelV2ToolCallPart;
          const argsString =
            typeof p.input === "string"
              ? p.input
              : JSON.stringify(p.input ?? {});
          toolCalls.push({
            id: p.toolCallId,
            type: "function",
            function: {
              name: p.toolName,
              arguments: argsString,
            },
          });
        }
      }

      const assistantMsg: any = { role: "assistant", content: assistantText };
      if (toolCalls.length) assistantMsg.tool_calls = toolCalls;
      messages.push(assistantMsg);
    } else if (message.role === "tool") {
      // Emit one Azure tool message per tool-result
      for (const result of message.content) {
        const toolResult = result as LanguageModelV2ToolResultPart;
        let content: string;

        if (toolResult.output.type === "text") {
          content = toolResult.output.value;
        } else if (toolResult.output.type === "json") {
          content = JSON.stringify(toolResult.output.value);
        } else if (toolResult.output.type === "error-text") {
          content = toolResult.output.value;
        } else if (toolResult.output.type === "error-json") {
          content = JSON.stringify(toolResult.output.value);
        } else if (toolResult.output.type === "content") {
          content = toolResult.output.value
            .map((c) => (c.type === "text" ? c.text : "[media omitted]"))
            .join("\n");
        } else {
          content = "[unsupported output type]";
        }

        messages.push({
          role: "tool",
          content,
          tool_call_id: toolResult.toolCallId,
        });
      }
    } else {
      // Fallback for any unexpected role: treat as text
      const content = "";
      messages.push({ role: "user", content });
    }
  }
  return messages;
}
