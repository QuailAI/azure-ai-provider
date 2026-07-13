# AI SDK - Azure Custom Provider

The Azure custom provider for the [AI SDK](https://sdk.vercel.ai/docs) enables integration with Azure-hosted language models that use Azure's native APIs instead of the OpenAI API format.

## Status

- ✅ Chat Completions: Working with both streaming and non-streaming responses
- ✅ Vision/Images: Support for image inputs with compatible models
- ⚠️ Tool Calling: Implementation present but functionality depends on model
  capabilities
- ⚠️ Embeddings: Implementation present but untested

## Installation

The Azure provider is available in the `@ai-sdk/azure` module. You can install it with

```bash
npm i @quail-ai/azure-ai-provider
```

## Setup

1. Create an Azure AI resource and get your endpoint URL and API key
2. Set up environment variables in your .env file:

```env
AZURE_API_ENDPOINT=https://<your-resource-endpoint>
AZURE_API_KEY=<your-api-key>
```

## Basic Usage

```ts
import { createAzure } from "@quail-ai/azure-ai-provider";
import { generateText } from "ai";

// Create provider instance
const azure = createAzure({
  endpoint: process.env.AZURE_API_ENDPOINT,
  apiKey: process.env.AZURE_API_KEY,
});

// Generate text
const { text } = await generateText({
  model: azure("your-deployment-name"),
  prompt: "Write a story about a robot.",
});
```

## Advanced Features

### Streaming Responses

```ts
import { streamText } from "ai";

const stream = await streamText({
  model: azure("your-deployment-name"),
  prompt: "Generate a long story...",
});

for await (const chunk of stream.textStream) {
  process.stdout.write(chunk);
}
```

### Tool Calling

```ts
import { generateText } from "ai";
import { z } from "zod";

const result = await generateText({
  model: azure("your-deployment-name"),
  messages: [{ role: "user", content: "What's the weather?" }],
  tools: {
    get_weather: {
      description: "Get weather for location",
      inputSchema: z.object({
        location: z.string(),
      }),
      execute: async ({ location }) => {
        return `Weather in ${location}: Sunny`;
      },
    },
  },
});
```

### Text Embeddings

```ts
const model = azure.embeddingModel("your-embedding-deployment");

const result = await model.doEmbed({
  values: ["Encode this text"],
});
```

### Image Processing

```ts
import fs from "fs/promises";
import { generateText } from "ai";

// Load image data
const imageData = await fs.readFile("path/to/image.png");

const result = await generateText({
  model: azure("gpt-4o"), // Use a model that supports vision
  messages: [
    {
      role: "user",
      content: [
        {
          type: "image",
          image: imageData, // Pass the raw Buffer
          mimeType: "image/png",
        },
        {
          type: "text",
          text: "Describe this image in detail.",
        },
      ],
    },
  ],
  temperature: 0,
});

console.log("Description:", result.text);
```

## Configuration Options

```ts
const azure = createAzure({
  endpoint: "https://your-endpoint.com",
  apiKey: "your-key",
  apiVersion: "2024-02-15-preview", // Optional API version
});
```

## Error Handling

```ts
try {
  const result = await generateText({
    model: azure("your-deployment"),
    prompt: "Generate text...",
  });
} catch (error) {
  if (error.response) {
    console.error("API Error:", error.message);
  }
}
```

## Example Usage

```ts
import { createAzure } from "@quail-ai/azure-ai-provider";
import { ModelMessage, generateText, smoothStream, streamText, tool } from "ai";
import { z } from "zod";
import dotenv from "dotenv";
import * as readline from "node:readline/promises";

dotenv.config();

const terminal = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const messages: ModelMessage[] = [];

const azure = createAzure({
  endpoint: process.env.AZURE_API_ENDPOINT,
  apiKey: process.env.AZURE_API_KEY,
});

async function streaming() {
  while (true) {
    const userInput = await terminal.question("You: ");
    messages.push({ role: "user", content: userInput });

    const result = streamText({
      model: azure("DeepSeek-R1"),
      messages,
      transform: smoothStream({ chunking: "word" }),
      temperature: 0,
      maxOutputTokens: 400,
      system:
        "You are an assistant that can answer questions and perform tasks",
    });

    process.stdout.write("Assistant: ");
    let assistantResponse = "";
    for await (const part of result.textStream) {
      process.stdout.write(part);
      assistantResponse += part;
    }
    process.stdout.write("\n");

    messages.push({ role: "assistant", content: assistantResponse });
  }
}

async function blocking() {
  while (true) {
    const userInput = await terminal.question("You: ");
    messages.push({ role: "user", content: userInput });

    const result = await generateText({
      model: azure("DeepSeek-R1"),
      messages,
      tools: {
        get_weather: tool({
          description:
            "Get the current weather in a given location (in Celsius)",
          inputSchema: z.object({
            location: z.string().describe("The city to get the weather for"),
          }),
          execute: async ({ location }) =>
            "The weather in " + location + " is 0 degrees Celsius.",
        }),
      },
      temperature: 0,
      maxOutputTokens: 400,
      system:
        "You are an assistant that can answer questions and perform tasks.",
    });

    console.log("Assistant:", result.text);
  }
}

streaming().catch(console.error);
```

## Related Links

- [AI SDK Documentation](https://sdk.vercel.ai/docs)
- [Azure AI Services](https://azure.microsoft.com/products/ai-services)

## License

MIT
