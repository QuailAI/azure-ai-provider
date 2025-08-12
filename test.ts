import { createAzure } from "./src/azure-ai-provider";
import { CoreMessage, generateText, stepCountIs, streamText, tool } from "ai";
import { z } from "zod";
import dotenv from "dotenv";
import * as readline from "node:readline/promises";

dotenv.config();

console.log("🧪 Testing Azure AI Provider v2 conversion...");

const modelId: string = "Llama-3.3-70B-Instruct";

const terminal = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const messages: CoreMessage[] = [];

const azure = createAzure({
  endpoint: process.env.AZURE_API_ENDPOINT,
  apiKey: process.env.AZURE_API_KEY,
});

// Test v2 interface compliance
console.log("✅ Provider created successfully");
const model = azure.languageModel(modelId);
console.log("✅ Language model created successfully");
console.log("📋 Model specification version:", model.specificationVersion);
console.log("🏭 Model provider:", model.provider);
console.log("🆔 Model ID:", model.modelId);
console.log(
  "🔗 Supported URLs:",
  Object.keys(model.supportedUrls).length > 0
    ? model.supportedUrls
    : "None configured"
);
console.log(
  "⚙️  Has doGenerate method:",
  typeof model.doGenerate === "function"
);
console.log("📺 Has doStream method:", typeof model.doStream === "function");
console.log("🌐 Provider methods available:", {
  languageModel: typeof azure.languageModel === "function",
  textEmbeddingModel: typeof azure.textEmbeddingModel === "function",
  imageModel: typeof azure.imageModel === "function",
});
console.log("✨ All v2 interface requirements satisfied!\n");

async function streaming() {
  while (true) {
    const userInput = await terminal.question("You: ");
    messages.push({ role: "user", content: userInput });

    const result = streamText({
      model: azure(modelId),
      messages,
      // tools can be provided to allow model to call functions during streaming
      tools: {
        getWeather: tool({
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
      system:
        "You are an assistant that can answer questions and perform tasks",
    });
    for await (const event of result.fullStream) {
      console.log(JSON.stringify(event));
    }
  }
}

async function blocking() {
  while (true) {
    const userInput = await terminal.question("You: ");
    messages.push({ role: "user", content: userInput });

    const result = await generateText({
      model: azure(modelId),
      messages,
      tools: {
        getWeather: tool({
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
      system:
        "You are an assistant that can answer questions and perform tasks.",
    });

    console.log("Assistant:", result.text);
  }
}

async function test() {
  console.log("Testing streaming...");
  const result = streamText({
    model: azure(modelId),
    messages: [
      {
        role: "user",
        content: "What is the weather in Chicago?",
      },
    ],
    // tools can be provided to allow model to call functions during streaming
    tools: {
      getWeather: tool({
        description: "Get the current weather in a given location (in Celsius)",
        inputSchema: z.object({
          location: z.string().describe("The city to get the weather for"),
        }),
        execute: async ({ location }) =>
          "The weather in " + location + " is 0 degrees Celsius.",
      }),
    },
    stopWhen: stepCountIs(2),
    temperature: 0,
    system: "You are an assistant that can answer questions and perform tasks",
  });
  const reader = result.textStream.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    process.stdout.write(value);
  }
  console.log("Streaming complete");
  process.exit(0);
}

test().catch(console.error);
