import { TokenTextSplitter } from "langchain/text_splitter";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { loadSummarizationChain } from "langchain/chains";
import fetch from "node-fetch";
import deepl, { DeeplLanguages } from "deepl";
import { htmlToMarkdown } from "webforai";
import dotenv from "dotenv";
import { loadHtml } from "webforai/loaders/playwright";

dotenv.config();

const splitter = new TokenTextSplitter({
  encodingName: "cl100k_base",
  chunkSize: 800,
  chunkOverlap: 100,
});

async function get_story(id: number) {
  const response = await fetch(
    `https://hacker-news.firebaseio.com/v0/item/${id}.json`
  );
  const data = await response.json();
  return {
    ...data,
    hnUrl: `https://news.ycombinator.com/item?id=${id}`,
  };
}

async function get_top_stories(limit: number = 10) {
  const response = await fetch(
    "https://hacker-news.firebaseio.com/v0/topstories.json"
  );
  const ids = await response.json();
  const stories = await Promise.all(
    ids.slice(0, limit).map((id: number) => get_story(id))
  );
  return stories;
}

export const translate = async (text: string, target: DeeplLanguages) => {
  const res = await deepl({
    text,
    target_lang: target,
    auth_key: process.env.DEEPL_API_KEY as string,
    free_api: true,
  })
    .then((result) => result.data.translations[0].text)
    .catch((error) => {
      console.error({
        message: error.message,
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
      });
      return error;
    });

  return res;
};

async function main() {
  const top_stories = await get_top_stories(10);
  console.log(top_stories);

  for (const item of top_stories) {
    const html = await loadHtml(item.url);

    const markdown = htmlToMarkdown(html, { solveLinks: item.url });

    console.log(markdown);

    if (markdown.length > 15000) {
      console.log("skip ai...");
      await fetch(process.env.DISCORD_WEBHOOK_URL as string, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: `${item.url}`,
        }),
      });
      continue;
    }

    const docs = await splitter.createDocuments([markdown]);

    const model = new ChatOpenAI({
      modelName: "gpt-3.5-turbo-0613",
      temperature: 0,
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const chain = loadSummarizationChain(model, { type: "map_reduce" });
    const result = await chain.call({ input_documents: docs });

    const translated = await translate(result.text, "JA");

    await fetch(process.env.DISCORD_WEBHOOK_URL as string, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: `${item.url}\n\n${translated}`,
      }),
    });
  }
}

main();
