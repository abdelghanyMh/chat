require("dotenv").config();
const express = require("express");
const puppeteer = require("puppeteer");
const { HfInference } = require("@huggingface/inference");
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors());

const hf = new HfInference(process.env.HUGGINGFACE_TOKEN);

// Generic web scraping function
async function scrapeWebData(searchQuery) {
  try {
    const browser = await puppeteer.launch({ headless: "new" });
    const page = await browser.newPage();
    await page.goto(
      `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`
    );

    // Extract relevant data from the page
    const results = await page.evaluate(() => {
      const items = document.querySelectorAll(".g");
      return Array.from(items, (item) => ({
        title: item.querySelector("h3")?.textContent || "",
        description: item.querySelector(".VwiC3b")?.textContent || "",
        link: item.querySelector("a")?.href || "",
      })).slice(0, 5); // Limit to top 5 results
    });

    await browser.close();
    return results;
  } catch (error) {
    console.error("Scraping error:", error);
    return [];
  }
}

app.post("/api/chat", async (req, res) => {
  try {
    const { messages } = req.body;

    // Validate incoming messages format
    if (
      !Array.isArray(messages) ||
      !messages.every(
        (msg) =>
          msg.role &&
          ["system", "user", "assistant"].includes(msg.role) &&
          typeof msg.content === "string"
      )
    ) {
      return res.status(400).json({
        error:
          "Invalid messages format. Expected array of {role, content} objects",
      });
    }

    // Get the latest user message
    const lastUserMessage =
      messages.findLast((msg) => msg.role === "user")?.content || "";

    // Scrape relevant web data if keywords are detected
    let scrapedData = [];
    const keywordsToScrape = [
      "news",
      "weather",
      "review",
      "price",
      "comparison",
    ];

    if (
      keywordsToScrape.some((keyword) =>
        lastUserMessage.toLowerCase().includes(keyword)
      )
    ) {
      scrapedData = await scrapeWebData(lastUserMessage);
    }

    // Set up SSE
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    try {
      // Create chat completion stream with the provided messages
      const stream = await hf.chatCompletionStream({
        model: "mistralai/Mistral-Nemo-Instruct-2407",
        messages: messages,
        temperature: 0.7,
        max_tokens: 500,
        stream: true,
      });

      // Stream the response
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || "";
        if (content) {
          res.write(
            `data: ${JSON.stringify({
              content,
              scrapedData: scrapedData.length > 0 ? scrapedData : undefined,
            })}\n\n`
          );
        }
      }

      res.write("data: [DONE]\n\n");
      res.end();
    } catch (error) {
      console.error("Chat completion error:", error);
      res.write(
        `data: ${JSON.stringify({ error: "Chat completion failed" })}\n\n`
      );
      res.end();
    }
  } catch (error) {
    console.error("API error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
