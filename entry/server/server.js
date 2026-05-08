import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";
import { createRequire } from "module";

dotenv.config();

const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

const PORT = Number(process.env.PORT || 3000);
const DATA_DIR = path.join(__dirname, "data");
const DOC_DIR = path.join(DATA_DIR, "documents");
const VECTOR_DIR = path.join(DATA_DIR, "vectors");

const CHUNK_SIZE = 1200;
const CHUNK_OVERLAP = 200;
const TOP_K = 6;

app.use(cors());
app.use(express.json({ limit: "5mb" }));

const documents = new Map();

async function ensureDirs() {
  await fs.mkdir(DOC_DIR, { recursive: true });
  await fs.mkdir(VECTOR_DIR, { recursive: true });
}

function makeDocumentId(buffer, fileName) {
  const hash = crypto.createHash("sha256");
  hash.update(buffer);
  hash.update(fileName);
  return hash.digest("hex").slice(0, 24);
}

function safeDecodeFileName(value) {
  if (!value) return "document.pdf";

  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function cleanText(text) {
  return String(text || "")
    .replace(/\r/g, "\n")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function chunkText(text, title, pageCount) {
  const cleaned = cleanText(text);
  const chunks = [];

  let start = 0;
  let chunkIndex = 0;

  while (start < cleaned.length) {
    const end = Math.min(start + CHUNK_SIZE, cleaned.length);
    const chunk = cleaned.slice(start, end).trim();

    if (chunk.length > 0) {
      const estimatedPage =
        pageCount > 0
          ? Math.max(1, Math.min(pageCount, Math.floor((start / cleaned.length) * pageCount) + 1))
          : 1;

      chunks.push({
        chunkId: `${chunkIndex}`,
        documentTitle: title,
        page: estimatedPage,
        text: chunk,
        embedding: []
      });

      chunkIndex++;
    }

    if (end >= cleaned.length) break;
    start = end - CHUNK_OVERLAP;
  }

  return chunks;
}

async function qwenPost(endpoint, payload) {
  const apiKey = process.env.DASHSCOPE_API_KEY;
  const baseUrl = process.env.QWEN_BASE_URL || "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";

  if (!apiKey) {
    throw new Error("DASHSCOPE_API_KEY is missing in .env");
  }

  const response = await fetch(`${baseUrl}${endpoint}`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  const text = await response.text();

  if (!response.ok) {
    throw new Error(`Qwen API error ${response.status}: ${text}`);
  }

  return JSON.parse(text);
}

async function embedTexts(texts) {
  const result = await qwenPost("/embeddings", {
    model: process.env.EMBEDDING_MODEL || "text-embedding-v3",
    input: texts
  });

  return result.data.map(item => item.embedding);
}

async function callChat(messages, maxTokens = 1200) {
  const result = await qwenPost("/chat/completions", {
    model: process.env.CHAT_MODEL || "qwen-plus",
    messages,
    temperature: 0.2,
    max_tokens: maxTokens
  });

  return result.choices?.[0]?.message?.content || "";
}

async function embedChunks(chunks) {
  const batchSize = 10;

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    const embeddings = await embedTexts(batch.map(chunk => chunk.text));

    for (let j = 0; j < batch.length; j++) {
      batch[j].embedding = embeddings[j];
    }
  }
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length && i < b.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function saveVectorFile(documentId, data) {
  const filePath = path.join(VECTOR_DIR, `${documentId}.json`);
  await fs.writeFile(filePath, JSON.stringify(data, null, 2), "utf-8");
}

async function loadVectorFile(documentId) {
  const filePath = path.join(VECTOR_DIR, `${documentId}.json`);
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw);
}

async function processDocumentAsync(documentId, pdfPath, title) {
  const state = documents.get(documentId);

  try {
    state.status = "extracting";
    state.progress = 20;

    const buffer = await fs.readFile(pdfPath);
    const parsed = await pdfParse(buffer);

    const text = cleanText(parsed.text);
    const pageCount = parsed.numpages || 0;

    if (text.length < 20) {
      throw new Error("No readable text was extracted. This PDF may be scanned or image-based and may need OCR.");
    }

    state.status = "chunking";
    state.progress = 40;
    state.pageCount = pageCount;

    const chunks = chunkText(text, title, pageCount);

    state.status = "embedding";
    state.progress = 60;

    await embedChunks(chunks);

    state.status = "storing";
    state.progress = 85;

    await saveVectorFile(documentId, {
      documentId,
      title,
      pageCount,
      createdAt: Date.now(),
      chunkCount: chunks.length,
      chunks
    });

    state.status = "ready";
    state.progress = 100;
    state.chunkCount = chunks.length;
  } catch (err) {
    state.status = "error";
    state.error = String(err.message || err);
    state.progress = 100;
  }
}

function getLengthGuide(length) {
  switch (length) {
    case "short":
      return "Keep the summary concise, around 1-2 paragraphs.";
    case "long":
      return "Provide a comprehensive summary, around 5-7 paragraphs.";
    case "detailed":
      return "Provide a detailed study-oriented summary with definitions, important concepts, and relationships.";
    default:
      return "Provide a medium-length summary, around 3-4 paragraphs.";
  }
}

function stripJsonFence(text) {
  let cleaned = String(text || "").trim();

  if (cleaned.startsWith("```json")) {
    cleaned = cleaned.slice(7);
  } else if (cleaned.startsWith("```")) {
    cleaned = cleaned.slice(3);
  }

  if (cleaned.endsWith("```")) {
    cleaned = cleaned.slice(0, -3);
  }

  return cleaned.trim();
}

function parseSummaryAnswer(answer) {
  const cleaned = stripJsonFence(answer);

  try {
    const parsed = JSON.parse(cleaned);

    return {
      summary: parsed.summary || answer,
      keyPoints: Array.isArray(parsed.keyPoints) ? parsed.keyPoints : [],
      studyQuestions: Array.isArray(parsed.studyQuestions) ? parsed.studyQuestions : []
    };
  } catch {
    return {
      summary: answer,
      keyPoints: [],
      studyQuestions: []
    };
  }
}

async function collectChunks(documentIds) {
  const allChunks = [];

  for (const documentId of documentIds) {
    const vectorFile = await loadVectorFile(documentId);

    for (const chunk of vectorFile.chunks) {
      allChunks.push(chunk);
    }
  }

  return allChunks;
}

async function collectVectorFiles(documentIds) {
  const vectorFiles = [];

  for (const documentId of documentIds) {
    const vectorFile = await loadVectorFile(documentId);

    vectorFiles.push({
      documentId: vectorFile.documentId,
      title: vectorFile.title || documentId,
      pageCount: vectorFile.pageCount || 0,
      chunkCount: vectorFile.chunkCount || 0,
      chunks: Array.isArray(vectorFile.chunks) ? vectorFile.chunks : []
    });
  }

  return vectorFiles;
}

function buildBalancedSummaryContext(vectorFiles, totalLimit = 30000) {
  if (vectorFiles.length === 0) {
    return "";
  }

  const perDocumentLimit = Math.max(1200, Math.floor(totalLimit / vectorFiles.length));
  const sections = [];

  for (const vectorFile of vectorFiles) {
    let docText = "";
    const title = vectorFile.title || vectorFile.documentId;

    for (const chunk of vectorFile.chunks) {
      const piece =
        `[${title}, page ${chunk.page || 1}]\n` +
        `${chunk.text || ""}\n\n`;

      if (docText.length + piece.length > perDocumentLimit) {
        const remaining = perDocumentLimit - docText.length;

        if (remaining > 200) {
          docText += piece.slice(0, remaining);
        }

        break;
      }

      docText += piece;
    }

    sections.push(
      `===== Document: ${title} =====\n` +
      docText.trim()
    );
  }

  return sections.join("\n\n");
}

async function retrieveRelevantChunks(documentIds, question) {
  const allChunks = await collectChunks(documentIds);
  const questionEmbedding = (await embedTexts([question]))[0];

  const scored = allChunks.map(chunk => {
    return {
      ...chunk,
      score: cosineSimilarity(questionEmbedding, chunk.embedding)
    };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, TOP_K);
}

app.post(
  "/api/documents/upload",
  express.raw({ type: ["application/pdf", "application/octet-stream"], limit: "100mb" }),
  async (req, res) => {
    try {
      if (!req.body || req.body.length === 0) {
        return res.status(400).json({ error: "Empty PDF upload." });
      }

      const title = safeDecodeFileName(req.header("x-file-name"));
      const documentId = makeDocumentId(req.body, title);
      const pdfPath = path.join(DOC_DIR, `${documentId}.pdf`);

      await fs.writeFile(pdfPath, req.body);

      documents.set(documentId, {
        documentId,
        title,
        pageCount: 0,
        chunkCount: 0,
        status: "processing",
        progress: 5,
        error: ""
      });

      res.json({
        documentId,
        title,
        pageCount: 0,
        status: "processing",
        progress: 5
      });

      setImmediate(() => {
        processDocumentAsync(documentId, pdfPath, title);
      });
    } catch (err) {
      res.status(500).json({ error: String(err.message || err) });
    }
  }
);

app.get("/api/documents", (req, res) => {
  res.json(Array.from(documents.values()));
});

app.delete("/api/documents/:documentId", async (req, res) => {
  const documentId = req.params.documentId;

  try {
    documents.delete(documentId);

    const pdfPath = path.join(DOC_DIR, `${documentId}.pdf`);
    const vectorPath = path.join(VECTOR_DIR, `${documentId}.json`);

    await fs.rm(pdfPath, { force: true });
    await fs.rm(vectorPath, { force: true });

    res.json({
      ok: true,
      documentId
    });
  } catch (err) {
    res.status(500).json({
      ok: false,
      error: String(err.message || err)
    });
  }
});

app.get("/api/documents/:documentId/status", async (req, res) => {
  const state = documents.get(req.params.documentId);

  if (!state) {
    return res.status(404).json({ error: "Document not found." });
  }

  res.json(state);
});

app.post("/api/summarize", async (req, res) => {
  try {
    const { documentIds, length } = req.body;

    if (!Array.isArray(documentIds) || documentIds.length === 0) {
      return res.status(400).json({ error: "documentIds is required." });
    }

    const vectorFiles = await collectVectorFiles(documentIds);

    for (const vectorFile of vectorFiles) {
      if (!Array.isArray(vectorFile.chunks) || vectorFile.chunks.length === 0) {
        return res.status(400).json({
          error: `Document has no processed chunks: ${vectorFile.title}`
        });
      }
    }

    const lengthGuide = getLengthGuide(length);
    const context = buildBalancedSummaryContext(vectorFiles, 30000);

    const documentList = vectorFiles
      .map((doc, index) => `${index + 1}. ${doc.title}`)
      .join("\n");

    const prompt =
      "You are a study assistant. Summarize the selected course documents together.\n\n" +
      `${lengthGuide}\n\n` +
      "Selected documents:\n" +
      `${documentList}\n\n` +
      "Important requirements:\n" +
      "1. Cover every selected document.\n" +
      "2. Do not only summarize the first document.\n" +
      "3. Organize the summary by document when useful.\n" +
      "4. Include key points from each selected document.\n" +
      "5. Include study questions that cover all selected documents.\n\n" +
      "Return ONLY valid JSON in this exact shape:\n" +
      "{\n" +
      '  "summary": "string",\n' +
      '  "keyPoints": ["string"],\n' +
      '  "studyQuestions": ["string"]\n' +
      "}\n\n" +
      `Document text:\n${context}`;

    const answer = await callChat([{ role: "user", content: prompt }], 2500);

    res.json(parseSummaryAnswer(answer));
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    const { documentIds, question } = req.body;

    if (!Array.isArray(documentIds) || documentIds.length === 0) {
      return res.status(400).json({ error: "documentIds is required." });
    }

    if (!question || String(question).trim().length === 0) {
      return res.status(400).json({ error: "question is required." });
    }

    const chunks = await retrieveRelevantChunks(documentIds, question);

    const context = chunks
      .map((chunk, index) => `[Source ${index + 1}] ${chunk.documentTitle}, page ${chunk.page}\n${chunk.text}`)
      .join("\n\n");

    const prompt =
      "You are a document question-answering assistant.\n" +
      "Answer using only the provided document context. If the answer is not in the context, say that the document does not provide enough information.\n\n" +
      `Context:\n${context}\n\n` +
      `Question: ${question}\n\n` +
      "Answer clearly and concisely.";

    const answer = await callChat([{ role: "user", content: prompt }], 1200);

    res.json({
      answer,
      sources: chunks.map(chunk => ({
        documentTitle: chunk.documentTitle,
        page: chunk.page,
        text: chunk.text.slice(0, 300)
      }))
    });
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
});

async function loadExistingDocuments() {
  try {
    const files = await fs.readdir(VECTOR_DIR);

    for (const fileName of files) {
      if (!fileName.endsWith(".json")) {
        continue;
      }

      const filePath = path.join(VECTOR_DIR, fileName);
      const raw = await fs.readFile(filePath, "utf-8");
      const vectorFile = JSON.parse(raw);

      documents.set(vectorFile.documentId, {
        documentId: vectorFile.documentId,
        title: vectorFile.title,
        pageCount: vectorFile.pageCount || 0,
        chunkCount: vectorFile.chunkCount || 0,
        status: "ready",
        progress: 100,
        error: ""
      });
    }

    console.log(`Loaded ${documents.size} existing documents.`);
  } catch (err) {
    console.log("No existing vector documents loaded:", String(err.message || err));
  }
}

ensureDirs().then(async () => {
  await loadExistingDocuments();

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`RAG backend running at http://0.0.0.0:${PORT}`);
  });
});