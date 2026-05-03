import http from '@ohos.net.http';
import { fileIo as fs } from '@kit.CoreFileKit';
import { util } from '@kit.ArkTS';

import {
  PickedDocument,
  SummaryRequest,
  SummaryResponse,
  ChatRequest,
  ChatResponse,
  SourceChunk
} from '../model/Types';

const QWEN_API_KEY: string = 'sk-be2104080a69433a94bc563be1b660ee';
const QWEN_BASE_URL: string = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1';
const QWEN_MODEL: string = 'qwen-turbo';

const MAX_CONTEXT_CHARS: number = 15000;
const CHUNK_SIZE: number = 1200;
const CHUNK_OVERLAP: number = 200;
const TOP_K_CHUNKS: number = 4;

interface QwenMessage {
  role: string;
  content: string;
}

interface QwenChoice {
  message: QwenMessage;
}

interface QwenResponse {
  choices: QwenChoice[];
}

interface SummaryJsonResult {
  summary?: string;
  keyPoints?: string[];
  studyQuestions?: string[];
}

interface TextChunk {
  documentTitle: string;
  page: number;
  text: string;
  score: number;
}

export class RagApi {
  private static async repairExtractedTextWithLlm(text: string): Promise<string> {
    const context: string = RagApi.truncateText(text, 12000);

    if (context.trim().length < 80) {
      return text;
    }

    const prompt: string =
      'The following text was extracted locally from a PDF. It may contain broken spacing, repeated fragments, bad line breaks, or partial PDF extraction artifacts.\n\n' +
        'Clean and reconstruct the readable document text as much as possible.\n' +
        'Do not summarize it.\n' +
        'Do not add new information.\n' +
        'Return only the cleaned document text.\n\n' +
        'Extracted text:\n' +
        context;

    try {
      const repaired: string = await RagApi.callQwen(prompt);
      const cleaned: string = RagApi.cleanText(repaired);

      if (cleaned.length > text.trim().length * 0.5) {
        return cleaned;
      }

      return text;
    } catch (err) {
      return text;
    }
  }

  static async extractPdfText(file: PickedDocument): Promise<string> {
    const fileBytes: Uint8Array = RagApi.readFileAsBytes(file.uri);
    let text: string = RagApi.parsePdfText(fileBytes);

    if (text.trim().length === 0) {
      return 'No readable text was extracted from this file. If this is a scanned PDF or image-based slide deck, OCR is needed.';
    }

    if (RagApi.isBadPdfText(text)) {
      text = await RagApi.repairExtractedTextWithLlm(text);
    }

    if (text.trim().length === 0) {
      return 'No readable text was extracted from this file.';
    }

    return text;
  }

  static async summarize(req: SummaryRequest, documentTexts: string[]): Promise<SummaryResponse> {
    const combinedText: string = RagApi.cleanText(documentTexts.join('\n\n'));
    const lengthGuide: string = RagApi.getLengthGuide(req.length);
    const context: string = RagApi.truncateText(combinedText, MAX_CONTEXT_CHARS);

    const prompt: string =
      'You are a study assistant. Summarize the provided course material.\n\n' +
        `${lengthGuide}\n\n` +
        'Return ONLY valid JSON in this exact shape:\n' +
        '{\n' +
        '  "summary": "string",\n' +
        '  "keyPoints": ["string"],\n' +
        '  "studyQuestions": ["string"]\n' +
        '}\n\n' +
        `Document text:\n${context}`;

    const answer: string = await RagApi.callQwen(prompt);
    return RagApi.parseSummaryAnswer(answer);
  }

  static async chat(req: ChatRequest, documentTexts: string[]): Promise<ChatResponse> {
    const chunks: TextChunk[] = RagApi.retrieveRelevantChunks(req.question, documentTexts);
    const context: string = chunks.map((chunk: TextChunk, index: number): string => {
      return `[Source ${index + 1}] ${chunk.text}`;
    }).join('\n\n');

    const prompt: string =
      'You are a document question-answering assistant.\n' +
        'Answer using only the provided document context. If the answer is not in the context, say that the document does not provide enough information.\n\n' +
        `Context:\n${context}\n\n` +
        `Question: ${req.question}\n\n` +
        'Answer clearly and concisely.';

    const answer: string = await RagApi.callQwen(prompt);

    const sources: SourceChunk[] = chunks.map((chunk: TextChunk): SourceChunk => {
      return {
        documentTitle: chunk.documentTitle,
        page: chunk.page,
        text: RagApi.truncateText(chunk.text, 300)
      };
    });

    return {
      answer: answer,
      sources: sources
    };
  }

  private static async callQwen(prompt: string): Promise<string> {
    if (QWEN_API_KEY === 'PUT_YOUR_QWEN_API_KEY_HERE' || QWEN_API_KEY.trim().length === 0) {
      throw new Error('Missing Qwen API key. Set QWEN_API_KEY before calling the model.');
    }

    const messages: QwenMessage[] = new Array<QwenMessage>();

    messages.push({
      role: 'user',
      content: prompt
    });

    const requestBody: Record<string, Object> = {
      'model': QWEN_MODEL,
      'messages': messages,
      'temperature': 0.2
    };

    const request = http.createHttp();

    try {
      const response = await request.request(`${QWEN_BASE_URL}/chat/completions`, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${QWEN_API_KEY}`
        },
        extraData: JSON.stringify(requestBody),
        expectDataType: http.HttpDataType.STRING,
        connectTimeout: 60000,
        readTimeout: 120000
      });

      const body: string = response.result as string;

      if (response.responseCode < 200 || response.responseCode >= 300) {
        throw new Error(`Qwen API error ${response.responseCode}: ${body}`);
      }

      const result: QwenResponse = JSON.parse(body) as QwenResponse;

      if (!result.choices || result.choices.length === 0 || !result.choices[0].message) {
        throw new Error('Invalid Qwen response: missing choices.');
      }

      return result.choices[0].message.content;
    } finally {
      request.destroy();
    }
  }

  private static parseSummaryAnswer(answer: string): SummaryResponse {
    const cleaned: string = RagApi.stripJsonFence(answer);

    try {
      const parsed: SummaryJsonResult = JSON.parse(cleaned) as SummaryJsonResult;

      return {
        summary: parsed.summary ? parsed.summary : answer,
        keyPoints: parsed.keyPoints ? parsed.keyPoints : [],
        studyQuestions: parsed.studyQuestions ? parsed.studyQuestions : []
      };
    } catch (err) {
      return {
        summary: answer,
        keyPoints: [],
        studyQuestions: []
      };
    }
  }

  private static stripJsonFence(text: string): string {
    let cleaned: string = text.trim();

    if (cleaned.startsWith('```json')) {
      cleaned = cleaned.substring(7);
    } else if (cleaned.startsWith('```')) {
      cleaned = cleaned.substring(3);
    }

    if (cleaned.endsWith('```')) {
      cleaned = cleaned.substring(0, cleaned.length - 3);
    }

    return cleaned.trim();
  }

  private static getLengthGuide(length: string): string {
    switch (length) {
      case 'short':
        return 'Keep the summary concise, around 1-2 paragraphs.';
      case 'long':
        return 'Provide a comprehensive summary, around 5-7 paragraphs.';
      case 'detailed':
        return 'Provide a detailed study-oriented summary with definitions, important concepts, and relationships.';
      default:
        return 'Provide a medium-length summary, around 3-4 paragraphs.';
    }
  }

  private static parsePdfText(bytes: Uint8Array): string {
    const rawText: string = RagApi.bytesToPdfString(bytes);

    const literalText: string = RagApi.extractPdfLiteralStrings(rawText);
    const hexText: string = RagApi.extractPdfHexStrings(rawText);

    const combined: string = literalText + '\n' + hexText;
    return RagApi.postProcessPdfText(combined);
  }

  private static bytesToPdfString(bytes: Uint8Array): string {
    const parts: string[] = new Array<string>();
    let chunk: string = '';

    for (let i = 0; i < bytes.length; i++) {
      chunk += String.fromCharCode(bytes[i]);

      if (chunk.length >= 4096) {
        parts.push(chunk);
        chunk = '';
      }
    }

    if (chunk.length > 0) {
      parts.push(chunk);
    }

    return parts.join('');
  }

  private static extractPdfLiteralStrings(rawText: string): string {
    const results: string[] = new Array<string>();
    const regex: RegExp = /\((?:\\.|[^\\()])*\)/g;

    let match: RegExpExecArray | null = regex.exec(rawText);

    while (match !== null) {
      const rawItem: string = match[0];

      if (rawItem.length > 2) {
        const inner: string = rawItem.substring(1, rawItem.length - 1);
        const decoded: string = RagApi.decodePdfLiteralString(inner);

        if (RagApi.looksLikeReadablePdfPiece(decoded)) {
          results.push(decoded);
        }
      }

      match = regex.exec(rawText);
    }

    return results.join(' ');
  }

  private static extractPdfHexStrings(rawText: string): string {
    const results: string[] = new Array<string>();
    const regex: RegExp = /<([0-9A-Fa-f\s]{4,})>/g;

    let match: RegExpExecArray | null = regex.exec(rawText);

    while (match !== null) {
      const hexBody: string = match[1];
      const decoded: string = RagApi.decodePdfHexString(hexBody);

      if (RagApi.looksLikeReadablePdfPiece(decoded)) {
        results.push(decoded);
      }

      match = regex.exec(rawText);
    }

    return results.join(' ');
  }

  private static decodePdfLiteralString(text: string): string {
    let result: string = '';

    for (let i = 0; i < text.length; i++) {
      const ch: string = text.charAt(i);

      if (ch !== '\\') {
        result += ch;
        continue;
      }

      if (i + 1 >= text.length) {
        break;
      }

      const next: string = text.charAt(i + 1);

      if (next === 'n') {
        result += '\n';
        i++;
      } else if (next === 'r') {
        result += '\n';
        i++;
      } else if (next === 't') {
        result += ' ';
        i++;
      } else if (next === 'b') {
        result += ' ';
        i++;
      } else if (next === 'f') {
        result += ' ';
        i++;
      } else if (next === '(') {
        result += '(';
        i++;
      } else if (next === ')') {
        result += ')';
        i++;
      } else if (next === '\\') {
        result += '\\';
        i++;
      } else if (RagApi.isOctalChar(next)) {
        let octal: string = next;
        let used: number = 1;

        while (used < 3 && i + 1 + used < text.length) {
          const octalChar: string = text.charAt(i + 1 + used);

          if (RagApi.isOctalChar(octalChar)) {
            octal += octalChar;
            used++;
          } else {
            break;
          }
        }

        const code: number = parseInt(octal, 8);

        if (code >= 32 && code <= 126) {
          result += String.fromCharCode(code);
        } else {
          result += ' ';
        }

        i += used;
      } else {
        result += next;
        i++;
      }
    }

    return result;
  }

  private static decodePdfHexString(hexText: string): string {
    let hex: string = hexText.replace(/\s+/g, '');

    if (hex.length < 4) {
      return '';
    }

    if (hex.length % 2 !== 0) {
      hex += '0';
    }

    let result: string = '';

    if (hex.startsWith('FEFF') || hex.startsWith('feff')) {
      for (let i = 4; i + 3 < hex.length; i += 4) {
        const code: number = parseInt(hex.substring(i, i + 4), 16);

        if (code >= 32 && code <= 65535) {
          result += String.fromCharCode(code);
        } else {
          result += ' ';
        }
      }
    } else {
      for (let i = 0; i + 1 < hex.length; i += 2) {
        const code: number = parseInt(hex.substring(i, i + 2), 16);

        if (code >= 32 && code <= 126) {
          result += String.fromCharCode(code);
        } else {
          result += ' ';
        }
      }
    }

    return result;
  }

  private static isOctalChar(ch: string): boolean {
    return ch >= '0' && ch <= '7';
  }

  private static looksLikeReadablePdfPiece(text: string): boolean {
    const trimmed: string = text.trim();

    if (trimmed.length < 2) {
      return false;
    }

    if (trimmed.startsWith('/') || trimmed.startsWith('%')) {
      return false;
    }

    let readableCount: number = 0;

    for (let i = 0; i < trimmed.length; i++) {
      const code: number = trimmed.charCodeAt(i);

      const isReadable: boolean =
        code === 9 ||
          code === 10 ||
          code === 13 ||
          code === 32 ||
          code === 45 ||
          code === 46 ||
          code === 44 ||
          code === 58 ||
          code === 59 ||
          code === 39 ||
          code === 34 ||
          code === 40 ||
          code === 41 ||
          code === 91 ||
          code === 93 ||
          (code >= 48 && code <= 57) ||
          (code >= 65 && code <= 90) ||
          (code >= 97 && code <= 122);

      if (isReadable) {
        readableCount++;
      }
    }

    return readableCount >= 2 && readableCount >= trimmed.length * 0.6;
  }

  private static postProcessPdfText(text: string): string {
    return text
      .replace(/\r/g, '\n')
      .replace(/-\s+/g, '')
      .replace(/[ \t]+/g, ' ')
      .replace(/\n[ \t]+/g, '\n')
      .replace(/\s{2,}/g, ' ')
      .trim();
  }

  private static isBadPdfText(text: string): boolean {
    const cleaned: string = text.trim();

    if (cleaned.length < 300) {
      return true;
    }

    const lower: string = cleaned.toLowerCase();

    let pdfTokenCount: number = 0;

    if (lower.includes('/font')) {
      pdfTokenCount++;
    }

    if (lower.includes('endobj')) {
      pdfTokenCount++;
    }

    if (lower.includes('xref')) {
      pdfTokenCount++;
    }

    if (lower.includes('flatedecode')) {
      pdfTokenCount++;
    }

    if (lower.includes('stream')) {
      pdfTokenCount++;
    }

    if (pdfTokenCount >= 2) {
      return true;
    }

    let letterOrNumberCount: number = 0;

    for (let i = 0; i < cleaned.length; i++) {
      const code: number = cleaned.charCodeAt(i);

      const isLetterOrNumber: boolean =
        (code >= 48 && code <= 57) ||
          (code >= 65 && code <= 90) ||
          (code >= 97 && code <= 122);

      if (isLetterOrNumber) {
        letterOrNumberCount++;
      }
    }

    return letterOrNumberCount < cleaned.length * 0.35;
  }

  private static cleanText(text: string): string {
    return text
      .replace(/\r/g, '\n')
      .replace(/[ \t]+/g, ' ')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  private static truncateText(text: string, maxLength: number): string {
    if (text.length <= maxLength) {
      return text;
    }

    return text.substring(0, maxLength);
  }

  private static retrieveRelevantChunks(question: string, documentTexts: string[]): TextChunk[] {
    const allChunks: TextChunk[] = [];

    for (let i = 0; i < documentTexts.length; i++) {
      const title: string = `Document ${i + 1}`;
      const chunks: TextChunk[] = RagApi.chunkText(documentTexts[i], title);
      allChunks.push(...chunks);
    }

    const queryTerms: string[] = RagApi.tokenize(question);

    for (let i = 0; i < allChunks.length; i++) {
      allChunks[i].score = RagApi.scoreChunk(queryTerms, allChunks[i].text);
    }

    allChunks.sort((a: TextChunk, b: TextChunk): number => {
      return b.score - a.score;
    });

    const selected: TextChunk[] = [];

    for (let i = 0; i < allChunks.length && selected.length < TOP_K_CHUNKS; i++) {
      if (allChunks[i].score > 0 || selected.length === 0) {
        selected.push(allChunks[i]);
      }
    }

    return selected;
  }

  private static chunkText(text: string, documentTitle: string): TextChunk[] {
    const cleaned: string = RagApi.cleanText(text);
    const chunks: TextChunk[] = [];

    if (cleaned.length === 0) {
      return chunks;
    }

    let start: number = 0;
    let page: number = 1;

    while (start < cleaned.length) {
      const end: number = Math.min(start + CHUNK_SIZE, cleaned.length);
      const chunkText: string = cleaned.substring(start, end).trim();

      if (chunkText.length > 0) {
        chunks.push({
          documentTitle: documentTitle,
          page: page,
          text: chunkText,
          score: 0
        });
      }

      if (end >= cleaned.length) {
        break;
      }

      start = end - CHUNK_OVERLAP;
      page++;
    }

    return chunks;
  }

  private static tokenize(text: string): string[] {
    const lower: string = text.toLowerCase();
    const normalized: string = lower.replace(/[^a-z0-9]+/g, ' ');
    const parts: string[] = normalized.split(' ');
    const tokens: string[] = [];

    for (let i = 0; i < parts.length; i++) {
      const token: string = parts[i].trim();

      if (token.length >= 3) {
        tokens.push(token);
      }
    }

    return tokens;
  }

  private static scoreChunk(queryTerms: string[], text: string): number {
    const lowerText: string = text.toLowerCase();
    let score: number = 0;

    for (let i = 0; i < queryTerms.length; i++) {
      if (lowerText.includes(queryTerms[i])) {
        score++;
      }
    }

    return score;
  }

  private static readFileAsBytes(uri: string): Uint8Array {
    const file = fs.openSync(uri, fs.OpenMode.READ_ONLY);

    try {
      const stat = fs.statSync(file.fd);
      const size: number = stat.size;

      if (size <= 0) {
        return new Uint8Array();
      }

      const buffer: ArrayBuffer = new ArrayBuffer(size);
      const readLength: number = fs.readSync(file.fd, buffer, {
        offset: 0,
        length: size
      });

      return new Uint8Array(buffer, 0, readLength);
    } finally {
      fs.closeSync(file);
    }
  }
}