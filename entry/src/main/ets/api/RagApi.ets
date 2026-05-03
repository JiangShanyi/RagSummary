import http from '@ohos.net.http';
import { fileIo as fs } from '@kit.CoreFileKit';
import util from '@ohos.util';

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

export class RagApi {
  static async extractPdfText(file: PickedDocument): Promise<string> {
    const fileBytes: Uint8Array = RagApi.readFileAsBytes(file.uri);
    const text: string = RagApi.parsePdfText(fileBytes);
    return text;
  }

  static async summarize(req: SummaryRequest, documentTexts: string[]): Promise<SummaryResponse> {
    const combinedText: string = documentTexts.join('\n\n');
    const lengthGuide: string = RagApi.getLengthGuide(req.length);

    const prompt: string = `Please summarize the following document(s). ${lengthGuide}\n\nProvide a JSON response with:\n- summary: the main summary\n- keyPoints: array of key points\n- studyQuestions: array of study questions\n\nDocument(s):\n${combinedText.substring(0, 15000)}`;

    const answer: string = await RagApi.callQwen(prompt);

    try {
      const parsed = JSON.parse(answer);
      return {
        summary: parsed.summary || answer,
        keyPoints: parsed.keyPoints || [],
        studyQuestions: parsed.studyQuestions || []
      };
    } catch {
      return {
        summary: answer,
        keyPoints: [],
        studyQuestions: []
      };
    }
  }

  static async chat(req: ChatRequest, documentTexts: string[]): Promise<ChatResponse> {
    const combinedText: string = documentTexts.join('\n\n');

    const prompt: string = `Based on the following document(s), answer the question.\n\nDocument(s):\n${combinedText.substring(0, 15000)}\n\nQuestion: ${req.question}\n\nAnswer:`;

    const answer: string = await RagApi.callQwen(prompt);

    const sources: SourceChunk[] = [{
      documentTitle: 'Current Document',
      page: 1,
      text: combinedText.substring(0, 200)
    }];

    return {
      answer: answer,
      sources: sources
    };
  }

  private static async callQwen(prompt: string): Promise<string> {
    const request = http.createHttp();

    try {
      const response = await request.request(`${QWEN_BASE_URL}/chat/completions`, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${QWEN_API_KEY}`
        },
        extraData: JSON.stringify({
          model: 'qwen-turbo',
          messages: [
            { role: 'user', content: prompt }
          ]
        }),
        expectDataType: http.HttpDataType.STRING,
        connectTimeout: 60000,
        readTimeout: 120000
      });

      if (response.responseCode < 200 || response.responseCode >= 300) {
        throw new Error(`API error: ${response.responseCode}`);
      }

      const result = JSON.parse(response.result as string);
      return result.choices[0].message.content;
    } finally {
      request.destroy();
    }
  }

  private static getLengthGuide(length: string): string {
    switch (length) {
      case 'short': return 'Keep the summary concise (1-2 paragraphs).';
      case 'long': return 'Provide a comprehensive summary (5-7 paragraphs).';
      case 'detailed': return 'Provide a very detailed summary with all important information.';
      default: return 'Provide a medium-length summary (3-4 paragraphs).';
    }
  }

  private static parsePdfText(bytes: Uint8Array): string {
    const decoder = new util.TextDecoder('utf-8');
    let text: string = decoder.decodeWithStream(bytes);

    text = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, ' ');
    text = text.replace(/\s+/g, ' ');

    return text;
  }

  private static readFileAsBytes(uri: string): Uint8Array {
    const CHUNK_SIZE: number = 64 * 1024;
    const chunks: Uint8Array[] = [];

    const file = fs.openSync(uri, fs.OpenMode.READ_ONLY);
    let offset: number = 0;

    try {
      while (true) {
        const chunkBuffer = new ArrayBuffer(CHUNK_SIZE);
        const readLen: number = fs.readSync(file.fd, chunkBuffer, {
          currentOffset: offset,
          maxLength: CHUNK_SIZE
        });

        if (readLen <= 0) {
          break;
        }

        chunks.push(new Uint8Array(chunkBuffer, 0, readLen));
        offset += readLen;
      }
    } finally {
      fs.closeSync(file);
    }

    return RagApi.mergeBytes(chunks);
  }

  private static mergeBytes(parts: Uint8Array[]): Uint8Array {
    let totalLength: number = 0;
    for (let i = 0; i < parts.length; i++) {
      totalLength += parts[i].byteLength;
    }

    const result = new Uint8Array(totalLength);
    let offset: number = 0;
    for (let i = 0; i < parts.length; i++) {
      result.set(parts[i], offset);
      offset += parts[i].byteLength;
    }

    return result;
  }
}
