import http from '@ohos.net.http';
import { fileIo as fs } from '@kit.CoreFileKit';

import {
  PickedDocument,
  UploadedDocument,
  DocumentStatusResponse,
  SummaryRequest,
  SummaryResponse,
  ChatRequest,
  ChatResponse
} from '../model/Types';

const SERVER_BASE_URL: string = 'http://192.168.12.5:3000';

interface ErrorResponse {
  error?: string;
}

export class RagApi {
  static async uploadDocument(file: PickedDocument): Promise<UploadedDocument> {
    const bytes: Uint8Array = RagApi.readFileAsBytes(file.uri);

    if (bytes.length <= 0) {
      throw new Error('Selected PDF is empty.');
    }

    const body: ArrayBuffer = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength
    );

    const responseText: string = await RagApi.requestRaw(
      `${SERVER_BASE_URL}/api/documents/upload`,
      body,
      file.name
    );

    return JSON.parse(responseText) as UploadedDocument;
  }

  static async getDocumentStatus(documentId: string): Promise<DocumentStatusResponse> {
    const responseText: string = await RagApi.requestGet(
      `${SERVER_BASE_URL}/api/documents/${documentId}/status`
    );

    return JSON.parse(responseText) as DocumentStatusResponse;
  }

  static async listDocuments(): Promise<DocumentStatusResponse[]> {
    const responseText: string = await RagApi.requestGet(
      `${SERVER_BASE_URL}/api/documents`
    );

    return JSON.parse(responseText) as DocumentStatusResponse[];
  }

  private static async requestDelete(url: string): Promise<string> {
    const request = http.createHttp();

    try {
      const response = await request.request(url, {
        method: http.RequestMethod.DELETE,
        expectDataType: http.HttpDataType.STRING,
        connectTimeout: 30000,
        readTimeout: 60000
      });

      return RagApi.handleResponse(response.responseCode, response.result as string);
    } finally {
      request.destroy();
    }
  }

  static async deleteDocument(documentId: string): Promise<void> {
    await RagApi.requestDelete(
      `${SERVER_BASE_URL}/api/documents/${documentId}`
    );
  }

  static async waitForDocumentReady(
    documentId: string,
    onStatus: (status: DocumentStatusResponse) => void
  ): Promise<DocumentStatusResponse> {
    while (true) {
      const status: DocumentStatusResponse = await RagApi.getDocumentStatus(documentId);
      onStatus(status);

      if (status.status === 'ready') {
        return status;
      }

      if (status.status === 'error') {
        throw new Error(status.error);
      }

      await RagApi.sleep(1000);
    }
  }

  static async summarize(req: SummaryRequest): Promise<SummaryResponse> {
    const responseText: string = await RagApi.requestJson(
      `${SERVER_BASE_URL}/api/summarize`,
      req
    );

    return JSON.parse(responseText) as SummaryResponse;
  }

  static async chat(req: ChatRequest): Promise<ChatResponse> {
    const responseText: string = await RagApi.requestJson(
      `${SERVER_BASE_URL}/api/chat`,
      req
    );

    return JSON.parse(responseText) as ChatResponse;
  }

  private static async requestRaw(url: string, body: ArrayBuffer, fileName: string): Promise<string> {
    const request = http.createHttp();

    try {
      const response = await request.request(url, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/pdf',
          'X-File-Name': encodeURIComponent(fileName)
        },
        extraData: body,
        expectDataType: http.HttpDataType.STRING,
        connectTimeout: 60000,
        readTimeout: 120000
      });

      return RagApi.handleResponse(response.responseCode, response.result as string);
    } finally {
      request.destroy();
    }
  }

  private static async requestJson(url: string, body: Object): Promise<string> {
    const request = http.createHttp();

    try {
      const response = await request.request(url, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json'
        },
        extraData: JSON.stringify(body),
        expectDataType: http.HttpDataType.STRING,
        connectTimeout: 60000,
        readTimeout: 120000
      });

      return RagApi.handleResponse(response.responseCode, response.result as string);
    } finally {
      request.destroy();
    }
  }

  private static async requestGet(url: string): Promise<string> {
    const request = http.createHttp();

    try {
      const response = await request.request(url, {
        method: http.RequestMethod.GET,
        expectDataType: http.HttpDataType.STRING,
        connectTimeout: 30000,
        readTimeout: 60000
      });

      return RagApi.handleResponse(response.responseCode, response.result as string);
    } finally {
      request.destroy();
    }
  }

  private static handleResponse(code: number, body: string): string {
    if (code >= 200 && code < 300) {
      return body;
    }

    try {
      const parsed: ErrorResponse = JSON.parse(body) as ErrorResponse;

      if (parsed.error) {
        throw new Error(`Server error ${code}: ${parsed.error}`);
      }
    } catch (err) {
    }

    throw new Error(`Server error ${code}: ${body}`);
  }

  private static sleep(ms: number): Promise<void> {
    return new Promise<void>((resolve) => {
      setTimeout(() => {
        resolve();
      }, ms);
    });
  }

  private static readFileAsBytes(uri: string): Uint8Array {
    try {
      return RagApi.readBytesFromPath(uri);
    } catch (err) {
      const normalizedPath: string = RagApi.normalizeLocalFilePath(uri);

      if (normalizedPath !== uri) {
        return RagApi.readBytesFromPath(normalizedPath);
      }

      throw new Error(`Cannot read PDF file. uri=${uri}, error=${err}`);
    }
  }

  private static normalizeLocalFilePath(uri: string): string {
    if (uri.startsWith('file://')) {
      return uri.substring(7);
    }

    return uri;
  }

  private static readBytesFromPath(path: string): Uint8Array {
    const file = fs.openSync(path, fs.OpenMode.READ_ONLY);

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