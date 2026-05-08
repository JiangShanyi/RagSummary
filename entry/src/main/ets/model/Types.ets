export interface UploadedDocument {
  documentId: string;
  title: string;
  pageCount: number;
  status: string;
  progress: number;
}

export interface DocumentStatusResponse {
  documentId: string;
  title: string;
  pageCount: number;
  chunkCount: number;
  status: string;
  progress: number;
  error: string;
}

export interface PickedDocument {
  uri: string;
  name: string;
  mimeType: string;
}

export interface SummaryRequest {
  documentIds: string[];
  length: string;
}

export interface SummaryResponse {
  summary: string;
  keyPoints: string[];
  studyQuestions: string[];
}

export interface ChatRequest {
  documentIds: string[];
  question: string;
}

export interface SourceChunk {
  documentTitle: string;
  page: number;
  text: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceChunk[];
}