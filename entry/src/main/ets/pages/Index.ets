import { common } from '@kit.AbilityKit';

import { RagApi } from '../api/RagApi';
import { FilePickerUtil } from '../utils/FilePickerUtil';
import {
  PickedDocument,
  UploadedDocument,
  DocumentStatusResponse,
  SummaryRequest,
  SummaryResponse,
  ChatRequest,
  ChatResponse,
  SourceChunk
} from '../model/Types';

interface DocumentItem {
  documentId: string;
  name: string;
  status: string;
  progress: number;
  pageCount: number;
  chunkCount: number;
  error: string;
}

@Entry
@Component
struct Index {
  @State documents: DocumentItem[] = new Array<DocumentItem>();
  @State uiRefreshKey: number = 0;
  @State selectedLength: string = 'medium';
  @State selectedSummaryDocumentIds: string[] = new Array<string>();

  @State summaryText: string = '';
  @State keyPoints: string[] = new Array<string>();
  @State studyQuestions: string[] = new Array<string>();

  @State question: string = '';
  @State answer: string = '';
  @State sourcesText: string = '';

  @State statusText: string = '';
  @State isLoading: boolean = false;
  private getDocumentIds(): string[] {
    const ids: string[] = new Array<string>();

    for (let i = 0; i < this.documents.length; i++) {
      ids.push(this.documents[i].documentId);
    }

    return ids;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise<void>((resolve) => {
      setTimeout(() => {
        resolve();
      }, ms);
    });
  }
  private updateDocumentStatus(status: DocumentStatusResponse): void {
    const nextDocuments: DocumentItem[] = new Array<DocumentItem>();

    for (let i = 0; i < this.documents.length; i++) {
      const oldDoc: DocumentItem = this.documents[i];

      if (oldDoc.documentId === status.documentId) {
        nextDocuments.push({
          documentId: oldDoc.documentId,
          name: status.title,
          status: status.status,
          progress: status.progress,
          pageCount: status.pageCount,
          chunkCount: status.chunkCount,
          error: status.error
        });
      } else {
        nextDocuments.push(oldDoc);
      }
    }

    this.documents = nextDocuments;
    this.uiRefreshKey = this.uiRefreshKey + 1;
  }

  private async pollDocumentStatus(documentId: string): Promise<void> {
    while (true) {
      const status: DocumentStatusResponse = await RagApi.getDocumentStatus(documentId);

      this.updateDocumentStatus(status);

      this.statusText = `${status.title}: ${status.status} ${status.progress}%`;

      if (status.status === 'ready') {
        this.statusText = `Ready: ${status.title}`;
        break;
      }

      if (status.status === 'error') {
        this.statusText = `Failed: ${status.error}`;
        break;
      }

      await this.sleep(1000);
    }
  }

  private errorToString(err: Object): string {
    return `${err}`;
  }

  private async selectAndProcessPdf(): Promise<void> {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;

    this.isLoading = true;
    this.statusText = 'Selecting document...';

    try {
      const pickedFile: PickedDocument = await FilePickerUtil.pickPdf(context);

      this.statusText = `Uploading ${pickedFile.name}...`;

      const uploaded: UploadedDocument = await RagApi.uploadDocument(pickedFile);

      const newDocument: DocumentItem = {
        documentId: uploaded.documentId,
        name: uploaded.title,
        status: uploaded.status,
        progress: uploaded.progress,
        pageCount: uploaded.pageCount,
        chunkCount: 0,
        error: ''
      };

      const nextDocuments: DocumentItem[] = new Array<DocumentItem>();

      for (let i = 0; i < this.documents.length; i++) {
        nextDocuments.push(this.documents[i]);
      }

      nextDocuments.push(newDocument);
      this.documents = nextDocuments;
      this.uiRefreshKey = this.uiRefreshKey + 1;

      await this.pollDocumentStatus(uploaded.documentId);
    } catch (err) {
      this.statusText = `Failed: ${this.errorToString(err as Object)}`;
    } finally {
      this.isLoading = false;
    }
  }

  private async removeDocument(index: number): Promise<void> {
    if (index < 0 || index >= this.documents.length) {
      return;
    }

    const target: DocumentItem = this.documents[index];

    try {
      this.statusText = `Removing ${target.name}...`;

      await RagApi.deleteDocument(target.documentId);

      const nextDocuments: DocumentItem[] = new Array<DocumentItem>();

      for (let i = 0; i < this.documents.length; i++) {
        if (i !== index) {
          nextDocuments.push(this.documents[i]);
        }
      }

      this.documents = nextDocuments;
      this.uiRefreshKey = this.uiRefreshKey + 1;

      this.statusText = `Removed ${target.name}`;
    } catch (err) {
      this.statusText = `Remove failed: ${this.errorToString(err as Object)}`;
    }
  }

  private async generateSummary(): Promise<void> {
    const documentIds: string[] = this.getSummaryDocumentIds();

    if (documentIds.length === 0) {
      this.summaryText = 'Please select at least one ready PDF.';
      return;
    }

    this.isLoading = true;
    this.summaryText = '';
    this.keyPoints = new Array<string>();
    this.studyQuestions = new Array<string>();

    try {
      const req: SummaryRequest = {
        documentIds: documentIds,
        length: this.selectedLength
      };

      const result: SummaryResponse = await RagApi.summarize(req);

      this.summaryText = result.summary;
      this.keyPoints = result.keyPoints;
      this.studyQuestions = result.studyQuestions;
    } catch (err) {
      this.summaryText = `Failed to generate summary: ${this.errorToString(err as Object)}`;
    } finally {
      this.isLoading = false;
    }
  }

  private async askQuestion(): Promise<void> {
    if (this.documents.length === 0) {
      this.answer = 'Please select at least one PDF first.';
      return;
    }

    if (this.question.trim().length === 0) {
      this.answer = 'Please enter a question.';
      return;
    }

    this.isLoading = true;
    this.answer = '';
    this.sourcesText = '';

    try {
      const req: ChatRequest = {
        documentIds: this.getDocumentIds(),
        question: this.question
      };

      const result: ChatResponse = await RagApi.chat(req);

      this.answer = result.answer;

      const sourceLines: string[] = new Array<string>();

      for (let i = 0; i < result.sources.length; i++) {
        const source: SourceChunk = result.sources[i];
        sourceLines.push(`${source.documentTitle}, page ${source.page}: ${source.text}`);
      }

      this.sourcesText = sourceLines.join('\n\n');
    } catch (err) {
      this.answer = `Failed to answer question: ${this.errorToString(err as Object)}`;
    } finally {
      this.isLoading = false;
    }
  }

  private getReadyDocuments(): DocumentItem[] {
    const readyDocs: DocumentItem[] = new Array<DocumentItem>();

    for (let i = 0; i < this.documents.length; i++) {
      if (this.documents[i].status === 'ready') {
        readyDocs.push(this.documents[i]);
      }
    }

    return readyDocs;
  }

  private isSummaryDocumentSelected(documentId: string): boolean {
    for (let i = 0; i < this.selectedSummaryDocumentIds.length; i++) {
      if (this.selectedSummaryDocumentIds[i] === documentId) {
        return true;
      }
    }

    return false;
  }

  private getReadyDocumentIds(): string[] {
    const readyDocs: DocumentItem[] = this.getReadyDocuments();
    const ids: string[] = new Array<string>();

    for (let i = 0; i < readyDocs.length; i++) {
      ids.push(readyDocs[i].documentId);
    }

    return ids;
  }

  private selectAllSummaryDocuments(): void {
    this.selectedSummaryDocumentIds = this.getReadyDocumentIds();
  }

  private clearSummaryDocuments(): void {
    this.selectedSummaryDocumentIds = new Array<string>();
  }

  private toggleSummaryDocument(documentId: string): void {
    const nextIds: string[] = new Array<string>();
    let found: boolean = false;

    for (let i = 0; i < this.selectedSummaryDocumentIds.length; i++) {
      const selectedId: string = this.selectedSummaryDocumentIds[i];

      if (selectedId === documentId) {
        found = true;
      } else {
        nextIds.push(selectedId);
      }
    }

    if (!found) {
      nextIds.push(documentId);
    }

    this.selectedSummaryDocumentIds = nextIds;
  }

  private getSummaryDocumentIds(): string[] {
    const readyDocs: DocumentItem[] = this.getReadyDocuments();
    const ids: string[] = new Array<string>();

    for (let i = 0; i < readyDocs.length; i++) {
      const doc: DocumentItem = readyDocs[i];

      if (this.isSummaryDocumentSelected(doc.documentId)) {
        ids.push(doc.documentId);
      }
    }

    return ids;
  }

  private isAllSummaryDocumentsSelected(): boolean {
    const readyDocs: DocumentItem[] = this.getReadyDocuments();

    if (readyDocs.length === 0) {
      return false;
    }

    for (let i = 0; i < readyDocs.length; i++) {
      if (!this.isSummaryDocumentSelected(readyDocs[i].documentId)) {
        return false;
      }
    }

    return true;
  }

  private getSelectedSummaryLabel(): string {
    const readyCount: number = this.getReadyDocuments().length;
    const selectedCount: number = this.getSummaryDocumentIds().length;

    if (readyCount === 0) {
      return 'No ready documents';
    }

    if (selectedCount === 0) {
      return 'No summary documents selected';
    }

    if (selectedCount === readyCount) {
      return `All ${readyCount} ready document(s)`;
    }

    return `${selectedCount} of ${readyCount} ready document(s)`;
  }

  private async loadDocumentsFromServer(): Promise<void> {
    try {
      const serverDocs: DocumentStatusResponse[] = await RagApi.listDocuments();

      const nextDocuments: DocumentItem[] = new Array<DocumentItem>();

      for (let i = 0; i < serverDocs.length; i++) {
        const doc: DocumentStatusResponse = serverDocs[i];

        nextDocuments.push({
          documentId: doc.documentId,
          name: doc.title,
          status: doc.status,
          progress: doc.progress,
          pageCount: doc.pageCount,
          chunkCount: doc.chunkCount,
          error: doc.error
        });
      }

      this.documents = nextDocuments;
      this.uiRefreshKey = this.uiRefreshKey + 1;
      this.selectAllSummaryDocuments();
    } catch (err) {
      this.statusText = `Load documents failed: ${this.errorToString(err as Object)}`;
    }
  }

  aboutToAppear(): void {
    this.loadDocumentsFromServer();
  }

  build() {
    Column() {
      Text('Document Study Assistant')
        .fontSize(26)
        .fontWeight(FontWeight.Bold)
        .margin({ top: 20, bottom: 12 })

      if (this.statusText.length > 0) {
        Text(this.statusText)
          .fontSize(13)
          .fontColor('#666666')
          .margin({ bottom: 8 })
      }

      Tabs() {
        TabContent() {
          this.DocumentTab()
        }
        .tabBar('Documents')

        TabContent() {
          this.SummaryTab()
        }
        .tabBar('Summary')

        TabContent() {
          this.ChatTab()
        }
        .tabBar('Q&A')
      }
      .barPosition(BarPosition.Start)
      .width('100%')
      .height('100%')
    }
    .padding(16)
    .width('100%')
    .height('100%')
    .backgroundColor('#F6F7FB')
  }

  @Builder
  DocumentTab() {
    Column() {
      Button(this.isLoading ? 'Working...' : 'Select PDF')
        .width('100%')
        .height(48)
        .margin({ bottom: 12 })
        .enabled(!this.isLoading)
        .onClick(() => {
          this.selectAndProcessPdf();
        })

      Text('Processed Documents')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .margin({ bottom: 8 })

      if (this.documents.length === 0) {
        Text('No documents selected yet.')
          .fontSize(15)
          .fontColor('#777777')
          .margin({ top: 12 })
      } else {
        List() {
          ForEach(
            this.documents,
            (doc: DocumentItem, index: number) => {
              ListItem() {
                Row() {
                  Column() {
                    Text(doc.name)
                      .fontSize(16)
                      .fontWeight(FontWeight.Medium)

                    Text(`${doc.status} ${doc.progress}% | pages: ${doc.pageCount} | chunks: ${doc.chunkCount}`)
                      .fontSize(13)
                      .fontColor('#666666')

                    if (doc.error.length > 0) {
                      Text(doc.error)
                        .fontSize(12)
                        .fontColor('#FF3B30')
                        .maxLines(3)
                    }
                  }
                  .flexGrow(1)

                  Button('Remove')
                    .height(32)
                    .backgroundColor('#FF3B30')
                    .onClick(() => {
                      this.removeDocument(index);
                    })
                }
                .padding(12)
                .width('100%')
                .backgroundColor(Color.White)
                .borderRadius(12)
                .margin({ bottom: 8 })
              }
            },
            (doc: DocumentItem, index: number): string => {
              return `${doc.documentId}-${doc.status}-${doc.progress}-${doc.chunkCount}-${this.uiRefreshKey}`;
            }
          )
        }
        .width('100%')
      }
    }
    .width('100%')
  }

  @Builder
  SummaryTab() {
    Column() {
      Text('Summary Documents')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .margin({ bottom: 8 })

      Text(`Current: ${this.getSelectedSummaryLabel()}`)
        .fontSize(14)
        .fontColor('#666666')
        .margin({ bottom: 8 })

      Column() {
        Row() {
          Button('All')
            .height(36)
            .backgroundColor(this.isAllSummaryDocumentsSelected() ? '#0A59F7' : '#999999')
            .onClick(() => {
              this.selectAllSummaryDocuments();
            })

          Button('Clear')
            .height(36)
            .margin({ left: 8 })
            .backgroundColor('#999999')
            .onClick(() => {
              this.clearSummaryDocuments();
            })
        }
        .margin({ bottom: 8 })

        ForEach(
          this.getReadyDocuments(),
          (doc: DocumentItem) => {
            Button(`${this.isSummaryDocumentSelected(doc.documentId) ? '✓ ' : ''}${doc.name}`)
              .width('100%')
              .height(40)
              .margin({ bottom: 6 })
              .backgroundColor(this.isSummaryDocumentSelected(doc.documentId) ? '#0A59F7' : '#999999')
              .onClick(() => {
                this.toggleSummaryDocument(doc.documentId);
              })
          },
          (doc: DocumentItem): string => {
            return `summary-doc-${doc.documentId}-${this.isSummaryDocumentSelected(doc.documentId)}-${this.uiRefreshKey}`;
          }
        )
      }
      .margin({ bottom: 16 })

      Text('Summary Length')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .margin({ bottom: 8 })

      Row() {
        Button('Short')
          .backgroundColor(this.selectedLength === 'short' ? '#0A59F7' : '#999999')
          .onClick(() => {
            this.selectedLength = 'short';
          })

        Button('Medium')
          .margin({ left: 8 })
          .backgroundColor(this.selectedLength === 'medium' ? '#0A59F7' : '#999999')
          .onClick(() => {
            this.selectedLength = 'medium';
          })

        Button('Long')
          .margin({ left: 8 })
          .backgroundColor(this.selectedLength === 'long' ? '#0A59F7' : '#999999')
          .onClick(() => {
            this.selectedLength = 'long';
          })

        Button('Detailed')
          .margin({ left: 8 })
          .backgroundColor(this.selectedLength === 'detailed' ? '#0A59F7' : '#999999')
          .onClick(() => {
            this.selectedLength = 'detailed';
          })
      }
      .margin({ bottom: 12 })

      Text(`Current length: ${this.selectedLength}`)
        .fontSize(14)
        .fontColor('#666666')
        .margin({ bottom: 12 })

      Button(this.isLoading ? 'Generating...' : 'Generate Summary')
        .width('100%')
        .height(48)
        .enabled(!this.isLoading)
        .onClick(() => {
          this.generateSummary();
        })

      Scroll() {
        Column() {
          Text(this.summaryText)
            .fontSize(16)
            .width('100%')
            .maxLines(200)
            .margin({ top: 16, bottom: 16 })

          if (this.keyPoints.length > 0) {
            Text('Key Points')
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .width('100%')
              .margin({ bottom: 8 })

            ForEach(
              this.keyPoints,
              (point: string) => {
                Text(`• ${point}`)
                  .fontSize(15)
                  .width('100%')
                  .maxLines(20)
                  .margin({ bottom: 6 })
              },
              (point: string, index: number): string => {
                return `${index}-${point}`;
              }
            )
          }

          if (this.studyQuestions.length > 0) {
            Text('Study Questions')
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .width('100%')
              .margin({ top: 16, bottom: 8 })

            ForEach(
              this.studyQuestions,
              (q: string) => {
                Text(`• ${q}`)
                  .fontSize(15)
                  .width('100%')
                  .maxLines(30)
                  .margin({ bottom: 8 })
              },
              (q: string, index: number): string => {
                return `${index}-${q}`;
              }
            )
          }
        }
        .width('100%')
        .padding({ bottom: 60 })
      }
      .width('100%')
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
  }

  @Builder
  ChatTab() {
    Column() {
      TextArea({
        placeholder: 'Ask a question based on the selected documents...',
        text: this.question
      })
        .height(100)
        .width('100%')
        .backgroundColor(Color.White)
        .borderRadius(12)
        .padding(12)
        .onChange((value: string) => {
          this.question = value;
        })

      Button(this.isLoading ? 'Thinking...' : 'Ask')
        .width('100%')
        .height(48)
        .margin({ top: 12, bottom: 16 })
        .enabled(!this.isLoading)
        .onClick(() => {
          this.askQuestion();
        })

      Scroll() {
        Column() {
          Text('Answer')
            .fontSize(18)
            .fontWeight(FontWeight.Bold)
            .margin({ bottom: 8 })

          Text(this.answer)
            .fontSize(16)
            .margin({ bottom: 16 })

          if (this.sourcesText.length > 0) {
            Text('Sources')
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .margin({ bottom: 8 })

            Text(this.sourcesText)
              .fontSize(14)
              .fontColor('#666666')
          }
        }
      }
      .height('70%')
    }
    .width('100%')
  }
}