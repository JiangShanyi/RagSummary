import { common } from '@kit.AbilityKit';

import { RagApi } from '../api/RagApi';
import { FilePickerUtil } from '../utils/FilePickerUtil';
import {
  PickedDocument,
  SummaryResponse,
  ChatResponse
} from '../model/Types';

@Entry
@Component
struct Index {
  @State documents: Array<{ name: string; text: string }> = [];

  @State selectedLength: string = 'medium';

  @State summaryText: string = '';
  @State keyPoints: string[] = [];
  @State studyQuestions: string[] = [];

  @State question: string = '';
  @State answer: string = '';
  @State sourcesText: string = '';

  @State statusText: string = '';
  @State isLoading: boolean = false;

  private async selectAndProcessPdf(): Promise<void> {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;

    this.isLoading = true;
    this.statusText = 'Selecting document...';

    try {
      const pickedFile: PickedDocument = await FilePickerUtil.pickPdf(context);

      this.statusText = `Processing ${pickedFile.name}...`;
      const text: string = await RagApi.extractPdfText(pickedFile);

      this.documents = [...this.documents, { name: pickedFile.name, text: text }];
      this.statusText = `Processed: ${pickedFile.name}`;
    } catch (err) {
      this.statusText = `Failed: ${JSON.stringify(err)}`;
    } finally {
      this.isLoading = false;
    }
  }

  private removeDocument(index: number): void {
    this.documents = this.documents.filter((_, i) => i !== index);
  }

  private async generateSummary(): Promise<void> {
    if (this.documents.length === 0) {
      this.summaryText = 'Please select at least one PDF first.';
      return;
    }

    this.isLoading = true;
    this.summaryText = '';

    try {
      const texts: string[] = this.documents.map(d => d.text);
      const result: SummaryResponse = await RagApi.summarize({
        documentIds: [],
        length: this.selectedLength
      }, texts);

      this.summaryText = result.summary;
      this.keyPoints = result.keyPoints;
      this.studyQuestions = result.studyQuestions;
    } catch (err) {
      this.summaryText = `Failed to generate summary: ${JSON.stringify(err)}`;
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
      const texts: string[] = this.documents.map(d => d.text);
      const result: ChatResponse = await RagApi.chat({
        documentIds: [],
        question: this.question
      }, texts);

      this.answer = result.answer;
      this.sourcesText = result.sources
        .map((source) => `${source.documentTitle}, page ${source.page}: ${source.text}`)
        .join('\n\n');
    } catch (err) {
      this.answer = `Failed to answer question: ${JSON.stringify(err)}`;
    } finally {
      this.isLoading = false;
    }
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
          ForEach(this.documents, (doc: { name: string; text: string }, index: number) => {
            ListItem() {
              Row() {
                Column() {
                  Text(doc.name)
                    .fontSize(16)
                    .fontWeight(FontWeight.Medium)

                  Text(`${doc.text.length} characters`)
                    .fontSize(13)
                    .fontColor('#666666')
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
          })
        }
        .width('100%')
      }
    }
    .width('100%')
  }

  @Builder
  SummaryTab() {
    Column() {
      Text('Summary Length')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .margin({ bottom: 8 })

      Row() {
        Button('Short')
          .onClick(() => this.selectedLength = 'short')

        Button('Medium')
          .margin({ left: 8 })
          .onClick(() => this.selectedLength = 'medium')

        Button('Long')
          .margin({ left: 8 })
          .onClick(() => this.selectedLength = 'long')

        Button('Detailed')
          .margin({ left: 8 })
          .onClick(() => this.selectedLength = 'detailed')
      }
      .margin({ bottom: 12 })

      Text(`Current: ${this.selectedLength}`)
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
            .margin({ top: 16, bottom: 16 })

          if (this.keyPoints.length > 0) {
            Text('Key Points')
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .margin({ bottom: 8 })

            ForEach(this.keyPoints, (point: string) => {
              Text(`• ${point}`)
                .fontSize(15)
                .margin({ bottom: 6 })
            })
          }

          if (this.studyQuestions.length > 0) {
            Text('Study Questions')
              .fontSize(18)
              .fontWeight(FontWeight.Bold)
              .margin({ top: 16, bottom: 8 })

            ForEach(this.studyQuestions, (q: string) => {
              Text(`• ${q}`)
                .fontSize(15)
                .margin({ bottom: 6 })
            })
          }
        }
      }
      .height('70%')
    }
    .width('100%')
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
