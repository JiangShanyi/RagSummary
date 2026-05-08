import { picker } from '@kit.CoreFileKit';
import { common } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';

import { PickedDocument } from '../model/Types';

export class FilePickerUtil {
  static async pickPdf(context: common.UIAbilityContext): Promise<PickedDocument> {
    const documentSelectOptions = new picker.DocumentSelectOptions();
    documentSelectOptions.maxSelectNumber = 1;
    documentSelectOptions.fileSuffixFilters = ['.pdf'];

    const documentViewPicker = new picker.DocumentViewPicker(context);

    let uris: Array<string>;

    try {
      uris = await documentViewPicker.select(documentSelectOptions);
    } catch (err) {
      throw new Error(`Document picker select failed: ${FilePickerUtil.formatError(err as Object)}`);
    }

    if (uris.length === 0) {
      throw new Error('No file selected.');
    }

    const selectedUri: string = uris[0];
    const fileName: string = FilePickerUtil.guessFileName(selectedUri);

    console.info(`[FilePickerUtil] selectedUri=${selectedUri}`);
    console.info(`[FilePickerUtil] fileName=${fileName}`);

    return {
      uri: selectedUri,
      name: fileName,
      mimeType: 'application/pdf'
    };
  }

  private static guessFileName(uri: string): string {
    let text: string = uri;

    const queryIndex: number = text.indexOf('?');
    if (queryIndex >= 0) {
      text = text.substring(0, queryIndex);
    }

    const slashIndex: number = text.lastIndexOf('/');
    let name: string = slashIndex >= 0 ? text.substring(slashIndex + 1) : text;

    try {
      name = decodeURIComponent(name);
    } catch (err) {
    }

    if (name.length === 0 || !name.toLowerCase().endsWith('.pdf')) {
      return `picked_${Date.now()}.pdf`;
    }

    return name;
  }

  private static formatError(err: Object): string {
    const businessErr = err as BusinessError;

    const codeText: string = businessErr.code === undefined ? 'undefined' : businessErr.code.toString();
    const messageText: string = businessErr.message === undefined ? String(err) : businessErr.message;

    return `code=${codeText}, message=${messageText}`;
  }
}