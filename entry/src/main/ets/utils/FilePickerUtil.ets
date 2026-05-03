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

    try {
      const uris: Array<string> = await documentViewPicker.select(documentSelectOptions);

      if (uris.length === 0) {
        throw new Error('No file selected.');
      }

      const uri: string = uris[0];

      return {
        uri: uri,
        name: FilePickerUtil.guessFileName(uri),
        mimeType: 'application/pdf'
      };
    } catch (err) {
      const businessErr = err as BusinessError;
      throw new Error(`File picker failed. code=${businessErr.code}, message=${businessErr.message}`);
    }
  }

  private static guessFileName(uri: string): string {
    const cleanUri = decodeURIComponent(uri);
    const parts: string[] = cleanUri.split('/');
    let name: string = parts.length > 0 ? parts[parts.length - 1] : '';

    if (name.length === 0 || name.indexOf('.') < 0) {
      name = `document_${Date.now()}.pdf`;
    }

    return name.replace(/["\r\n]/g, '_');
  }
}