import os
import numpy as np
import pandas as pd
from pytesseract import pytesseract, TesseractNotFoundError, TesseractError

TESSERACT_OCR_TIMEOUT = os.getenv('TESSERACT_OCR_TIMEOUT', '60')

# TODO : update training data

class TesseractOCR:
    def __init__(self, config=r'--oem 2 --psm 3', lang='eng', ocr_time=int(TESSERACT_OCR_TIMEOUT)):
        self.tesseract_ocr_timeout = ocr_time
        self.config = config
        self.lang = lang

    def detect_text_in_image_sync(self, image_pil):
        try:
            image_data = pytesseract.image_to_data(image_pil, output_type=pytesseract.Output.DICT,
                                                   timeout=self.tesseract_ocr_timeout,
                                                   config=self.config, lang=self.lang)
            return image_data
        except TesseractNotFoundError:
            raise Exception("Tesseract executable not found. Please ensure Tesseract is installed and accessible.")
        except TesseractError as e:
            raise Exception(f"An error occurred during Tesseract OCR: {e}")


class TesseractResultTransformer:
    def __init__(self, image):
        self.image = image

    @classmethod
    def transform(cls, file_name, ori_name, tesseract_ocr_data):
        """
        Google ocr to custom ocr format
        """
        pages = []
        for page in sorted(tesseract_ocr_data.keys()):
            try:
                page_data = tesseract_ocr_data.get(page)
                pages.append(cls.transform_page(page, page_data))
            except Exception as e:
                raise e
        result = {'file_name': ori_name, 'pages': pages}
        return result

    @classmethod
    def transform_page(cls, page_id, page_data):
        """
        Format transform for one page, from tesseract format to custom one.
        """
        custom_page_data = {
            'bbox': [0, 0, 100, 100],
            'confidence': 0.0,
            'page_id': page_id,
            'paragraphs': [],
            'signatures': [],
            'stamps': [],
            'tables': []
        }
        if not page_data.get('responses', None):
            return custom_page_data

        total_page_confidence = 0
        one_page = page_data['responses']
        block_indices = {'paragraphs': 1, 'signatures': 1, 'stamps': 1, 'tables': 1, 'lines': 1}

        # initialize variables to keep track of current block and paragraph
        customized_paragraphs_list = []

        current_word_list = []
        current_bbox_list = []
        current_conf = 0

        valid_text_cnt = 0
        df_one_page = pd.DataFrame(one_page)
        paragraph_idx = block_indices['paragraphs']
        line_idx = block_indices['lines']
        previous_block_num = None
        previous_par_num = None
        previous_line_num = None

        # processing each OCR row by line level
        for idx, row in df_one_page.iterrows():
            if row['level'] == 1:
                # get width and height of the page
                custom_page_data['bbox'][2] = row['width']
                custom_page_data['bbox'][3] = row['height']

            if len(row['text']):
                block_num = row['block_num']
                par_num = row['par_num']
                line_num = row['line_num']
                # check if new block, paragraph or line has started
                if block_num != previous_block_num or par_num != previous_par_num or line_num != previous_line_num:
                    # check if there are accumulated words and combine them as a paragraph
                    if current_word_list:
                        customized_paragraphs = cls.get_one_line_paragraph_list(current_word_list,
                                                                                current_bbox_list,
                                                                                current_conf,
                                                                                paragraph_idx, line_idx, page_id)
                        customized_paragraphs_list.append(customized_paragraphs)
                        current_word_list = []
                        current_bbox_list = []
                        current_conf = 0
                        paragraph_idx += 1
                        if par_num == previous_par_num and block_num == previous_block_num:
                            line_idx += 1
                        else:
                            line_idx = 1

                    valid_text_cnt += 1
                    total_page_confidence += float(row['conf'])
                    # different tesseract ocr engine may vary in confidence format
                    current_word_list.append(row['text'])
                    current_conf += float(row['conf'])
                    current_bbox_list.append(
                        [row['left'], row['top'], row['left'] + row['width'], row['top'] + row['height']])
                else:
                    valid_text_cnt += 1
                    total_page_confidence += float(row['conf'])
                    # different tesseract ocr engine may vary in confidence format
                    current_word_list.append(row['text'])
                    current_conf += float(row['conf'])
                    current_bbox_list.append(
                        [row['left'], row['top'], row['left'] + row['width'], row['top'] + row['height']])

                previous_block_num = row['block_num']
                previous_par_num = row['par_num']
                previous_line_num = row['line_num']

        if current_word_list:
            # for last line
            customized_paragraphs = cls.get_one_line_paragraph_list(current_word_list,
                                                                    current_bbox_list,
                                                                    current_conf, paragraph_idx, line_idx, page_id)
            customized_paragraphs_list.append(customized_paragraphs)

        custom_page_data['paragraphs'].extend(customized_paragraphs_list)
        custom_page_data['confidence'] = round(total_page_confidence / (100 * valid_text_cnt), 2) if valid_text_cnt else 0

        return custom_page_data

    @classmethod
    def get_one_line_paragraph_list(cls, word_list, bbox_list, conf, paragraph_idx, line_idx, page_id):

        paragraph = {'bbox': cls.stack_bbox(bbox_list),
                     'confidence': round(conf / (100 * len(word_list)), 2),
                     'paragraph_id': paragraph_idx,
                     'lines': cls.get_one_line(bbox_list, word_list, line_idx),
                     'text': ' '.join(word_list)}

        return paragraph

    @classmethod
    def get_one_line(cls, bbox_list, word_list, line_idx):
        bbox = cls.stack_bbox(bbox_list)
        text = ' '.join(word_list)
        # todo: Question 8

        line = {'bbox': bbox,
                # 'char_height': char_height,
                # 'char_width': char_width,
                # 'chars': {'text': char_list,
                #           'x0_list': char_x0_list},
                'line_id': line_idx,
                'text': text
                }
        return [line]

    @classmethod
    def stack_bbox(cls, bbox_list):

        if not len(bbox_list):
            return None
        if len(bbox_list) == 1:
            return bbox_list[0]

        new_bb = np.stack(bbox_list, axis=0)
        new_bb = [int(min(new_bb[:, 0])), int(min(new_bb[:, 1])), int(max(new_bb[:, 2])), int(max(new_bb[:, 3]))]

        return new_bb