import uuid
import numpy as np
from fitz import fitz, Page


def normalize(idp_file):
    """
    后处理idpdoc:
    1. 添加字段: raw_file_id, raw_file_name, page, paragraph_id, line_id, uuid, confidence
    2. 将numpy array转换为list等普通类型, 以便json序列化
    """

    idp_file['raw_file_id'] = '-'
    idp_file['raw_file_name'] = '-'

    def deep_json_serizalizable(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, fitz.Rect):
                    obj[k] = list(v)
                elif callable(getattr(v, 'tolist', None)):
                    obj[k] = v.tolist()
                elif isinstance(v, (list, dict)):
                    deep_json_serizalizable(v)
        elif isinstance(obj, list):
            for i in obj:
                deep_json_serizalizable(i)
        else:
            pass #ignore other types

    for i, page in enumerate(idp_file['pages']):
        page['page'] = i + 1

        deep_rounding(page, 1.0)

        for o in ['signatures', 'stamps', 'tables']:
            if o not in page:
                page[o] = []

        if 'confidence' not in page:
            page['confidence'] = 0.9

        for j, para in enumerate(page['paragraphs']):
            para['paragraph_id'] = j + 1
            para['uuid'] = uuid.uuid4().__str__()
            if 'confidence' not in para:
                para['confidence'] = 0.9

            for k, l in enumerate(para['lines']):
                l['line_id'] = k + 1
                if 'confidence' not in l:
                    l['confidence'] = 0.9
                l['uuid'] = uuid.uuid4().__str__()

    deep_json_serizalizable(idp_file)

    return idp_file


def deep_rounding(obj, scale):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ['bbox', 'x0_list', 'char_height', 'char_width']:
                obj[k] = np.rint(np.array(v) * scale).astype(int)
            elif isinstance(v, (list, dict)):
                deep_rounding(v, scale)
    elif isinstance(obj, list):
        for i in obj:
            deep_rounding(i, scale)
    else:
        pass


def extract_page_blocks(pdf_page: Page):
    all_blocks = pdf_page.get_text('rawdict')['blocks']
    existing_numbers = {blk['number'] for blk in all_blocks if blk['type'] == 1}

    # back compatibility
    for blk in all_blocks:
        if blk['type'] == 0:
            for ln in blk['lines']:
                for span in ln['spans']:
                    span['text'] = ''.join(char['c'] for char in span['chars'])

    # pymupdf only includes images which are fully within media_bbox
    # we check if there are any images partially within media_bbox
    # more info: https://github.com/pymupdf/PyMuPDF/discussions/2714#discussioncomment-7500199
    extended_blocks = pdf_page.get_text('rawdict', clip=pdf_page.mediabox + (-10, -10, 10, 10))['blocks']
    extended_image_blocks = [blk for blk in extended_blocks if blk['type'] == 1]
    for image_block in extended_image_blocks:
        if image_block['number'] not in existing_numbers:
            all_blocks.append(image_block)

    return all_blocks


