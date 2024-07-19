import io
import json
import pathlib
import re
import numpy as np
import pydash
from fitz import fitz
from typing import List
from assignment_1.assignment_1_1.helper_convert import normalize, extract_page_blocks


def dump_paragraphs(page_blocks: List[dict]):
    words = blocks_to_words(page_blocks)

    if not words:
        return

    words = remove_zero_bboxes(words)
    words = pydash.arrays.uniq_by(words, lambda x: (x[0], x[1], x[2], x[3], x[4]))
    words = remove_dir_outliner(words)
    words = filter_top_half(words)
    words = remove_large_bbox(words)

    words = sorted(words, key=lambda x: (x[5], x[6], x[7]))
    np_word_bboxes = np.array([
        it[:4] for it in words
    ])

    np_word_bboxes = np.stack([
        np.min(np_word_bboxes[:, [0, 2]], axis=1),
        np.min(np_word_bboxes[:, [1, 3]], axis=1),
        np.max(np_word_bboxes[:, [0, 2]], axis=1),
        np.max(np_word_bboxes[:, [1, 3]], axis=1),
    ], axis=1)

    word_text = [it[4] for it in words]

    I_BLOCK_NO = 5
    I_LINE_NO = 6
    I_WORD_NO = 7

    def buf_to_line(line_buf):
        bbs = np_word_bboxes[line_buf]
        bb = np.array([
            bbs[:, 0].min(),
            bbs[:, 1].min(),
            bbs[:, 2].max(),
            bbs[:, 3].max(),
        ])

        chars = []
        x0_list = []

        char_heights = []
        char_widths = []

        for i, it in enumerate(line_buf):
            if i > 0:
                if words[i - 1][I_WORD_NO] != words[i][I_WORD_NO]:
                    chars.append(' ')
                    x0_list.append(np_word_bboxes[line_buf[i - 1], 2])

            wc = len(word_text[it])
            bb_i = np_word_bboxes[it]
            for j in range(wc):
                chars.append(word_text[it][j])
                x0_list.append(bb_i[0] + j * (bb_i[2] - bb_i[0]) / wc)
                char_heights.append(bb_i[3] - bb_i[1])
                char_widths.append((bb_i[2] - bb_i[0]) / wc)

        ret = {
            "bbox": bb,
            "char_height": np.array(char_heights).mean(),
            "char_width": np.array(char_widths).mean(),
            "text": ''.join(chars),
            "chars": {"text": chars, "x0_list": x0_list}
        }

        return ret

    def buf_to_para(lines):
        bbs = np.array([it['bbox'] for it in lines])

        bb = np.array([bbs[:, 0].min(), bbs[:, 1].min(), bbs[:, 2].max(), bbs[:, 3].max()])

        ret = {
            "bbox": bb,
            "lines": lines,
            "text": '\n'.join([it['text'] for it in lines])
        }

        return ret

    line_buf = []
    para_buf = []

    cur_line_no = words[0][I_LINE_NO]
    cur_block_no = words[0][I_BLOCK_NO]

    for i, it in enumerate(words + [[-1] * 8]):
        block_no = it[I_BLOCK_NO]
        line_no = it[I_LINE_NO]

        if cur_line_no == line_no and cur_block_no == block_no:
            line_buf.append(i)
        else:
            para_buf.append(buf_to_line(line_buf))
            if cur_block_no == block_no:
                cur_block_no = block_no
                cur_line_no = line_no
                line_buf = [i]
            else:
                para = buf_to_para(para_buf)
                yield para
                cur_line_no = line_no
                cur_block_no = block_no
                line_buf = [i]
                para_buf = []


def blocks_to_words(page_blocks: List[dict]):
    """
    Args:
        page_blocks   list of dict by page.get_text("dict")["blocks"]

    Returns [(x0, y0, x1, y1, "word", block_no, line_no, word_no, dir_x, dir_y)]
    """
    acc = []

    for block_no, block in enumerate(page_blocks):
        for line_no, line in enumerate(block.get('lines') or []):
            dir_x, dir_y = line['dir']
            for word_no, span in enumerate(line['spans']):
                span_text = span['text']
                for m in re.finditer(r'\S+|\s+', span_text):
                    bb = merge_bbox(*[char['bbox'] for char in span['chars'][m.start():m.end()]])
                    word = m.group()
                    acc.append((*bb, word, block_no, line_no, word_no, dir_x, dir_y))

    return acc


def merge_bbox(*bbox_list):
    if bbox_list:
        merge_x0 = min([bbox[0] for bbox in bbox_list])
        merge_y0 = min([bbox[1] for bbox in bbox_list])
        merge_x1 = max([bbox[2] for bbox in bbox_list])
        merge_y1 = max([bbox[3] for bbox in bbox_list])
        return [merge_x0, merge_y0, merge_x1, merge_y1]
    return None


def remove_zero_bboxes(words):
    np_bboxes = np.array([it[:4] for it in words])
    np_bboxes_sz = (np_bboxes[:, 2] - np_bboxes[:, 0]) * (np_bboxes[:, 3] - np_bboxes[:, 1])
    np_bboxes_sz_zero = np_bboxes_sz == 0
    words = [it for i, it in enumerate(words) if not np_bboxes_sz_zero[i]]
    return words


def remove_dir_outliner(words, threshold=3):
    dir_x = [x[8] for x in words]
    dir_y = [x[9] for x in words]
    # todo:  Question 1
    return words


def filter_top_half(words):
    # todo: Question 2
    return words


def remove_large_bbox(words):
    # todo: Question 3
    return words


def pymupdf_transform_to_idp_format(pdf_bin):
    buf = io.BytesIO(pdf_bin)
    doc = fitz.Document(stream=buf, filetype='pdf')
    pages = []
    map_type_to_names = {
        0: 'text',
        1: 'image'
    }
    for page_id, page in enumerate(doc):
        page_blocks = extract_page_blocks(doc.load_page(page_id))
        groups = pydash.group_by([b['bbox'] + (b['type'],) for b in page_blocks], 4)
        blocks_by_type = {map_type_to_names[k]: len(v) for k, v in groups.items()}
        if blocks_by_type.get('text', 0) <= 0:
            pages.append(None)
            continue
        paragraphs = []
        for paragraph in dump_paragraphs(page_blocks):
            paragraphs.append(paragraph)
        pages.append({
            "bbox": list(page.rect),
            "paragraphs": paragraphs
        })

    return {"pages": pages}


def convert_e_pdf(pdf_bin):
    pymupdf_convert_result = pymupdf_transform_to_idp_format(pdf_bin)
    text_result = normalize(pymupdf_convert_result)
    return text_result

def extract_images(pdf_bin, save_path):
    # todo: Question 4
    pass


if __name__ == "__main__":
    dataset_dir = pathlib.Path(__file__).parents[1]
    input_path = 'assignment_1_1/data/epdf_sample.pdf'
    pdf_path = dataset_dir / input_path
    save_path = 'assignment_1_1/data/epdf_sample'
    test_file = dataset_dir / f'{save_path}.json'

    pdf_bin = pdf_path.read_bytes()
    result = convert_e_pdf(pdf_bin)
    with open(test_file, 'w') as f:
        json.dump(result, f, indent=2)

    # extract_images(pdf_bin, save_path)








