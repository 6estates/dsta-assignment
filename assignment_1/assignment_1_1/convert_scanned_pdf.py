import json
import pathlib
import pydash
import numpy as np
from PIL import Image
from fitz import fitz
from assignment_1.assignment_1_1.helper_convert import extract_page_blocks, final_idp_doc
from assignment_1.assignment_1_1.tesseract_ocr import TesseractOCR, TesseractResultTransformer


def convert_one_page_from_pdf_page(pdf_doc, page_id):
    pdf_page = pdf_doc[page_id]
    rgb_array = pdf_render_page(pdf_page)
    img_pil = Image.fromarray(rgb_array)
    response = TesseractOCR().detect_text_in_image_sync(img_pil)
    rt = TesseractResultTransformer(np.array(rgb_array))
    ocr_data = {page_id + 1: {"responses": response}}
    if hasattr(rt, 'add_normalized_bbox'):
        rt.add_normalized_bbox(ocr_data)
    filename = "-"
    # 后处理
    ocr_data_trans = rt.transform(filename, filename, ocr_data)
    idp_page = {
        **ocr_data_trans["pages"][0],
        "bbox": [0, 0, rgb_array.shape[1], rgb_array.shape[0]],
    }
    return idp_page


def pdf_render_page(page: fitz.Page):
    pix = page.get_pixmap()
    img_Image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    canvas = np.array(img_Image)
    return canvas


def ocr_transform_to_idp_format(pdf_path):
    pages = []
    doc = fitz.open(pathlib.Path(pdf_path))
    map_type_to_names = {
        0: 'text',
        1: 'image'
    }
    page_count = doc.page_count
    for page_id in range(page_count):
        page_blocks = extract_page_blocks(doc.load_page(page_id))
        groups = pydash.group_by([b['bbox'] + (b['type'],) for b in page_blocks], 4)
        blocks_by_type = {map_type_to_names[k]: len(v) for k, v in groups.items()}
        if blocks_by_type.get('image', 0) <= 0:
            continue
        pages.append(convert_one_page_from_pdf_page(pdf_doc=doc, page_id=page_id))
    return {"pages": pages}


def convert_scanned_pdf(pdf_path):
    ocr_convert_result = ocr_transform_to_idp_format(pdf_path)
    text_rotation_result = final_idp_doc(ocr_convert_result)
    return text_rotation_result


if __name__ == "__main__":
    dataset_dir = pathlib.Path(__file__).parents[2]
    input_path = 'test data/testocr.png'
    pdf_path = dataset_dir / input_path
    save_path = 'test result/testocr'
    test_file = dataset_dir / f'{save_path}.json'
    result = convert_scanned_pdf(pdf_path)
    with open(test_file, 'w') as f:
        json.dump(result, f, indent=2)