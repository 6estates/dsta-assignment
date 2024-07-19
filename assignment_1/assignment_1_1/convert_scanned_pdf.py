import io
import json
import pathlib
import pydash
import numpy as np
import PIL
from PIL import Image, ExifTags
from fitz import fitz
from assignment_1.assignment_1_1.helper_convert import extract_page_blocks, normalize
from assignment_1.assignment_1_1.tesseract_ocr import TesseractOCR, TesseractResultTransformer

def image_to_pdf(image_path):
    with open(image_path, 'rb') as file:
        bin_in = file.read()
        bin_in = io.BytesIO(bin_in)

    try:
        pil_image = Image.open(bin_in)
    except PIL.UnidentifiedImageError:
        raise

    if pil_image.mode in ('RGBA', 'LA') or (pil_image.mode == 'P' and 'transparency' in pil_image.info):
        pil_image = pil_image.convert('RGB')

    for orientation_key in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation_key] == 'Orientation':
            if hasattr(pil_image, '_getexif'):
                img_exif = pil_image._getexif()
                if hasattr(img_exif, 'items'):
                    exif = dict(pil_image._getexif().items())
                    orientation = exif.get(orientation_key)
                    # Handle EXIF Orientation
                    if orientation == 1:
                        # Normal image - nothing to do
                        pass
                    # todo: Question 9
            break

    pdf_fp_out = io.BytesIO()
    pil_image.save(pdf_fp_out, "PDF", creationDate=None, modDate=None)

    return pdf_fp_out.getvalue()

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


def ocr_transform_to_idp_format(pdf_bin):
    buf = io.BytesIO(pdf_bin)
    doc = fitz.Document(stream=buf, filetype='pdf')
    pages = []
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


def convert_scanned_pdf(pdf_bin):
    ocr_convert_result = ocr_transform_to_idp_format(pdf_bin)
    text_result = normalize(ocr_convert_result)
    return text_result


if __name__ == "__main__":
    dataset_dir = pathlib.Path(__file__).parents[1]
    input_path = 'assignment_1_1/data/scanned_pdf_sample.pdf'
    file_path = dataset_dir / input_path
    save_path = 'assignment_1_1/data/scanned_pdf_sample'
    test_file = dataset_dir / f'{save_path}.json'

    # todo: Question 9

    pdf_bin = file_path.read_bytes()
    result = convert_scanned_pdf(pdf_bin)
    with open(test_file, 'w') as f:
        json.dump(result, f, indent=2)