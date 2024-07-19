## Suggested Answers
Note: `...` means that there is no change to the original code.
## Section 1: ePDFs
### Question 1
   1. Extract directional components from words (dir_x and dir_y)    
   2. Calculate the angle between the positive x-axis and (dir_x, dir_y)
   3. Extract mean and standard deviation of angles    
   4. Filter words where absolute z-score is less than threshold

```Python
# line 161 in `convert_epdf.py`
def remove_dir_outliner(words, threshold=3):
    dir_x = [x[8] for x in words] # horizontal direction vector
    # [1, 0] means text line flows from left to right
    # [-1, 0] means text line flows from right to left
    
    dir_y = [x[9] for x in words] # vertical direction vector
    # [0, 1] means text line flows from bottom to top
    # [0, -1] means text line flows from top to bottom
    
    angle = np.arctan2(dir_y, dir_x) # to find the angle between the positive x-axis and (dir_x, dir_y)
    angle_mean = np.mean(angle) 
    angle_std = np.std(angle) 
    if angle_std > 0: 
        z_score = (angle - angle_mean) / angle_std 
        z_score_abs = np.abs(z_score)
        filtered = [x for xi, x in enumerate(words) if z_score_abs[xi] < threshold]
        return filtered
    else:
        return words
```

### Question 2
```Python
# line 168 in convert_epdf.py
def filter_top_half(words):
    y0 = [w[1] for w in words]
    mid = min(y0) + (max(y0) - min(y0)) / 2
    words = [w for w in words if w[1] <= mid]
    return words
```

### Question 3
```Python
# line 173 in convert_epdf.py
def remove_large_bbox(words):
    np_bboxes = np.array([it[:4] for it in words])
    np_bboxes_height = np_bboxes[:,3] - np_bboxes[:,1]
    sz_threshold = max(np_bboxes_height) * 0.8
    words = [it for i, it in enumerate(words) if np_bboxes_height[i] <= sz_threshold]
    return words
```

### Question 4
If you read the function `pymupdf_transform_to_idp_format` in `convert_epdf.py` line 178,
you will notice that images and text are mapped to different groups as shown below.
```Python
def pymupdf_transform_to_idp_format(pdf_bin):
    buf = io.BytesIO(pdf_bin)
    doc = fitz.Document(stream=buf, filetype='pdf')
    pages = []
    map_type_to_names = {
        0: 'text',
        1: 'image'
    }
    ...
```
Suggested answer: 
```Python
# line 209 in convert_epdf.py
def extract_images(pdf_bin, save_path):
    buf = io.BytesIO(pdf_bin)
    doc = fitz.Document(stream=buf, filetype='pdf')
    image_dicts = []
    for page_id, page in enumerate(doc):
        page_blocks = extract_page_blocks(doc.load_page(page_id))
        image_blocks = [blk for blk in page_blocks if blk['type'] == 1] # image is mapped to type 1
        image_dicts.extend(image_blocks)
        
    if len(image_dicts) > 0:
        save_path.mkdir(parents=True, exist_ok=True)
        for image_dict in image_dicts:
            save_stem = hashlib.sha256(image_dict['image']).hexdigest()
            path = save_path / f'{save_stem}.{image_dict["ext"]}'
            with open(path, 'wb') as f:
                f.write(image_dict['image'])
```

### Question 5
```Python
# line 6 in helper_convert.py
def normalize(idp_file):
    ...
    
    for i, page in enumerate(idp_file['pages']):
        if not page:
            continue
            
        page['page'] = i + 1
        deep_rounding(page, 1.0)
        ...
    
     return idp_file
```

### Question 6
```Python
# line 70 in convert_scanned_pdf.py
def ocr_transform_to_idp_format(pdf_bin, selected_pages=None):
    buf = io.BytesIO(pdf_bin)
    doc = fitz.Document(stream=buf, filetype='pdf')
    pages = []
    if not selected_pages or not isinstance(selected_pages, (list, set)):
        selected_pages = range(doc.page_count)
    else:
        selected_pages = [p - 1 for p in selected_pages]
    map_type_to_names = {
        0: 'text',
        1: 'image'
    }
    for page_id in selected_pages:
        ...
    
    return {"pages": pages}

# line 90 in convert_scanned_pdf.py
def convert_scanned_pdf(pdf_bin, selected_pages=None):
    ocr_convert_result = ocr_transform_to_idp_format(pdf_bin, selected_pages)
    text_result = normalize(ocr_convert_result)
    return text_result
```

### Question 7
```Python
# line 48 in tesseract_ocr.py
def transform_page(cls, page_id, page_data):
    ... 
    
    for idx, row in df_one_page.iterrows():
        if row['level'] == 1:
            # get width and height of the page
            custom_page_data['bbox'][2] = row['width']
            custom_page_data['bbox'][3] = row['height']

        if len(row['text']):
            block_num = row['block_num']
            par_num = row['par_num']
            if block_num != previous_block_num or par_num != previous_par_num:
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
                    if block_num == previous_block_num:
                        line_idx += 1
                    else:
                        line_idx = 1

                valid_text_cnt += 1
                total_page_confidence += float(row['conf'])
                ...
            
    return custom_page_data
```

### Question 8
Average width of characters can be calculated by taking total length of words divide by number of characters in a line.
```Python
# line 156 in tesseract_ocr.py
def get_one_line(cls, bbox_list, word_list, line_idx):
    bbox = cls.stack_bbox(bbox_list)
    text = ' '.join(word_list)
    char_height = round(sum([bb[3]-bb[1] for bb in bbox_list])/ len(bbox_list), 2)
    char_width_list = []
    char_list = []
    char_x0_list = []
    for word_idx, one_word in enumerate(word_list):
        char_ave_width = round((bbox_list[word_idx][2]-bbox_list[word_idx][0])/len(one_word), 2)
        char_width_list.append(char_ave_width)
        if word_idx != 0:
            char_x0_list.append(bbox_list[word_idx-1][2])
            char_list.append(' ')
        char_list.extend([char for char in one_word])
        for i in range(len(one_word)):
            char_x0_list.append(bbox_list[word_idx][0]+i*char_ave_width)
    char_width = round(sum(char_width_list)/len(word_list), 2)

    line = {'bbox': bbox,
            'char_height': char_height,
            'char_width': char_width,
            'chars': {'text': char_list,
                      'x0_list': char_x0_list},
            'line_id': line_idx,
            'text': text
            }
    return [line]
```

### Question 9
```Python
# line 95 in convert_scanned_pdf.py
if __name__ == "__main__":
    dataset_dir = pathlib.Path(__file__).parents[1]
    input_path = 'assignment_1_1/data/normal.jpg'
    file_path = dataset_dir / input_path
    save_path = 'assignment_1_1/data/normal'
    test_file = dataset_dir / f'{save_path}.json'
    
    result_pdf = image_to_pdf(file_path)
    doc = fitz.Document(stream=result_pdf, filetype='pdf')
    pdf_path = dataset_dir / 'assignment_1_1/data/normal.pdf'
    doc.save(pdf_path, no_new_id=True)
    pdf_bin = pdf_path.read_bytes()
    
    result = convert_scanned_pdf(pdf_bin)
    with open(test_file, 'w') as f:
        json.dump(result, f, indent=2)
```

### Question 10
```Python
# line 25 in convert_scanned_pdf.py
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
                elif orientation == 2:
                    # Mirrored left to right
                    pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    # Rotated 180 degrees
                    pil_image = pil_image.rotate(180)
                elif orientation == 4:
                    # Mirrored top to bottom
                    pil_image = pil_image.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    # Mirrored along top-left diagonal
                    pil_image = pil_image.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    # Rotated 90 degrees
                    pil_image = pil_image.rotate(-90, expand=True)
                elif orientation == 7:
                    # Mirrored along top-right diagonal
                    pil_image = pil_image.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    # Rotated 270 degrees
                    pil_image = pil_image.rotate(90, expand=True)
                break
```