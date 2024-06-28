import pydash
import numpy as np
import copy


def arrange_rows(paragraphs, connector='\t'):
    # flatten line obj from paragraph
    line_obj_list = []
    for paragraph in paragraphs:
        for line in paragraph["lines"]:
            line["y_central"] = (line["bbox"][1] + line["bbox"][3]) / 2
            line_obj_list.append(line)
    # sort line obj by y central
    sorted_line_obj_list = sorted(line_obj_list, key=lambda x: x["y_central"], reverse=False)
    # build line list for each row
    row_lines_matrix = []
    current_row = []
    for line_obj in sorted_line_obj_list:
        if pydash.get(line_obj, ['text']):
            line_obj['text'] = line_obj['text'].replace('|', ' ')
            line_obj['chars']['text'] = [char.replace('|', ' ') for char in line_obj['chars']['text']]
        if len(current_row) == 0:
            current_row.append(line_obj)
            continue
        else:
            row_y_central = np.mean([x["y_central"] for x in current_row])
            y_align_flag = abs(line_obj["y_central"] - row_y_central) <= 7
            if y_align_flag:
                x_no_overlap_flag = True
                for line_in_row in current_row:
                    if line_in_row["bbox"][0] <= line_obj["bbox"][0] <= line_in_row["bbox"][2] or \
                            line_in_row["bbox"][0] <= line_obj["bbox"][2] <= line_in_row["bbox"][2]:
                        x_no_overlap_flag = False
                        break
                if x_no_overlap_flag:
                    current_row.append(line_obj)
                    continue
            else:
                current_row = sorted(current_row, key=lambda x: x['bbox'][0])
                row_lines_matrix.append(current_row)
                current_row = [line_obj]
    if len(current_row) > 0:
        row_lines_matrix.append(current_row)

    # merge lines in row
    row_data = []
    all_space_width_list = []
    space_width = 7
    for lines in row_lines_matrix:
        merge_char_list = []
        merge_x0_list = []
        merge_x1_list = []
        merge_y0 = 0
        merge_y1 = 0
        merge_y_count = 0
        sorted_lines = sorted(lines, key=lambda x: x["bbox"][0])
        for index, line_obj in enumerate(sorted_lines):
            line_y0 = line_obj["bbox"][1]
            line_y1 = line_obj["bbox"][3]
            char_list = line_obj["chars"]["text"]
            line_count = len(char_list)
            merge_y0 += line_y0 * line_count
            merge_y1 += line_y1 * line_count
            merge_y_count += line_count
            x_list = copy.deepcopy(line_obj["chars"]["x0_list"])
            x_list.append(line_obj["bbox"][2])
            x0_list = x_list[:-1]
            x1_list = x_list[1:]
            if len(merge_char_list) > 0:
                # padding between lines with fixed space width
                previous_x_bound = merge_x1_list[-1]
                next_x_bound = line_obj["bbox"][0]
                if connector == ' ' and next_x_bound > previous_x_bound:
                    line_dist = next_x_bound - previous_x_bound
                    merge_char_list.append(' ' * int(round(line_dist / space_width)))
                else:
                    merge_char_list.append("\t")
                merge_x0_list.append(min(previous_x_bound, next_x_bound))
                merge_x1_list.append(max(previous_x_bound, next_x_bound))
            merge_char_list.extend(char_list)
            # remove extra x0 and x1
            merge_x0_list.extend(x0_list[:len(char_list)])
            merge_x1_list.extend(x1_list[:len(char_list)])

        merge_y_count = max(1, merge_y_count)
        merge_y0 /= merge_y_count
        merge_y1 /= merge_y_count
        merge_x0 = np.min(merge_x0_list)
        merge_x1 = np.max(merge_x1_list)

        row_obj = {
            "char_list": merge_char_list,
            "x0_list": merge_x0_list,
            "x1_list": merge_x1_list,
            "text": "".join(merge_char_list),
            "bbox": [int(merge_x0), int(merge_y0), int(merge_x1), int(merge_y1)]
        }
        row_data.append(row_obj)
    return row_data, row_lines_matrix, all_space_width_list


def create_document_string(page, prefix='### Input:\n\n', connector='\t', simple_join=False):
    """
    Takes in a page from idp-doc.
    chunk the document in two ways:
     1.Returns a string with the lines of text, padded with spaces as necessary.
     2.Returns a string by simply join all the paragraphs together.
    """
    document = prefix
    if not simple_join:  # TODO: cons and pros of handling information at page level?
        row_data, _, space_width = arrange_rows(page['paragraphs'], connector=connector)
        page_edge_x0 = min([row['bbox'][0] for row in row_data])
        avg_space_width = 7
        for line in row_data:
            text = line['text']
            bbox = line['bbox']
            x1, y1, x2, y2 = bbox
            if x1 > page_edge_x0:  # if the line doesn't start from the left edge of the page, pad with spaces
                if connector == ' ':
                    padding = connector * int(round(x1 - page_edge_x0) / avg_space_width)
                    text = padding + text
                elif int(round(x1 - page_edge_x0) / avg_space_width) >= 8:
                    text = connector + text
            document += text + '\n'
        document += '\n'

    else:
        # TODO: add new chunk method to organize the idp-doc paragraph, simply join all the paragraph texts.
        # 1. sort with bbox
        # 2. text join
        pass

    return document


