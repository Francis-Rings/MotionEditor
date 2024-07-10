import argparse
import os
import imageio
import numpy as np
import cv2


def find_character_bbox(mask):
    def binary_search(start, end, check_fn):
        while start < end:
            mid = (start + end) // 2
            if check_fn(mid):
                end = mid
            else:
                start = mid + 1
        return start

    def check_row(row):
        return any(mask[row])

    def check_column(col):
        return any(mask[:,col])

    top_row = binary_search(0, len(mask), check_row)
    bottom_row = binary_search(top_row, len(mask), lambda row: not check_row(row))

    left_col = binary_search(0, len(mask[0]), check_column)
    right_col = binary_search(left_col, len(mask[0]), lambda col: not check_column(col))

    width = right_col - left_col
    height = bottom_row - top_row
    top_left = (left_col, top_row)

    return width, height, top_left


def find_person_boundaries(mask):
    left, right, top, bottom = float('inf'), 0, float('inf'), 0

    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == 1:
                left = min(left, j)
                right = max(right, j)
                top = min(top, i)
                bottom = max(bottom, i)

    return left, right, top, bottom


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Alignment Demo", add_help=True)
    # parser.add_argument("-d", "--data", type=str, required=True, help="path to image file")
    parser.add_argument("--source_mask_path", type=str, required=True, help="source mask path")
    parser.add_argument("--target_mask_path", type=str, required=True, help="target_mask_path")
    parser.add_argument("--source_pose_path", type=str, required=True, help="source_pose_path")
    parser.add_argument("--target_pose_path", type=str, required=True, help="target_pose_path")
    parser.add_argument("--save_path", type=str, required=True, help="save path")
    args = parser.parse_args()

    tmp_point = None
    num_frames = 24

    for i in range(num_frames):
        source_mask_path = os.path.join(args.source_mask_path, f"frame_{i}.png")
        target_mask_path = os.path.join(args.target_mask_path, f"frame_{i}.png")
        source_mask = imageio.imread(source_mask_path).astype(np.float32)
        source_mask = source_mask / 255
        source_left, source_right, source_top, source_bottom = find_person_boundaries(source_mask)
        source_point = (source_left, source_bottom)

        if i % num_frames == 0:
            tmp_point = None
        if tmp_point is not None:
            source_point = tmp_point
        else:
            source_point = ((source_right + source_left) / 2, source_bottom)
            tmp_point = source_point

        print("===================source coordination==================")
        print(f"id: {i}")
        print(source_left)
        print(source_right)
        print(source_top)
        print(source_bottom)
        print("======================================================")
        target_mask = imageio.imread(target_mask_path).astype(np.float32)
        target_mask = target_mask / 255
        target_left, target_right, target_top, target_bottom = find_person_boundaries(target_mask)

        target_point = ((target_left + target_right) / 2, target_bottom)
        print("===================target coordination==================")
        print(target_left)
        print(target_right)
        print(target_top)
        print(target_bottom)
        print("======================================================")

        source_pose_path = os.path.join(args.source_pose_path, f"frame_{i}.png")
        target_pose_path = os.path.join(args.target_pose_path, f"frame_{i}.png")
        save_path = os.path.join(args.save_path, f"frame_{i}.png")

        image1 = cv2.imread(source_pose_path)
        image2 = cv2.imread(target_pose_path)
        image1 = cv2.resize(image1, (512, 512))
        image2 = cv2.resize(image2, (512, 512))
        x1, y1, w1, h1 = source_left, source_top, source_right - source_left, source_bottom - source_top
        person1_roi = image1[y1:y1 + h1, x1:x1 + w1]
        x2, y2, w2, h2 = target_left, target_top, target_right - target_left, target_bottom - target_top
        person2_roi = image2[y2:y2 + h2, x2:x2 + w2]

        ratio = w2 / float(h2)
        h_new = None
        w_new = round(ratio * h1)
        if h_new is not None:
            w_new = round(ratio * h_new)
        if x1 - (w_new - w1) < 0 and x1 + w_new > 512:
            w_new = round(w_new * 0.85)
        if h_new is not None:
            person2_resized = cv2.resize(person2_roi, (w_new, h_new))
        else:
            person2_resized = cv2.resize(person2_roi, (w_new, h1))
        aligned_image2 = np.zeros_like(image1)
        if w_new > w1:
            if x1 - (w_new - w1) > 0:
                if h_new is not None:
                    aligned_image2[y1 + (h1 - h_new):y1 + h1,
                    x1 - (w_new - w1):x1 - (w_new - w1) + w_new] = person2_resized
                    aligned_point = ((x1 - (w_new - w1) + x1 - (w_new - w1) + w_new) / 2, h1 + y1)
                else:
                    aligned_image2[y1:y1 + h1, x1 - (w_new - w1):x1 - (w_new - w1) + w_new] = person2_resized
                    aligned_point = ((x1 - (w_new - w1) + x1 - (w_new - w1) + w_new) / 2, h1 + y1)
            else:
                if h_new is not None:
                    aligned_image2[y1 + (h1 - h_new):y1 + h1, x1:x1 + w_new] = person2_resized
                    aligned_point = ((x1 + x1 + w_new) / 2, h1 + y1)
                else:
                    aligned_image2[y1:y1 + h1, x1:x1 + w_new] = person2_resized
                    aligned_point = ((x1 + x1 + w_new) / 2, h1 + y1)
        else:
            if h_new is not None:
                aligned_image2[y1 + (h1 - h_new):y1 + h1, x1:x1 + w_new] = person2_resized
                aligned_point = ((x1 + x1 + w_new) / 2, h1 + y1)
            else:
                aligned_image2[y1:y1 + h1, x1:x1 + w_new] = person2_resized
                aligned_point = ((x1 + x1 + w_new) / 2, h1 + y1)

        # dx = source_point[0] - aligned_point[0]
        # dy = source_point[1] - aligned_point[1]

        # dx = reference_point[0] - aligned_point[0]
        # dy = reference_point[1] - aligned_point[1]

        dx = target_point[0] - aligned_point[0]
        dy = target_point[1] - aligned_point[1]

        rows, cols, _ = image1.shape
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_image2 = cv2.warpAffine(aligned_image2, translation_matrix, (cols, rows))
        cv2.imwrite(save_path, aligned_image2)