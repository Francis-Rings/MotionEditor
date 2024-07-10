import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)
import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundedSAM.GroundingDINO.groundingdino.datasets.transforms as T
from GroundedSAM.GroundingDINO.groundingdino.models import build_model
from GroundedSAM.GroundingDINO.groundingdino.util import box_ops
from GroundedSAM.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundedSAM.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from GroundedSAM.segment_anything.segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import ipdb
import imageio
from tqdm import tqdm



def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    # width, height = image_pil.size
    # image_pil = image_pil.resize((width * 2, height * 2))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases, logits_filt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    # parser.add_argument("-d", "--data", type=str, required=True, help="path to image file")
    parser.add_argument("-t", "--text_prompt", type=str, required=True, help="text prompt")

    parser.add_argument(
        '-d',
        '--data',
        type=str,
        help='dir for images: data/dir/images',
        default=None,
        required=True,
    )

    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=False, help="output directory"
    )

    parser.add_argument("--config", type=str,
                        default="/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                        help="path to config file")

    parser.add_argument(
        "--grounded_checkpoint", type=str, default="checkpoints/groundingdino_swinb_cogcoor.pth",
        help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="checkpoints/sam_vit_h_4b8939.pth", help="path to checkpoint file"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")

    parser.add_argument("--masked_out", action='store_true', help="save the masked image")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    text_prompt = args.text_prompt

    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make dir
    text_prompt_dir = "-".join(text_prompt.split(" "))

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    subfolder_path = args.data
    images_subfolder_path = os.path.join(subfolder_path, "images")
    print(f"subfolder path: {subfolder_path}")
    print(f"images subfolder path: {images_subfolder_path}")
    mask_subfolder_path = os.path.join(subfolder_path, "masks")
    if not os.path.exists(mask_subfolder_path):
        os.makedirs(mask_subfolder_path)
        print(f"Folder created: {mask_subfolder_path}")
    else:
        print(f"Folder already exists: {mask_subfolder_path}")
    for root, dirs, files in os.walk(images_subfolder_path):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                print(file_path)
                file_name = os.path.splitext(file)[0]
                image_name = file_name + '.jpg'
                image_legal_path = os.path.join(images_subfolder_path, image_name)
                if os.path.exists(os.path.join(mask_subfolder_path, file_name + '.png')):
                    existed_path = os.path.join(mask_subfolder_path, file_name + '.png')
                    print(f"{existed_path} already exists!")
                    continue
                image_pil, image = load_image(image_legal_path)
                boxes_filt, pred_phrases, logits_filt = get_grounding_output(
                    model, image, text_prompt, box_threshold, text_threshold, device=device
                )
                try:
                    image = cv2.imread(image_legal_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Error occurs during the image loading:{image_legal_path}")
                    continue
                predictor.set_image(image)
                size = image_pil.size
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]
                boxes_filt = boxes_filt.cpu()
                print('bbox shape', boxes_filt.shape)
                if boxes_filt.size()[0] == 0:
                    print("The grounding dino model fails to detect the people in the image")
                    continue
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(device),
                    multimask_output=False,
                )
                max_logit_index = logits_filt.max(-1)[0].argmax().item()
                _mask = masks[max_logit_index, 0].cpu().numpy().astype(np.uint8) * 255
                mask_save_path = os.path.join(mask_subfolder_path, file_name + '.png')
                imageio.imwrite(mask_save_path, _mask)
                print(f"Finish Mask Extraction: {mask_save_path}")


