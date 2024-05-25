import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from einops import rearrange
import os.path as osp
from glob import glob
import imageio
import random
import cv2
import numpy as np


class VideoDataset(Dataset):
    def __init__(
            self,
            video_dir: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            # Update for motion editing
            condition: list[str] = 'openpose',  ## type of condition used
            video_suffix: str = '.jpg',
            condition_suffix: str = '.png',
            random_sample: bool = False,
            source_mask_dir: str = None,
            train_prompt: list[str] = 'openpose',
            **kwargs,
    ):
        self.video_dir = video_dir  ## path to the video dir
        self.video_path = osp.join(self.video_dir, 'images')
        self.condition = condition
        if isinstance(condition, str):
            condition = [condition]
        self.source_condition_path = {_condition: osp.join(self.video_dir, "source_condition", _condition) for _condition in condition}
        self.target_condition_path = {_condition: osp.join(self.video_dir, "target_condition", _condition) for _condition in condition}
        self.video_suffix = video_suffix
        self.condition_suffix = condition_suffix
        self.random_sample = random_sample

        self.source_mask_dir = source_mask_dir
        if source_mask_dir:
            self.source_mask_dir = osp.join(self.video_dir, source_mask_dir)

        frame_list_path = osp.join(self.video_dir, 'frame_list.txt')
        if not osp.isfile(frame_list_path):
            all_frames = sorted(glob(osp.join(self.video_path, '*')))
            self.frame_list = []
            with open(frame_list_path, 'w') as f:
                for _frame_path in all_frames:
                    _frame_name = osp.basename(_frame_path).split('.')[0]
                    self.frame_list.append(_frame_name)
                    f.write(_frame_name + '\n')
        else:
            with open(frame_list_path, 'r') as f:
                self.frame_list = f.read().splitlines()
        self.video_length = len(self.frame_list)
        self.prompt = prompt
        self.prompt_ids = None
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.source_img_embeddings = []
        self.train_prompt = train_prompt

        print('Training on Video {} \t totally {} frames'.format(self.video_dir.split('/')[-1], self.video_length))


    @torch.no_grad()
    def preprocess_img_embedding(self, feature_extractor, image_encoder):
        for f_name in self.frame_list:
            image = imageio.imread(osp.join(self.video_path, f_name + self.video_suffix))
            image = feature_extractor(images=image, return_tensors="pt").pixel_values
            image_embeds = image_encoder(image).image_embeds
            self.source_img_embeddings.append(image_embeds[0])  # 1,768 --> 768


    def __len__(self):
        return 1

    def __getitem__(self, index):
        video_indices = list(range(self.sample_start_idx, self.video_length, self.sample_frame_rate))
        video = []
        source_conditions = {_condition: [] for _condition in self.condition}
        target_conditions = {_condition: [] for _condition in self.condition}
        source_mask = []

        if self.random_sample:
            start_index = random.randint(0, len(video_indices) - self.n_sample_frames)  ## [a,b] include both
        else:
            start_index = 0
        sample_index = video_indices[start_index:start_index + self.n_sample_frames]

        for _f_idx in sample_index:
            _frame = imageio.imread(osp.join(self.video_path, self.frame_list[_f_idx] + self.video_suffix))
            if self.source_mask_dir:
                _source_mask = imageio.imread(osp.join(self.source_mask_dir, self.frame_list[_f_idx] + '.png')).astype(np.float32)  ## H,W 0 and 255
                _source_mask /= 255  # 0 and 1
            else:
                _source_mask = np.ones(_frame.shape[:2])

            video.append(_frame)
            source_mask.append(_source_mask)

            for _control_type, _control_path in self.source_condition_path.items():
                _source_condition = imageio.imread(
                    osp.join(_control_path, self.frame_list[_f_idx] + self.condition_suffix))  ##
                source_conditions[_control_type].append(_source_condition)
            for _control_type, _control_path in self.target_condition_path.items():
                _target_condition = imageio.imread(
                    osp.join(_control_path, self.frame_list[_f_idx] + self.condition_suffix))  ##
                target_conditions[_control_type].append(_target_condition)

        video = torch.from_numpy(np.stack(video, axis=0)).float()  # f,h,w,c
        video = rearrange(video, "f h w c -> f c h w")
        video = F.interpolate(video, size=(self.height, self.width), mode='bilinear')

        source_conditions_transform = {}
        for _control_type, condition in source_conditions.items():
            condition = torch.from_numpy(np.stack(condition, axis=0)).float()  # f,h,w,c
            condition = rearrange(condition, "f h w c -> f c h w")
            condition = F.interpolate(condition, size=(self.height, self.width), mode='bilinear')
            source_conditions_transform[_control_type] = condition / 255

        target_conditions_transform = {}
        for _control_type, condition in target_conditions.items():
            condition = torch.from_numpy(np.stack(condition, axis=0)).float()  # f,h,w,c
            condition = rearrange(condition, "f h w c -> f c h w")
            condition = F.interpolate(condition, size=(self.height, self.width), mode='bilinear')
            target_conditions_transform[_control_type] = condition / 255

        source_mask = torch.from_numpy(np.stack(source_mask, axis=0)).float()  # f,h,w
        source_mask = rearrange(source_mask[:, :, :, None], "f h w c -> f c h w")
        source_mask = F.interpolate(source_mask, size=(self.height, self.width), mode='nearest')

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "source_conditions": source_conditions_transform,
            "target_conditions": target_conditions_transform,
            "prompt_ids": self.prompt_ids,
            "source_masks": source_mask,
            "sample_indices": torch.LongTensor(sample_index),
            "prompt": self.prompt,
            "train_prompt": self.train_prompt,
        }

        return example
