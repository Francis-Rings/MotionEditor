# MotionEditor [CVPR2024]

This repository is the official implementation of **MotionEditor: Editing Video Motion via Content-Aware Diffusion**.

**[MotionEditor: Editing Video Motion via Content-Aware Diffusion](https://arxiv.org/abs/2311.18830)**
<br/>
Shuyuan Tu, [Qi Dai](https://scholar.google.com/citations?user=NSJY12IAAAAJ), [Zhi-Qi Cheng](https://scholar.google.com/citations?user=uB2He2UAAAAJ), [Han Hu](https://ancientmooner.github.io/), [Xintong Han](https://xthan.github.io/), [Zuxuan Wu](https://zxwu.azurewebsites.net/), [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=zh-CN)
<br/>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://francis-rings.github.io/MotionEditor/) [![arXiv](https://img.shields.io/badge/arXiv-2311.18830-b31b1b.svg)](https://arxiv.org/abs/2311.18830)

<p align="center">
<img src="./assets/figures/overview.jpg" width="1080px"/>  
<br>
<em>MotionEditor: A diffusion-based video editing method aimed at transferring motion from a reference to a source.</em>
</p>

# News
- :star2: **[July, 2024]** The codes of the data preparation are available to the public.
- :star2: **[February, 2024]** MotionEditor has been accepted by CVPR2024.

## Abstract
> Existing diffusion-based video editing models have made gorgeous advances for editing attributes of a source video over time but struggle to manipulate the motion information while preserving the original protagonist's appearance and background. To address this, we propose MotionEditor, the first diffusion model for video motion editing. MotionEditor incorporates a novel content-aware motion adapter into ControlNet to capture temporal motion correspondence.
While ControlNet enables direct generation based on skeleton poses, it encounters challenges when modifying the source motion in the inverted noise due to contradictory signals between the noise (source) and the condition (reference). Our adapter complements ControlNet by involving source content to transfer adapted control signals seamlessly. Further, we build up a two-branch architecture (a reconstruction branch and an editing branch) with a high-fidelity attention injection mechanism facilitating branch interaction. This mechanism enables the editing branch to query the key and value from the reconstruction branch in a decoupled manner, making the editing branch retain the original background and protagonist appearance. We also propose a skeleton alignment algorithm to address the discrepancies in pose size and position. Experiments demonstrate the promising motion editing ability of MotionEditor, both qualitatively and quantitatively. To the best of our knowledge, MotionEditor is the first diffusion-based model capable of video motion editing.

## Setup

### Requirements

```shell
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).

### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)).

**[ControlNet]** [ControlNet](https://arxiv.org/abs/2302.05543) is a conditioned text-to-image diffusion model capable of generating conditioned contents. The pre-trained ControlNet models can be downloaded from Hugging Face (e.g., [sd-controlnet-openpose](https://huggingface.co/lllyasviel/sd-controlnet-openpose)). 

**[GroundingDINO]** [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO): wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

**[Segment Anything]** [Segment Anything](https://github.com/facebookresearch/segment-anything): wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


## Usage

### Data Preparation
We firstly have to build the GroundedSAM environment.
```bash
cd data_preparation/GroundedSAM
pip install -e GroundingDINO
pip install -e segment_anything
```
For extracting the skeletons of the given frames:
```bash
cd data_preparation
python video_skeletons.py --which_cond openposefull --data /path/frames
```
It is worth noting that the structure of the given data file should be the same as the structure of data/case-1 in the github.

For extracting the masks of the given frames:
```bash
cd data_preparation
python video_masks.py --text_prompt human --data /path/frames
```
--text_prompt indicates the prompt description of the target protagonist, such as boy, girl, and human.

For aligning the source frames with target frames:
```bash
cd data_preparation
python alignment.py --text_prompt human --source_mask_path /path/frames --target_mask_path /path/frames --source_pose_path /path/frames --target_pose_path /path/frames --save_path /path/frames
```
--source_mask_path, --target_mask_path, --source_pose_path, --target_pose_path refer to the source mask file directory, target mask file directory, source pose file directory, target pose file directory, respectively.
--save_path indicates the saved file path of aligned frames.

### Training

To fine-tune the text-to-image diffusion models for background reconstruction and fine-tune the motion adaptor for motion controlling, run this command:

```bash
python train_bg.py --config="configs/case-1/train-bg.yaml"
python train_adaptor.py --config="configs/case-1/train-motion.yaml"
```
Note: The number of training steps is depend on the particular case.

### Inference

Once the training is done, run inference:

```bash
accelerate launch inference.py --config configs/case-1/eval-motion.yaml
```
Note: The null-text optimization is optional as it may increase the inference latency and computational cost. The target skeleton should be explicitly aligned with the source protagonist.
The triggering step of attention injection mechanism can be modified for editing the motions of the particular case.

## Contact
If you have any suggestions or find our work helpful, feel free to contact us

Email: francisshuyuan@gmail.com

If you find our work useful, please consider citing it:

```
@inproceedings{tu2024motioneditor,
  title={Motioneditor: Editing video motion via content-aware diffusion},
  author={Tu, Shuyuan and Dai, Qi and Cheng, Zhi-Qi and Hu, Han and Han, Xintong and Wu, Zuxuan and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7882--7891},
  year={2024}
}
```
