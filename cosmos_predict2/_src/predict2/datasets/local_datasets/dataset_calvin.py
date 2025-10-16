# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import traceback
import warnings
import random
from typing import Any
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log

"""
Test the dataset with the following command:
python -m cosmos_predict2.data.dataset_video
"""

# @ https://github.com/liufanfanlff/RoboUniview/blob/main/robouniview/data/real_dataset_hdf5.py#L202
class PatchMask(nn.Module):
    def __init__(self, patch_size=16, mask_ratio=0.35):
        super(PatchMask, self).__init__()
        self.patch_size=patch_size
        self.mask_ratio=mask_ratio
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Generate random mask coordinates.
        mask_coords = []
        for i in range(batch_size):
            for j in range(0, height, self.patch_size):
                for k in range(0, width, self.patch_size):
                    if random.random() < self.mask_ratio:
                        mask_coords.append((i, j, k))

        # Mask out the patches.
        masked_x = x.clone()
        for i, j, k in mask_coords:
            masked_x[i, :, j:j + self.patch_size, k:k + self.patch_size] = 0.0
        
        return masked_x



class CalvinDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        num_frames=16,
        video_size=(192, 256),
        fps=2,
        final_frames=5,
        whichset="training",
        task_info_root = '',
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = ['head'],
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=False,
        dataset_info_cache_path = None,
        action_type = "absolute",
        action_space = "joint",
        fix_epiidx = None,
        fix_sidx = None,
        fix_mem_idx = None,
    ) -> None:
        """Dataset class for loading frames from NPZ files.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames
            frames_key (str): Key name for frames array in NPZ file (default: "frames")

        NPZ file should contain:
            - frames: numpy array of shape [T, H, W, C] with uint8 values (0-255)
            
        Returns dict with:
            - video: RGB frames tensor [C,T,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.final_frames = final_frames
        # frames_dir = dataset_dir
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")

        self.whichset = whichset
        ann_list = os.path.join(self.dataset_dir, f"{whichset}/lang_annotations/auto_lang_ann.npy")
        annotations = np.load(ann_list, allow_pickle=True).item()
        ann_dict = dict(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
        self.episode_idx = annotations["info"]["indx"] # [(s, e), (s, e), ...] len == len of episodes
        self.json_file = os.path.join(dataset_dir, whichset, "episodes_mapping.json")

        restart = False
        self.episode_paths = {}
        if not self.load_json() or restart:
            self.build_mapping()
            # self.create_episode_path_mapping_large_scale()
            with open(self.json_file, 'w') as f:
                json.dump({str(k): v for k, v in self.episode_paths.items()}, f, indent=1)
                print(f"Saved to {self.json_file}")
           
        log.info(f"Found {len(self.episode_idx)} episodes with ?? total NPZ files")
        repeat = True
        if repeat is True:
            self.episode_idx = self.episode_idx * 100
            # breakpoint()
            log.info(f"Now Found {len(self.episode_idx)} episodes with ?? total NPZ files")

        self.wrong_number = 0

        self.clip_mean = (0.485, 0.456, 0.406)
        self.clip_std = (0.229, 0.224, 0.225)
        
        self.preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple(video_size)),
            # T.Normalize(self.clip_mean, self.clip_std)
            ])
        
        use_color_jitter = False
        if use_color_jitter:
            self.static_rgb_preprocess_train = T.Compose([
                T.ColorJitter(
                    brightness=0.05,
                    # contrast=0.05,
                    # hue=0.02
                )
            ])



        self.frames_key = 'rgb_static'
        self.depth_key = 'depth_static'
        # self.statistics = [
        #     os.path.join(self.episode_dirs, d)
        #     for d in os.listdir(self.dataset_dir)
        #     if d == "statistics.ymal"
        # ]
        self.use_data_augmentation = False
        self.fps=fps

    def load_json(self):
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                data= json.load(f)
            # Convert keys back to tuples
            for k, v in data.items():
                self.episode_paths[eval(k)] = v
            print(f"Loaded {len(self.episode_paths)} episodes from JSON")
            return True
        return False
    
    def build_mapping(self):
        # Get all batch folders
        training_dir = os.path.join(self.dataset_dir, self.whichset)
        batches = [f for f in os.listdir(training_dir) 
                  if f.startswith("batch") and os.path.isdir(os.path.join(training_dir, f))]
        # batches.sort(key=lambda x: int(x.split('_')[1]))
        
        # Build episode to path mapping
        ep_to_path = {}
        for batch in batches:
            batch_path = os.path.join(training_dir, batch)
            for file in os.listdir(batch_path):
                if file.endswith(".npz") and file.startswith("episode_"):
                    try:
                        ep_num = int(file.split("_")[1].split(".")[0])
                        ep_to_path[ep_num] = os.path.join(batch_path, file)
                    except:
                        continue
        
        # Create episode range mapping

        ep_nums = np.array(list(ep_to_path.keys()))
        ep_paths = list(ep_to_path.values())
        
        for start, end in self.episode_idx:
            episode_files = []
            # Check each episode number in sequence
            for ep_num in range(start, end + 1):
                if ep_num in ep_to_path:
                    episode_files.append(ep_to_path[ep_num])
                else:
                    break  # Stop if any episode is missing
            
            # Only add if we have the complete sequence
            if len(episode_files) == end-start+1:
                self.episode_paths[(start, end)] = episode_files
        
        print(f"Built mapping for {len(self.episode_paths)} episodes")

    def temporal_sample(self, video: torch.Tensor, expected_length: int, original_length, new_range) -> torch.Tensor:
        # sample consecutive video frames to match expected_length
        # original_length = video.shape[2]
        # video in [B C T H W] format
        start_frame = np.random.randint(0, original_length - expected_length)
        end_frame = start_frame + expected_length
        video = video[start_frame:end_frame, :, :, :]
        new_range = new_range[start_frame:end_frame]
        return video, new_range


    def pad_sample(self, video: torch.Tensor, expected_length: int, original_length: int, new_range, pad_last=True) -> torch.Tensor:
        # sample consecutive video frames to match expected_length
        # original_length = video.shape[2]
        # if original_length < expected_length:
        # video in [B C T H W] format


        padding_length = expected_length - original_length
        _, channels, height, width = video.shape
        
        if pad_last is True:
            # padding = torch.cat([video[-1,:,:,:]] * padding_length, dim=0)
            padding = video[-1:, :, :, :].repeat(padding_length, 1, 1, 1)  # Shape: [padding_length, C, H, W]
        else:
            padding = torch.zeros(
                padding_length, channels, height, width,
                dtype=video.dtype, device=video.device
            )
        video = torch.cat([video, padding], dim=0)
        new_range.extend([new_range[-1]] * padding_length)
        return video, new_range # t, c, h, w

    def __str__(self) -> str:
        return f"{len(self.episode_idx)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.episode_idx)

    def _load_frames(self, npz_paths: list[str], index_range) -> tuple[np.ndarray, float]:
        """Load frames from a list of NPZ files (all frames from one camera).
        Optimized: samples start index first, then only loads needed frames.
        
        Args:
            npz_paths: List of sorted paths to NPZ files for one camera (1 frame per file)
            
        Returns:
            frames: numpy array [sequence_length, H, W, C]
            fps: frames per second (default 16)
        """
        # Count total frames (assumes 1 frame per NPZ file for efficiency)
        total_frames = len(npz_paths)
        
        # if total_frames < self.sequence_length:
        if total_frames < self.sequence_length:
            start_frame =  np.random.randint(0, total_frames // 2)
            end_frame = total_frames - 1
            # warnings.warn(  # noqa: B028
            #     f"Camera sequence has only {total_frames} frames, "
            #     f"at least {self.sequence_length} frames are required."
            # )
            # raise ValueError(f"Camera sequence has insufficient frames.")

        else:
            max_start_idx = total_frames - self.sequence_length
            start_frame = np.random.randint(0, max_start_idx + 1)
            end_frame = start_frame + self.sequence_length - 1
        
        # Only load the frames we need
        selected_frames = []
        selected_index = []
        step_size = 16 // self.fps

        for i in range(start_frame, end_frame+1, step_size):
            data = np.load(npz_paths[i])
            
            if self.frames_key not in data:
                raise ValueError(f"Key '{self.frames_key}' not found in {npz_paths[i]}. Available keys: {list(data.keys())}")
            
            frames = data[self.frames_key] # 200,200,3
            
            # Handle single frame case
            if len(frames.shape) == 3:  # [H, W, C]
                frames = frames[np.newaxis, ...]  # [1, H, W, C]
            selected_index.append(int(npz_paths[i].split('_')[-1].split('.')[0]))
            selected_frames.append(frames)
        
        # Stack selected frames
        frames = np.concatenate(selected_frames, axis=0)  # [sequence_length, H, W, C]
        if frames.shape[0] < self.final_frames:
            assert selected_index[-1] > index_range[1] - 16 // self.fps
            final_d = np.load(npz_paths[end_frame])[self.frames_key][np.newaxis, ...] # [1, H, W, C]
            frames = np.concatenate([frames, final_d], axis=0)
            selected_index.append(int(npz_paths[end_frame].split('_')[-1].split('.')[0]))
        return frames, selected_index
    

    def _get_frames(self, npz_paths: list[str], index_range: tuple) -> tuple[torch.Tensor, float]:
        """Load and preprocess frames from a list of NPZ files.
        
        Args:
            npz_paths: List of paths to NPZ files for one camera
            
        Returns:
            frames: torch tensor [T, C, H, W]
            fps: frames per second
        """
        frames, new_range = self._load_frames(npz_paths, index_range)
        
        # Ensure uint8
        frames = frames.astype(np.uint8)
        
        # Convert to torch and rearrange: [T, H, W, C] -> [T, C, H, W]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        # Apply preprocessing (resize, normalization, etc.)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
                
        return frames, new_range

    def __getitem__(self, index) -> dict | Any:
        # try:
            data = dict()
            episode_idx = self.episode_idx[index] # tuple
            path = self.episode_paths[episode_idx]
            videos = {}
            # fps_dict = {}
            video, new_range = self._get_frames(path, episode_idx)
            original_length = video.shape[0]
            # log.info(f"original_length is {original_length}")

            if original_length > self.final_frames:
                video, new_range = self.temporal_sample(video, self.final_frames, video.shape[0], new_range)
            elif original_length < self.final_frames:
                video, new_range = self.pad_sample(video, self.final_frames, video.shape[0], new_range)
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W] is tensor after norm
            videos[self.frames_key] = video
            # fps_dict[camera_name] = fps
            
            # Use first camera for metadata (or specify a primary camera)
            primary_camera = list(videos.keys())[0]
            primary_video = videos[primary_camera]
            # primary_fps = fps_dict[primary_camera]
            
            # Get T5 embedding path (you may need to adjust this based on your structure)
            # episode_name = os.path.basename(os.path.dirname(list(episode[primary_camera])[0]))
            t5_embedding_path = os.path.join(self.dataset_dir , "t5_xxl", f"0{episode_idx[0]}_0{episode_idx[1]}.pt")
            
            data["video"] = video
            data["video_name"] = {
                "video_path": torch.tensor(new_range).unsqueeze(dim=0),
                "t5_embedding_path": t5_embedding_path,
            }

            _, _, h, w = primary_video.shape

            # Load T5 embeddings [len_of_token, 1024]
            with open(t5_embedding_path, "rb") as f:
                t5_embedding_raw = pickle.load(f)
                assert isinstance(t5_embedding_raw, list)
                assert len(t5_embedding_raw) == 1
                t5_embedding = t5_embedding_raw[0]  # [n_tokens, CosmosTextEncoderConfig.EMBED_DIM]
                assert isinstance(t5_embedding, np.ndarray)
                assert len(t5_embedding.shape) == 2
            
            n_tokens = t5_embedding.shape[0]
            if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
                t5_embedding = np.concatenate(
                    [
                        t5_embedding,
                        np.zeros(
                            (CosmosTextEncoderConfig.NUM_TOKENS - n_tokens, CosmosTextEncoderConfig.EMBED_DIM),
                            dtype=np.float32,
                        ),
                    ],
                    axis=0,
                )
            t5_text_mask = torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS, dtype=torch.int64)
            t5_text_mask[:n_tokens] = 1

            data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = self.fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = video.shape[1]
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        # except Exception:
        #     warnings.warn(  # noqa: B028
        #         f"Invalid data encountered: {self.episode_idx[index]}. Skipped "
        #         f"(by randomly sampling another sample in the same dataset)."
        #     )
        #     warnings.warn("FULL TRACEBACK:")  # noqa: B028
        #     warnings.warn(traceback.format_exc())  # noqa: B028
        #     self.wrong_number += 1
        #     log.info(self.wrong_number, rank0_only=False)
        #     return self[np.random.randint(len(self))]  # Fixed: was self.samples, should be self


if __name__ == "__main__":
    dataset = CalvinDataset(
        dataset_dir="/home/jiangyuxin/data/calvin_debug_dataset",
        whichset="validation",
        num_frames=93,
        # video_size=[480, 832],
        # frames_key="frames",  # Specify key name in NPZ
    )
    import time

    # Sleep for 5 seconds
    time.sleep(10)
    indices = [0, 2, -1]
    for idx in indices:
        data = dataset[idx]
        log.info(
            f"{idx=} "
            f"{data['video'].sum()=}\n"
            f"{data['video'].shape=}\n"
            f"{data['video_name']=}\n"
            f"{data['t5_text_embeddings'].shape=}\n"
            "---"
        )