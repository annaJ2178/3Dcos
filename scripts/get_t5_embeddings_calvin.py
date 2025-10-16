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

import argparse
import os
import pickle

import numpy as np

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR

"""example command
python -m scripts.get_t5_embeddings --dataset_path datasets/hdvila
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, default="/nas/jiangyuxin/task_D_D", help="Root path to the dataset") # 512093
    parser.add_argument(
        "--max_length",
        type=int,
        help="Maximum length of the text embedding",
    )
    parser.add_argument("--cache_dir", type=str, default="/nas/jiangyuxin/huggingface/cache/models--google-t5--t5-11b/snapshots/90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3", help="Directory to cache the T5 model")
    return parser.parse_args()


def main(args) -> None:
    
    # breakpoint()
    metas_dir = args.dataset_path
    episode_dir = os.path.join(metas_dir, "training")    

    # annotations = np.load(f"{args.path}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
    # annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
    ann_list = os.path.join(episode_dir, "lang_annotations/auto_lang_ann.npy")
    # "lang_BERT/auto_lang_ann.npy"
    annotations = np.load(ann_list, allow_pickle=True).item()
    ann_dict = dict(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
    print(f"length of the dataset is {len(ann_dict)}")
    breakpoint()
    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)

    # Initialize T5
    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    for k, prompt in ann_dict.items():

        # Compute T5 embeddings
        encoded_text, mask_bool = encoder.encode_prompts(
            prompt, max_length=args.max_length, return_mask=True
        )  # list of np.ndarray in (len, embed_dim)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)

        # trim zeros to save space
        encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]
        t5_xxl_filename = os.path.join(t5_xxl_dir, f'0{k[0]}_0{k[1]}.pt')
        if os.path.exists(t5_xxl_filename):
            # Skip if the file already exists
            print("exist")
        # Save T5 embeddings as pickle file
        # breakpoint()
        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)
    print("finish finish finish finish finish finish")


if __name__ == "__main__":
    args = parse_args()
    main(args)
