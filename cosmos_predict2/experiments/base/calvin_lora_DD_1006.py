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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_calvin import (
    CalvinDataset,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


example_video_dataset = L(CalvinDataset)(
    dataset_dir="/nas/jiangyuxin/task_D_D",
    # dataset_dir="/home/jiangyuxin/data/calvin_debug_dataset",
    num_frames=93,
    video_size=(704, 1280),
    fps=2,
    whichset="training"
)

# example_val_video_dataset = L(CalvinDataset)(
#     # dataset_dir="/nas/jiangyuxin/task_D_D",
#     dataset_dir="/home/jiangyuxin/data/calvin_debug_dataset",
#     num_frames=93,
#     video_size=(704, 1280),
#     fps=2,
#     whichset="validation"
# )

# Create DataLoader with distributed sampler
dataloader_train_gr1 = L(get_generic_dataloader)(
    dataset=example_video_dataset,
    sampler=L(get_sampler)(dataset=example_video_dataset),
    batch_size=2,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# dataloader_val_gr1 = L(get_generic_dataloader)(
#     dataset=example_val_video_dataset,
#     sampler=L(get_sampler)(dataset=example_val_video_dataset),
#     batch_size=2,
#     drop_last=True,
#     num_workers=4,
#     pin_memory=True,
# )

# Video2World post-training configuration for 2B model
# torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_video2world_training_2b_groot_gr1_480
predict2_video2world_training_2b_calvin = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=dataloader_train_gr1,
    # dataloader_val=dataloader_val_gr1,
    checkpoint=dict(
        save_iter=1000,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_calvin_5000",
    ),
    optimizer=dict(
        lr=2 ** (-10.5),
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[0],
        # warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        run_validation=False,
        validation_iter=1,
        logging_iter=1,
        max_iter=100000,
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=100,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=1,
                fps=2,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=1000,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            tensorboard=dict(
                every_n=1
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    model=dict(
        config=dict(
            min_num_conditional_frames=1,
            max_num_conditional_frames=1,
            conditional_frames_probs={0: 0.1, 1: 0.9, 2:0.0},
            # loss_scale=10.0,
            # adjust_video_noise=False,
            # scaling="rectified_flow",

            fsdp_shard_size=1,
            resolution="720",
            state_t=2,
            # resize_online=True,
            # high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
            # sigma_data=1.0,
            # high_sigma_ratio=0.05,
            # rectified_flow_loss_weight_uniform=False,
            
            
            
            # net=dict(
            #     rope_enable_fps_modulation=False,
            #     rope_h_extrapolation_ratio=3.0,
            #     rope_w_extrapolation_ratio=3.0,
            #     rope_t_extrapolation_ratio=24.0 / 24,
            #     sac_config=dict(
            #         mode="predict2_2b_720_aggressive",
            #     ),
            #     use_crossattn_projection=True,
            #     crossattn_proj_in_channels=100352,
            #     crossattn_emb_channels=1024,
            # ),
            # conditioner=dict(
            #     use_video_condition=dict(
            #         dropout_rate=0.0,
            #     ),
            #     text=dict(
            #         dropout_rate=0.2,
            #         use_empty_string=False,
            #     ),
            # ),

            # tokenizer=dict(
            #     temporal_window=16,
            # ),
            # text_encoder_class="reason1p1_7B",
            
            
            
            # text_encoder_config=dict(
            #     embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
            #     compute_online=True,
            #     ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
            # ),
            
            # -----------------------lora config-----------------------------
            use_lora=True,
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
        )
    ),

)

cs = ConfigStore.instance()

for _item in [
    predict2_video2world_training_2b_calvin,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
