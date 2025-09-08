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
import sys
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '3'

import torch
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

from configs import (
    InferenceEmbeddingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from dataset.random_inference_dataset import RandomInferenceDataGenerator
from dataset.utils import FeatureConfig
from modules.gpu_memory_usage import print_gpu_memory_usage

sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR

# Set cuda device
torch.cuda.set_device(1)
device_id = torch.cuda.current_device()
print(f"Current device: cuda:{torch.cuda.current_device()}")

log_dir = "logs"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/inference_benchmark_{current_time}.log"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.propagate = False

file_handler = logging.FileHandler(log_file, mode='a', encoding=None, delay=False)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference benchmark with variable batch size")
    parser.add_argument('--batch_size', type=int, default=8, choices=[1, 2, 4, 5, 6, 7, 8, 10, 12, 14, 16],
                        help='Batch size for inference')
    parser.add_argument('--use_kvcache', action='store_true', default=False,
                        help='Whether to use kv_cache')
    parser.add_argument('--use_cudagraph', action='store_true', default=False,
                        help='Whether to use cuDagraph')
    parser.add_argument('--full_mode', action='store_true', default=False,
                        help='Whether to run in full mode')

    args = parser.parse_args()
    return args

def run_ranking_gr_inference(inference_batch_size=8, _use_kvcache=True, _use_cudagraph=False, _full_mode=True):
    # print("Current working directory:", os.getcwd())
    max_batch_size = 16
    # max_seqlen = 4096
    max_seqlen = 10240
    max_num_candidates = 256
    max_incremental_seqlen = 256

    # context_emb_size = 1000
    item_fea_name, item_vocab_size = "item_feat", 10000
    action_fea_name, action_vocab_size = "act_feat", 128
    feature_configs = [
        FeatureConfig(
            feature_names=[item_fea_name, action_fea_name],
            max_item_ids=[item_vocab_size - 1, action_vocab_size - 1],
            max_sequence_length=max_seqlen,
            is_jagged=False,
        ),
    ]
    max_contextual_seqlen = 0
    total_max_seqlen = sum(
        [fc.max_sequence_length * len(fc.feature_names) for fc in feature_configs]
    )

    hidden_dim_size = 1024
    num_heads = 4
    head_dim = 256
    num_layers = 8
    inference_dtype = torch.bfloat16
    hstu_cudagraph_configs = {
        "batch_size": [1, 2, 4, 8],
        "length_per_sequence": [i * 256 for i in range(2, 18)],
    }

    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=inference_dtype,
    )

    # _blocks_in_primary_pool = 40960
    _blocks_in_primary_pool = 20480
    _page_size = 32
    _offload_chunksize = 8192
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=_blocks_in_primary_pool,
        page_size=_page_size,
        offload_chunksize=_offload_chunksize,
        max_batch_size=max_batch_size,
        max_seq_len=total_max_seqlen,
    )
    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["context_feat", "item_feat"]
            if max_contextual_seqlen > 0
            else ["item_feat"],
            table_name="item",
            vocab_size=item_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=True,
        ),
    ]
    num_tasks = 3
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=[[128, 10, 1] for _ in range(num_tasks)],
    )

    logger.info("Configurations:")
    logger.info(f"max_batch_size: {max_batch_size}, max_seqlen: {max_seqlen}, max_num_candidates: {max_num_candidates}")
    logger.info(f"item_fea_name: {item_fea_name}, action_fea_name: {action_fea_name}")
    logger.info(f"hidden_dim_size: {hidden_dim_size}, num_heads: {num_heads}, num_layers: {num_layers}")
    logger.info(f"hstu_config: {hstu_config}, kv_cache_config: {kv_cache_config}")

    # print("before inference model:", end=' ')
    # print_gpu_memory_usage(0)
    # logger.info("before inference model:")
    # print_gpu_memory_usage(logger, device_id)

    # torch.cuda.memory._record_memory_history(max_entries=100000)
    with torch.inference_mode():
        model_predict = InferenceRankingGR(
            hstu_config=hstu_config,
            kvcache_config=kv_cache_config,
            task_config=task_config,
            logger=logger,
            use_cudagraph=_use_cudagraph,
            cudagraph_configs=hstu_cudagraph_configs,
        )
        model_predict.bfloat16()
        model_predict.eval()
        # print("after inference model:", end=' ')
        # print_gpu_memory_usage(0)
        # logger.info("after inference model:")
        # print_gpu_memory_usage(logger, device_id)

        data_generator = RandomInferenceDataGenerator(
            feature_configs=feature_configs,
            item_feature_name=item_fea_name,
            contextual_feature_names=[],
            action_feature_name=action_fea_name,
            max_num_users=1024,
            max_batch_size=inference_batch_size,  # test batch size
            max_seqlen=8448,
            max_num_candidates=max_num_candidates,
            max_incremental_seqlen=max_incremental_seqlen,
            full_mode=_full_mode,
        )

        num_warmup_batches = 32
        for idx in range(num_warmup_batches):
            uids = data_generator.get_inference_batch_user_ids()
            # print(f'idx:{idx}, uids:{uids}')
            # print_gpu_memory_usage(0)
            # logger.info(f'idx:{idx}, uids:{uids}')
            # print_gpu_memory_usage(logger, device_id)
            logger.info(f'Warmup idx:{idx}, uids:{uids}')


            if uids is None:
                break

            cached_start_pos, cached_len = model_predict.get_user_kvdata_info(uids)
            truncate_start_pos = cached_start_pos + cached_len
            batch = data_generator.get_random_inference_batch(uids, truncate_start_pos)

            model_predict.forward(batch, uids, truncate_start_pos)

            # try:
            #     torch.cuda.memory._dump_snapshot(f"memory.pickle")
            # except Exception as e:
            #     logger.error(f"Failed to capture memory snapshot {e}")

            # # Stop recording memory snapshot history.
            # torch.cuda.memory._record_memory_history(enabled=None)
            # sys.exit(0)

        num_test_batches = 8192
        ts_start, ts_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        predict_time = 0.0
        for idx in range(num_test_batches):
            uids = data_generator.get_inference_batch_user_ids()
            # print(f'idx:{idx}, uids:{uids}')
            # print_gpu_memory_usage(0)
            logger.info(f'idx:{idx}, uids:{uids}')
            # print_gpu_memory_usage(logger, device_id)

            if uids is None:
                break

            cached_start_pos, cached_len = model_predict.get_user_kvdata_info(uids)
            truncate_start_pos = cached_start_pos + cached_len
            batch = data_generator.get_random_inference_batch(uids, truncate_start_pos)

            torch.cuda.synchronize()
            ts_start.record()
            model_predict.forward(batch, uids, truncate_start_pos)
            ts_end.record()
            torch.cuda.synchronize()
            predict_time += ts_start.elapsed_time(ts_end)
        # print("Total time(ms):", predict_time)
        logger.info(f"Total time(ms): {predict_time}")


if __name__ == "__main__":
    # run_ranking_gr_inference()
    args = parse_args()
    run_ranking_gr_inference(
        inference_batch_size=args.batch_size,
        _use_kvcache=args.use_kvcache,
        _use_cudagraph=args.use_cudagraph,
        _full_mode=args.full_mode,
    )
