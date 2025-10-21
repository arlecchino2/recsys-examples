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

import json
import os
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from dynamicemb.dynamicemb_config import (
    DynamicEmbTableOptions,
    create_dynamicemb_table,
    dyn_emb_to_torch,
    torch_to_dyn_emb,
)
from dynamicemb.initializer import BaseDynamicEmbInitializer
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizerV2
from dynamicemb.types import Cache, Storage
from dynamicemb_extensions import (
    DynamicEmbTable,
    EvictStrategy,
    clear,
    count_matched,
    dyn_emb_capacity,
    dyn_emb_cols,
    dyn_emb_rows,
    export_batch,
    export_batch_matched,
    find_pointers,
    insert_and_evict,
    insert_or_assign,
    load_from_pointers,
    select,
    select_index,
)

torch_dtype_to_np_dtype = {
    torch.uint64: np.uint64,
    torch.int64: np.int64,
    torch.float32: np.float32,
}

KEY_TYPE = torch.int64
EMBEDDING_TYPE = torch.float32
SCORE_TYPE = torch.int64
OPT_STATE_TYPE = torch.float32


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error saving data to JSON file: {e}")


def load_from_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data from JSON file: {e}")


def batched_export_keys_values(
    dynamic_table: DynamicEmbTable,
    device: torch.device,
    batch_size: int = 65536,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    export keys, embeddings, opt_states, scores
    """

    search_capacity = dyn_emb_capacity(dynamic_table)

    offset = 0

    while offset < search_capacity:
        key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
        value_dtype = dyn_emb_to_torch(dynamic_table.value_type())
        dim = dyn_emb_cols(dynamic_table)
        optstate_dim = dynamic_table.optstate_dim()
        total_dim = dim + optstate_dim

        cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        keys = torch.empty(batch_size, dtype=key_dtype, device=cuda_device)
        values = torch.empty(
            batch_size * total_dim, dtype=value_dtype, device=cuda_device
        )
        scores = torch.zeros(batch_size, dtype=SCORE_TYPE, device=cuda_device)
        d_counter = torch.zeros(1, dtype=torch.uint64, device=cuda_device)

        export_batch(dynamic_table, batch_size, offset, d_counter, keys, values, scores)

        values = values.reshape(batch_size, total_dim)

        embeddings = values[:, :dim].contiguous()
        opt_states = values[:, dim:].contiguous()

        d_counter = d_counter.to(dtype=torch.int64)
        actual_length = d_counter.item()
        if actual_length > 0:
            yield (
                keys[:actual_length].to(KEY_TYPE).to(device),
                embeddings[:actual_length, :].to(EMBEDDING_TYPE).to(device),
                opt_states[:actual_length, :].to(OPT_STATE_TYPE).to(device),
                scores[:actual_length].to(SCORE_TYPE).to(device),
            )
        offset += batch_size


def load_key_values(
    dynamic_table: DynamicEmbTable,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    opt_states: Optional[torch.Tensor] = None,
):
    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()
    if not keys.is_cuda:
        raise RuntimeError("Keys must be on GPU")
    if not embeddings.is_cuda:
        raise RuntimeError("Embeddings must be on GPU")
    if scores is not None and not scores.is_cuda:
        raise RuntimeError("Scores must be on GPU")
    if opt_states is not None and not opt_states.is_cuda:
        raise RuntimeError("Opt states must be on GPU")

    if opt_states is None and optstate_dim > 0:
        opt_states = (
            torch.ones(
                keys.numel(),
                optstate_dim,
                dtype=dyn_emb_to_torch(dynamic_table.value_type()),
                device=embeddings.device,
            )
            * dynamic_table.get_initial_optstate()
        )

    values = (
        torch.cat([embeddings.view(-1, dim), opt_states.view(-1, optstate_dim)], dim=-1)
        if opt_states is not None
        else embeddings
    )

    if dynamic_table.evict_strategy() == EvictStrategy.KLru:
        if scores is not None:
            raise RuntimeError("Scores are not supported for KLru evict strategy")
    else:
        if scores is None:
            raise RuntimeError("Scores are required for non-KLru evict strategy")

    key_type = dyn_emb_to_torch(dynamic_table.key_type())
    value_type = dyn_emb_to_torch(dynamic_table.value_type())
    if scores is not None:
        insert_or_assign(
            dynamic_table,
            keys.numel(),
            keys.to(key_type),
            values.to(value_type),
            scores.to(SCORE_TYPE),
        )
    else:
        insert_or_assign(
            dynamic_table, keys.numel(), keys.to(key_type), values.to(value_type)
        )


class KeyValueTable(
    Cache, Storage[DynamicEmbTableOptions, BaseDynamicEmbeddingOptimizerV2]
):
    def __init__(
        self,
        options: DynamicEmbTableOptions,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        self.options = options
        self.table = create_dynamicemb_table(options)
        self._capacity = options.max_capacity
        self.optimizer = optimizer
        self.score: int = None
        self._score_update = False
        self._emb_dim = self.options.dim
        self._emb_dtype = self.options.embedding_dtype
        self._de_emb_dtype = torch_to_dyn_emb(self._emb_dtype)
        self._value_dim = self._emb_dim + optimizer.get_state_dim(self._emb_dim)
        self._initial_optim_state = optimizer.get_initial_optim_states()

        device_idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{device_idx}")
        props = torch.cuda.get_device_properties(device_idx)
        self._threads_in_wave = (
            props.multi_processor_count * props.max_threads_per_multi_processor
        )

        self._cache_metrics = torch.zeros(10, dtype=torch.long, device="cpu")
        self._record_cache_metrics = False
        self._use_score = self.table.evict_strategy() != EvictStrategy.KLru

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if unique_keys.dtype != self.key_type():
            unique_keys = unique_keys.to(self.key_type())

        if unique_embs.dtype != self.value_type():
            raise RuntimeError(
                "Embedding dtype not match {} != {}".format(
                    unique_embs.dtype, self.value_type()
                )
            )

        batch = unique_keys.size(0)
        assert unique_embs.dim() == 2
        assert unique_embs.size(0) == batch

        load_dim = unique_embs.size(1)

        device = unique_keys.device
        if founds is None:
            founds = torch.empty(batch, dtype=torch.bool, device=device)
        pointers = torch.empty(batch, dtype=torch.long, device=device)

        if self._score_update:
            # TODO: support score
            find_pointers(self.table, batch, unique_keys, pointers, founds, self.score)
        else:
            find_pointers(self.table, batch, unique_keys, pointers, founds)

        self.value_dim()

        if load_dim != 0:
            load_from_pointers(pointers, unique_embs)

        missing = torch.logical_not(founds)
        num_missing_0: torch.Tensor = torch.empty(1, dtype=torch.long, device=device)
        num_missing_1: torch.Tensor = torch.empty(1, dtype=torch.long, device=device)
        missing_keys: torch.Tensor = torch.empty_like(unique_keys)
        missing_indices: torch.Tensor = torch.empty(
            batch, dtype=torch.long, device=device
        )
        select(missing, unique_keys, missing_keys, num_missing_0)
        select_index(missing, missing_indices, num_missing_1)

        if self._record_cache_metrics:
            self._cache_metrics[0] = batch
            self._cache_metrics[1] = founds.sum().item()

        h_num_missing = num_missing_0.cpu().item()
        return (
            h_num_missing,
            missing_keys[:h_num_missing],
            missing_indices[:h_num_missing],
        )

    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return self.find_impl(unique_keys, unique_embs, founds)

    def find_missed_keys(
        self,
        unique_keys: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        # dummy tensor
        unique_embs = torch.empty(
            unique_keys.numel(), 0, device=unique_keys.device, dtype=self._emb_dtype
        )
        return self.find_impl(unique_keys, unique_embs, founds)

    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return self.find_impl(unique_keys, unique_vals, founds)

    def insert(
        self,
        unique_keys: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        h_num_unique_keys = unique_keys.numel()
        if self._use_score:
            if scores is None:
                scores = torch.empty(
                    h_num_unique_keys, device=unique_keys.device, dtype=torch.uint64
                )
                scores.fill_(self.score)
        else:
            scores = None

        insert_or_assign(
            self.table,
            h_num_unique_keys,
            unique_keys,
            unique_values.to(self.value_type()),
            scores,
        )

    def update(
        self,
        keys: torch.Tensor,
        grads: torch.Tensor,
        return_missing: bool = True,
    ) -> Tuple[Optional[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self._score_update == False, "update is called only in backward."

        batch = keys.size(0)

        device = keys.device
        founds = torch.empty(batch, dtype=torch.bool, device=device)
        pointers = torch.empty(batch, dtype=torch.long, device=device)
        find_pointers(self.table, batch, keys, pointers, founds)

        self.optimizer.fused_update_with_pointer(
            grads.to(self.value_type()), pointers, self._de_emb_dtype
        )

        if return_missing:
            missing = torch.logical_not(founds)
            num_missing_0: torch.Tensor = torch.empty(
                1, dtype=torch.long, device=device
            )
            num_missing_1: torch.Tensor = torch.empty(
                1, dtype=torch.long, device=device
            )
            missing_keys: torch.Tensor = torch.empty_like(keys)
            missing_indices: torch.Tensor = torch.empty(
                batch, dtype=torch.long, device=device
            )
            select(missing, keys, missing_keys, num_missing_0)
            select_index(missing, missing_indices, num_missing_1)
            h_num_missing = num_missing_0.cpu().item()
            return (
                h_num_missing,
                missing_keys[:h_num_missing],
                missing_indices[:h_num_missing],
            )
        return None, None, None

    def enable_update(self) -> bool:
        return True

    def set_score(
        self,
        score: int,
    ) -> None:
        self.score = score

    @property
    def score_update(
        self,
    ) -> None:
        return self._score_update

    @score_update.setter
    def score_update(self, value: bool):
        self._score_update = value

    def dump(
        self,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool,
        include_meta: bool,
    ) -> None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if include_meta:
            meta_data = {}
            meta_data.update(self.optimizer.get_opt_args())
            save_to_json(meta_data, meta_json_file_path)

        fkey = open(emb_key_path, "wb")
        fembedding = open(embedding_file_path, "wb")
        fscore = open(score_file_path, "wb") if self._use_score else None
        fopt_states = open(opt_file_path, "wb") if include_optim else None

        for keys, embeddings, opt_states, scores in batched_export_keys_values(
            self.table, device
        ):
            fkey.write(keys.cpu().numpy().tobytes())
            fembedding.write(embeddings.cpu().numpy().tobytes())
            if fopt_states:
                fopt_states.write(opt_states.cpu().numpy().tobytes())
            if fscore:
                fscore.write(scores.cpu().numpy().tobytes())
        fkey.close()
        fembedding.close()

        if fscore:
            fscore.close()

        if fopt_states:
            fopt_states.close()

        return

    def load(
        self,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool,
    ) -> None:
        meta_data = load_from_json(meta_json_file_path)
        opt_type = meta_data.get(
            "opt_type", None
        )  # for compatibility with old format, which doesn't have opt_type
        if opt_type and self.optimizer.get_opt_args().get("opt_type", None) != opt_type:
            include_optim = False
            print(
                f"Optimizer type mismatch: {opt_type} != {self.optimizer.get_opt_args().get('opt_type')}. Will not load optimizer states."
            )

        if not opt_file_path or not os.path.exists(opt_file_path):
            include_optim = False
            print(
                f"Optimizer file {opt_file_path} does not exist. Will not load optimizer states."
            )

        if self._use_score:
            assert score_file_path and os.path.exists(
                score_file_path
            ), f"Score file {score_file_path} not found."

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        dim = dyn_emb_cols(self.table)
        optstate_dim = self.table.optstate_dim()

        if optstate_dim == 0:
            include_optim = False

        if include_optim:
            self.optimizer.set_opt_args(meta_data)

        fkey = open(emb_key_path, "rb")
        fembedding = open(embedding_file_path, "rb")
        fscore = open(score_file_path, "rb") if self._use_score else None
        fopt_states = open(opt_file_path, "rb") if include_optim else None

        num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize

        num_embeddings = (
            os.path.getsize(embedding_file_path) // EMBEDDING_TYPE.itemsize // dim
        )

        if num_keys != num_embeddings:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of embeddings in {embedding_file_path}."
            )

        if fscore:
            num_scores = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
            if num_keys != num_scores:
                raise ValueError(
                    f"The number of keys in {emb_key_path} does not match with number of scores in {score_file_path}."
                )

        if fopt_states:
            num_opt_states = (
                os.path.getsize(opt_file_path)
                // OPT_STATE_TYPE.itemsize
                // optstate_dim
            )
            if num_keys != num_opt_states:
                raise ValueError(
                    f"The number of keys in {emb_key_path} does not match with number of opt_states in {opt_file_path}."
                )

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        batch_size = 65536
        for start in range(0, num_keys, batch_size):
            num_keys_to_read = min(num_keys - start, batch_size)
            keys_bytes = fkey.read(KEY_TYPE.itemsize * num_keys_to_read)

            embedding_bytes = fembedding.read(
                EMBEDDING_TYPE.itemsize * dim * num_keys_to_read
            )
            embeddings = torch.tensor(
                np.frombuffer(
                    embedding_bytes, dtype=torch_dtype_to_np_dtype[EMBEDDING_TYPE]
                ),
                dtype=EMBEDDING_TYPE,
                device=device,
            ).view(-1, dim)

            opt_states = None
            if fopt_states:
                opt_state_bytes = fopt_states.read(
                    OPT_STATE_TYPE.itemsize * optstate_dim * num_keys_to_read
                )
                opt_states = torch.tensor(
                    np.frombuffer(
                        opt_state_bytes, dtype=torch_dtype_to_np_dtype[OPT_STATE_TYPE]
                    ),
                    dtype=OPT_STATE_TYPE,
                    device=device,
                ).view(-1, optstate_dim)

            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )

            scores = None
            if fscore:
                score_bytes = fscore.read(SCORE_TYPE.itemsize * num_keys_to_read)
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )

            masks = keys % world_size == rank
            keys = keys[masks]
            embeddings = embeddings[masks, :]
            if scores is not None:
                scores = scores[masks]
            if opt_states is not None:
                opt_states = opt_states[masks, :]
            load_key_values(self.table, keys, embeddings, scores, opt_states)

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def embedding_dtype(
        self,
    ) -> torch.dtype:
        return self._emb_dtype

    def value_dim(
        self,
    ) -> int:
        return self._value_dim

    def embedding_dim(
        self,
    ) -> int:
        return self._emb_dim

    def init_optimizer_state(
        self,
    ) -> float:
        return self._initial_optim_state

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = keys.numel()
        num_evicted: torch.Tensor = torch.zeros(1, dtype=torch.long, device=keys.device)
        evicted_keys: torch.Tensor = torch.empty_like(keys)
        evicted_values: torch.Tensor = torch.empty_like(values)
        evicted_scores: torch.Tensor = torch.empty(
            batch, dtype=torch.uint64, device=keys.device
        )
        insert_and_evict(
            self.table,
            batch,
            keys,
            values,
            self.score if self._use_score else None,
            evicted_keys,
            evicted_values,
            evicted_scores,
            num_evicted,
        )
        if self._record_cache_metrics:
            self._cache_metrics[2] = batch
            self._cache_metrics[3] = num_evicted.cpu().item()
        h_num_evict = num_evicted.cpu().item()
        return (
            h_num_evict,
            evicted_keys[:h_num_evict],
            evicted_values[:h_num_evict, :],
            evicted_scores[:h_num_evict],
        )

    def flush(self, storage: Storage) -> None:
        batch_size = self._threads_in_wave
        for keys, embeddings, opt_states, scores in batched_export_keys_values(
            self.table, self.device, batch_size
        ):
            if keys.numel() != 0:
                values = torch.cat((embeddings, opt_states), dim=1).contiguous()
                storage.insert(keys, values, scores)

    def reset(
        self,
    ) -> None:
        clear(self.table)

    @property
    def cache_metrics(self) -> Optional[torch.Tensor]:
        return self._cache_metrics if self._record_cache_metrics else None

    def set_record_cache_metrics(self, record: bool) -> None:
        self._record_cache_metrics = record
        return

    def count_matched(
        self,
        threshold: int,
        num_matched: torch.Tensor,
    ) -> None:
        count_matched(self.table, threshold, num_matched)

    def key_type(
        self,
    ) -> torch.dtype:
        return dyn_emb_to_torch(self.table.key_type())

    def value_type(
        self,
    ) -> torch.dtype:
        return dyn_emb_to_torch(self.table.value_type())

    def capacity(
        self,
    ) -> int:
        return self._capacity

    def export_batch_matched(
        self, threshold, batch_size, search_offset, d_count, d_keys, d_vals
    ) -> None:
        export_batch_matched(
            self.table,
            threshold,
            batch_size,
            search_offset,
            d_count,
            d_keys,
            d_vals,
        )

    def evict_strategy(self) -> EvictStrategy:
        return self.table.evict_strategy()

    def optim_state_dim(self) -> int:
        return self.value_dim() - self.embedding_dim()

    def size(self) -> int:
        return dyn_emb_rows(self.table)


def update_cache(
    cache: Cache,
    storage: Storage,
    missing_keys: torch.Tensor,
    missing_values: torch.Tensor,
):
    # need to update score.
    num_evicted, evicted_keys, evicted_values, evicted_scores = cache.insert_and_evict(
        missing_keys, missing_values
    )
    if num_evicted != 0:
        storage.insert(
            evicted_keys,
            evicted_values,
            evicted_scores,
        )


class KeyValueTableFunction:
    @staticmethod
    def lookup(
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        initializer: Callable,
        training: bool,
    ) -> None:
        assert unique_keys.dim() == 1
        h_num_toatl = unique_keys.numel()
        emb_dim = storage.embedding_dim()
        emb_dtype = storage.embedding_dtype()
        val_dim = storage.value_dim()

        if h_num_toatl == 0:
            return

        # 1. find in storage
        founds = torch.empty(h_num_toatl, device=unique_keys.device, dtype=torch.bool)
        (
            h_num_missing_in_storage,
            missing_keys_in_storage,
            missing_indices_in_storage,
        ) = storage.find_embeddings(unique_keys, unique_embs, founds=founds)

        # 2. initialize missing embeddings
        if h_num_missing_in_storage != 0:
            initializer(
                unique_embs,
                missing_indices_in_storage,
                unique_keys,
            )
        else:
            return

        # 3. insert missing values into table.
        if training:
            # insert missing values
            missing_values_in_storage = torch.empty(
                h_num_missing_in_storage,
                val_dim,
                device=unique_keys.device,
                dtype=emb_dtype,
            )
            missing_values_in_storage[:, :emb_dim] = unique_embs[
                missing_indices_in_storage, :
            ]
            if val_dim != emb_dim:
                missing_values_in_storage[
                    :, emb_dim - val_dim :
                ] = storage.init_optimizer_state()
            storage.insert(missing_keys_in_storage, missing_values_in_storage)
        # ignore the storage missed in eval mode

    @staticmethod
    def update(
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_grads: torch.Tensor,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        if storage.enable_update():
            storage.update(unique_keys, unique_grads, return_missing=False)
            return

        emb_dtype = storage.embedding_dtype()
        val_dim = storage.value_dim()
        h_num_toatl = unique_keys.numel()
        unique_values = torch.empty(
            h_num_toatl, val_dim, device=unique_keys.device, dtype=emb_dtype
        )
        founds = torch.empty(h_num_toatl, device=unique_keys.device, dtype=torch.bool)
        _, _, _ = storage.find(unique_keys, unique_values, founds=founds)

        keys_for_storage = unique_keys[founds].contiguous()
        values_for_storage = unique_values[founds, :].contiguous()
        grads_for_storage = unique_grads[founds, :].contiguous()
        optimizer.fused_update(
            grads_for_storage,
            values_for_storage,
        )

        storage.insert(keys_for_storage, values_for_storage)

        return


class KeyValueTableCachingFunction:
    @staticmethod
    def lookup(
        cache: Cache,  # partial emb + optimizer state
        storage: Storage,  # full emb + optimizer state
        unique_keys: torch.Tensor,  # input
        unique_embs: torch.Tensor,  # output
        initializer: Callable,
        enable_prefetch: bool,
        training: bool,
    ) -> None:
        assert unique_keys.dim() == 1
        unique_keys.numel()
        emb_dim = storage.embedding_dim()
        emb_dtype = storage.embedding_dtype()
        val_dim = (
            storage.value_dim()
        )  # value is generally composed of embedding and optimizer state

        # 1. find in cache
        h_num_keys_for_storage, missing_keys, missing_indices = cache.find_embeddings(
            unique_keys, unique_embs
        )
        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys

        founds = torch.empty(
            h_num_keys_for_storage, device=unique_keys.device, dtype=torch.bool
        )

        # 2. find in storage
        values_for_storage = torch.empty(
            h_num_keys_for_storage,
            val_dim,
            device=unique_keys.device,
            dtype=emb_dtype,
        )
        (
            h_num_missing_in_storage,
            missing_keys_in_storage,
            missing_indices_in_storage,
        ) = storage.find(keys_for_storage, values_for_storage, founds=founds)

        # 3. initialize missing embeddings
        if h_num_missing_in_storage != 0:
            initializer(
                values_for_storage[:, :emb_dim],
                missing_indices_in_storage,
                keys_for_storage,
            )

        # 4. copy embeddings to unique_embs
        unique_embs[missing_indices, :] = values_for_storage[:, :emb_dim]

        if training:
            if emb_dim != val_dim:
                values_for_storage[
                    missing_indices_in_storage, emb_dim - val_dim :
                ] = storage.init_optimizer_state()
            update_cache(cache, storage, keys_for_storage, values_for_storage)
        else:  # only update those found in the storage to cache.
            found_keys_in_storage = keys_for_storage[founds].contiguous()
            found_values_in_storage = values_for_storage[founds, :].contiguous()
            update_cache(cache, storage, found_keys_in_storage, found_values_in_storage)
        return

    @staticmethod
    def update(
        cache: Cache,
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_grads: torch.Tensor,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        h_num_keys_for_storage, missing_keys, missing_indices = cache.update(
            unique_keys, unique_grads
        )
        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys
        grads_for_storage = unique_grads[missing_indices, :].contiguous()

        if storage.enable_update():
            storage.update(keys_for_storage, grads_for_storage, return_missing=False)
            return

        emb_dtype = storage.embedding_dtype()
        val_dim = storage.value_dim()
        values_for_storage = torch.empty(
            h_num_keys_for_storage, val_dim, device=unique_keys.device, dtype=emb_dtype
        )
        founds = torch.empty(
            h_num_keys_for_storage, device=unique_keys.device, dtype=torch.bool
        )
        _, _, _ = storage.find(keys_for_storage, values_for_storage, founds=founds)

        keys_for_storage = keys_for_storage[founds].contiguous()
        values_for_storage = values_for_storage[founds, :].contiguous()
        grads_for_storage = grads_for_storage[founds, :].contiguous()
        optimizer.fused_update(
            grads_for_storage,
            values_for_storage,
        )

        storage.insert(keys_for_storage, values_for_storage)

        return

    @staticmethod
    def prefetch(
        cache: Cache,
        storage: Storage,
        unique_keys: torch.Tensor,
        initializer: BaseDynamicEmbInitializer,
        training: bool = True,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        assert cache is not None, "prefetch is available only when caching is enabled."
        emb_dtype = storage.embedding_dtype()
        h_num_keys_for_storage, missing_keys, _ = cache.find_missed_keys(unique_keys)

        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys

        val_dim = storage.value_dim()
        emb_dim = storage.embedding_dim()
        values_for_storage = torch.empty(
            h_num_keys_for_storage, val_dim, device=unique_keys.device, dtype=emb_dtype
        )
        founds = torch.empty(
            h_num_keys_for_storage, device=unique_keys.device, dtype=torch.bool
        )
        (
            num_missing_in_storage,
            missing_keys_in_storage,
            missing_indices_in_storage,
        ) = storage.find(keys_for_storage, values_for_storage, founds=founds)

        if num_missing_in_storage != 0:
            if training:
                embs_for_storage = values_for_storage[:, :emb_dim]
                initializer(
                    embs_for_storage,
                    missing_indices_in_storage,
                    keys_for_storage,
                )
                if val_dim != emb_dim:
                    values_for_storage[
                        missing_indices_in_storage, emb_dim - val_dim :
                    ] = storage.init_optimizer_state()
            else:
                keys_for_storage = keys_for_storage[founds].contiguous()
                values_for_storage = values_for_storage[founds, :].contiguous()

        update_cache(
            cache,
            storage,
            keys_for_storage,
            values_for_storage,
        )
