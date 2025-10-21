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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from dataset.utils import Batch, RankingBatch
from torch.utils.data.dataset import IterableDataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def load_seq(x: str):
    if isinstance(x, str):
        y = json.loads(x)
    else:
        y = x
    return y


def maybe_truncate_seq(
    y: List[int],
    max_seq_len: int,
) -> List[int]:
    y_len = len(y)
    if y_len > max_seq_len:
        y = y[:max_seq_len]
    return y


class InferenceDataset(IterableDataset[Batch]):
    """
    SequenceDataset is an iterable dataset designed for distributed recommendation systems.
    It handles loading, shuffling, and batching of sequence data for training models.

    Args:
        seq_logs_file (str): Path to the sequence logs file.
        batch_size (int): The batch size.
        max_seqlen (int): The maximum sequence length.
        item_feature_name (str): The name of the item feature.
        contextual_feature_names (list[str], optional): List of contextual feature names. Defaults to [].
        action_feature_name (str, optional): The name of the action feature. Defaults to None.
        max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        shuffle (bool): Whether to shuffle the data.
        random_seed (int): The random seed for shuffling.
        is_train_dataset (bool): Whether this dataset is for training.
        nrows (int, optional): The number of rows to read from the file. Defaults to None, meaning all rows are read.
    """

    def __init__(
        self,
        seq_logs_file: str,
        batch_logs_file: str,
        batch_size: int,
        max_seqlen: int,
        item_feature_name: str,
        contextual_feature_names: List[str],
        action_feature_name: str,
        max_num_candidates: int = 0,
        *,
        item_vocab_size: int,
        userid_name: str,
        date_name: str,
        sequence_endptr_name: str,
        timestamp_names: List[str],
        random_seed: int = 0,
        seq_nrows: Optional[int] = None,
        batch_nrows: Optional[int] = None,
        max_num_users: int = 1024,           # 最大用户数
        max_incremental_seqlen: int = 64,    # 最大增量序列长度
        full_mode: bool = False,             # 全批次模式标志
    ) -> None:
        super().__init__()
        self._device = torch.cuda.current_device()

        if seq_nrows is None or batch_nrows is None:
            self._seq_logs_frame = pd.read_csv(seq_logs_file, delimiter=",")
            self._batch_logs_frame = pd.read_csv(batch_logs_file, delimiter=",")
        else:
            self._seq_logs_frame = pd.read_csv(
                seq_logs_file, delimiter=",", nrows=seq_nrows
            )
            self._batch_logs_frame = pd.read_csv(
                batch_logs_file, delimiter=",", nrows=batch_nrows
            )

        self._batch_logs_frame.sort_values(by=timestamp_names, inplace=True)
        len(self._batch_logs_frame)

        self._num_samples = len(self._batch_logs_frame)
        self._max_seqlen = max_seqlen

        self._batch_size = batch_size
        self._random_seed = random_seed

        self._contextual_feature_names = contextual_feature_names
        if self._max_seqlen <= len(self._contextual_feature_names):
            raise ValueError(
                f"max_seqlen is too small. should > {len(self._contextual_feature_names)}"
            )
        self._item_feature_name = item_feature_name
        self._action_feature_name = action_feature_name
        self._max_num_candidates = max_num_candidates
        self._item_vocab_size = item_vocab_size
        self._userid_name = userid_name
        self._date_name = date_name
        self._seq_end_name = sequence_endptr_name

        self._sample_ids = np.arange(self._num_samples)

        self._max_num_users = min(max_num_users, 2**16)
        self._max_batch_size = batch_size
        self._max_hist_len = max_seqlen - max_num_candidates
        self._max_incr_fea_len = max(max_incremental_seqlen, 1)
        self._full_mode = full_mode
        
        self._item_history: Dict[int, torch.Tensor] = dict()
        self._action_history: Dict[int, torch.Tensor] = dict()

    # We do batching in our own
    def __len__(self) -> int:
        return math.ceil(self._num_samples / self._batch_size)

    def __iter__(self) -> Iterator[Batch]:
        for i in range(len(self)):
            batch_start = i * self._batch_size
            batch_end = min(
                (i + 1) * self._batch_size,
                len(self._sample_ids),
            )
            sample_ids = self._sample_ids[batch_start:batch_end]
            user_ids: List[int] = []
            dates: List[int] = []
            seq_endptrs: List[int] = []
            for sample_id in sample_ids:
                seq_endptr = self._batch_logs_frame.iloc[sample_id][self._seq_end_name]
                if seq_endptr > self._max_seqlen:
                    continue
                user_ids.append(
                    self._batch_logs_frame.iloc[sample_id][self._userid_name]
                )
                dates.append(self._batch_logs_frame.iloc[sample_id][self._date_name])
                seq_endptrs.append(seq_endptr)
            if len(user_ids) == 0:
                continue
            yield (
                torch.tensor(user_ids),
                torch.tensor(dates),
                torch.tensor(seq_endptrs),
            )

    # def get_input_batch(
    #     self,
    #     user_ids,
    #     dates,
    #     sequence_endptrs,
    #     sequence_startptrs,
    #     with_contextual_features=False,
    #     with_ranking_labels=False,
    # ):
    #     contextual_features: Dict[str, List[int]] = defaultdict(list)
    #     contextual_features_seqlen: Dict[str, List[int]] = defaultdict(list)
    #     item_features: List[int] = []
    #     item_features_seqlen: List[int] = []
    #     action_features: List[int] = []
    #     action_features_seqlen: List[int] = []
    #     num_candidates: List[int] = []
    #     labels: List[int] = []

    #     packed_user_ids: List[int] = []

    #     if len(user_ids) == 0:
    #         return None

    #     sequence_endptrs = torch.clip(sequence_endptrs, 0, self._max_seqlen)
    #     for idx in range(len(user_ids)):
    #         uid = user_ids[idx].item()
    #         date = dates[idx].item()
    #         end_pos = sequence_endptrs[idx].item()  # history_end_pos
    #         start_pos = sequence_startptrs[idx].item()

    #         data = self._seq_logs_frame[
    #             (self._seq_logs_frame[self._userid_name] == uid)
    #             & (self._seq_logs_frame[self._date_name] == date)
    #         ]
    #         data = data.iloc[0]
    #         if with_contextual_features:
    #             for contextual_feature_name in self._contextual_feature_names:
    #                 contextual_features[contextual_feature_name].append(
    #                     data[contextual_feature_name]
    #                 )
    #                 contextual_features_seqlen[contextual_feature_name].append(1)

    #         item_seq = load_seq(data[self._item_feature_name])[start_pos:end_pos]
    #         action_seq = load_seq(data[self._action_feature_name])[start_pos:end_pos]
    #         num_candidate = 0
    #         if self._max_num_candidates > 0:
    #             # randomly generated candidates
    #             if not with_ranking_labels:
    #                 # num_candidate = (torch.randint(self._max_num_candidates) + 1).item()
    #                 num_candidate = self._max_num_candidates
    #                 candidate_seq = torch.randint(
    #                     self._item_vocab_size, (num_candidate,)
    #                 ).tolist()

    #             # extract candidates from following sequences
    #             else:
    #                 all_seqs = self._seq_logs_frame[
    #                     (self._seq_logs_frame[self._userid_name] == uid)
    #                     & (self._seq_logs_frame[self._date_name] >= date)
    #                 ]
    #                 candidate_seq = sum(
    #                     [
    #                         load_seq(all_seqs.iloc[idx][self._item_feature_name])
    #                         for idx in range(len(all_seqs))
    #                     ],
    #                     start=[],
    #                 )[end_pos : end_pos + self._max_num_candidates]
    #                 num_candidate = len(candidate_seq)
    #                 label_seq = sum(
    #                     [
    #                         load_seq(all_seqs.iloc[idx][self._action_feature_name])
    #                         for idx in range(len(all_seqs))
    #                     ],
    #                     start=[],
    #                 )[end_pos : end_pos + self._max_num_candidates]

    #             all_item_seq = item_seq + candidate_seq

    #         item_features.extend(all_item_seq)
    #         item_features_seqlen.append(len(all_item_seq))
    #         num_candidates.append(num_candidate)
    #         if with_ranking_labels:
    #             labels.extend(label_seq)

    #         action_features.extend(action_seq)
    #         action_features_seqlen.append(len(action_seq))

    #         packed_user_ids.append(uid)

    #     if len(packed_user_ids) == 0:
    #         return None

    #     feature_to_max_seqlen = {}
    #     for name in self._contextual_feature_names:
    #         feature_to_max_seqlen[name] = max(
    #             contextual_features_seqlen[name], default=0
    #         )

    #     ### Currently use clipped maxlen. check how this impacts the hstu results
    #     feature_to_max_seqlen[self._item_feature_name] = max(item_features_seqlen)
    #     feature_to_max_seqlen[self._action_feature_name] = max(action_features_seqlen)

    #     if with_contextual_features:
    #         contextual_features_tensor = torch.tensor(
    #             [contextual_features[name] for name in self._contextual_feature_names],
    #         ).view(-1)
    #         contextual_features_lengths_tensor = torch.tensor(
    #             [
    #                 contextual_features_seqlen[name]
    #                 for name in self._contextual_feature_names
    #             ]
    #         ).view(-1)
    #     else:
    #         contextual_features_tensor = torch.empty((0,), dtype=torch.int64)
    #         contextual_features_lengths_tensor = torch.tensor(
    #             [0 for name in self._contextual_feature_names]
    #         ).view(-1)
    #     features = KeyedJaggedTensor.from_lengths_sync(
    #         keys=self._contextual_feature_names
    #         + [self._item_feature_name, self._action_feature_name],
    #         values=torch.concat(
    #             [
    #                 contextual_features_tensor.to(device=self._device),
    #                 torch.tensor(item_features, device=self._device),
    #                 torch.tensor(action_features, device=self._device),
    #             ]
    #         ).long(),
    #         lengths=torch.concat(
    #             [
    #                 contextual_features_lengths_tensor.to(device=self._device),
    #                 torch.tensor(item_features_seqlen, device=self._device),
    #                 torch.tensor(action_features_seqlen, device=self._device),
    #             ]
    #         ).long(),
    #     )
    #     labels = torch.tensor(labels, dtype=torch.int64, device=self._device)
    #     batch_kwargs = dict(
    #         features=features,
    #         batch_size=self._batch_size,
    #         feature_to_max_seqlen=feature_to_max_seqlen,
    #         contextual_feature_names=self._contextual_feature_names,
    #         item_feature_name=self._item_feature_name,
    #         action_feature_name=self._action_feature_name,
    #         max_num_candidates=self._max_num_candidates,
    #         num_candidates=torch.tensor(num_candidates, device=self._device)
    #         if self._max_num_candidates > 0
    #         else None,
    #     )
    #     if with_ranking_labels:
    #         return RankingBatch(labels=labels, **batch_kwargs)

    #     return Batch(**batch_kwargs)
    
    def get_input_batch(
        self,
        user_ids,
        dates,
        sequence_endptrs,
        sequence_startptrs,
        with_contextual_features=False,
        with_ranking_labels=False,
    ):
        contextual_features: Dict[str, List[int]] = defaultdict(list)
        contextual_features_seqlen: Dict[str, List[int]] = defaultdict(list)
        item_features: List[int] = []
        item_features_seqlen: List[int] = []
        action_features: List[int] = []
        action_features_seqlen: List[int] = []
        num_candidates: List[int] = []
        labels: List[int] = []

        packed_user_ids: List[int] = []

        if len(user_ids) == 0:
            return None
        
        if isinstance(with_contextual_features, bool):
            contextual_mask = [with_contextual_features] * len(user_ids)
        else:
            contextual_mask = with_contextual_features

        sequence_endptrs = torch.clip(sequence_endptrs, 0, self._max_seqlen)
        for idx in range(len(user_ids)):
            uid = user_ids[idx].item()
            date = dates[idx].item()
            end_pos = sequence_endptrs[idx].item()  # history_end_pos
            start_pos = sequence_startptrs[idx].item()

            data = self._seq_logs_frame[
                (self._seq_logs_frame[self._userid_name] == uid)
                & (self._seq_logs_frame[self._date_name] == date)
            ]
            data = data.iloc[0]
            # if with_contextual_features:
            #     for contextual_feature_name in self._contextual_feature_names:
            #         contextual_features[contextual_feature_name].append(
            #             data[contextual_feature_name]
            #         )
            #         contextual_features_seqlen[contextual_feature_name].append(1)
            if contextual_mask[idx]:
                for contextual_feature_name in self._contextual_feature_names:
                    contextual_features[contextual_feature_name].append(
                        data[contextual_feature_name]
                    )
                    contextual_features_seqlen[contextual_feature_name].append(1)
            else:
                for contextual_feature_name in self._contextual_feature_names:
                    contextual_features_seqlen[contextual_feature_name].append(0)

            item_seq = load_seq(data[self._item_feature_name])[start_pos:end_pos]
            action_seq = load_seq(data[self._action_feature_name])[start_pos:end_pos]
            num_candidate = 0
            if self._max_num_candidates > 0:
                # randomly generated candidates
                if not with_ranking_labels:
                    # num_candidate = (torch.randint(self._max_num_candidates) + 1).item()
                    num_candidate = self._max_num_candidates
                    candidate_seq = torch.randint(
                        self._item_vocab_size, (num_candidate,)
                    ).tolist()

                # extract candidates from following sequences
                else:
                    all_seqs = self._seq_logs_frame[
                        (self._seq_logs_frame[self._userid_name] == uid)
                        & (self._seq_logs_frame[self._date_name] >= date)
                    ]
                    candidate_seq = sum(
                        [
                            load_seq(all_seqs.iloc[idx][self._item_feature_name])
                            for idx in range(len(all_seqs))
                        ],
                        start=[],
                    )[end_pos : end_pos + self._max_num_candidates]
                    num_candidate = len(candidate_seq)
                    label_seq = sum(
                        [
                            load_seq(all_seqs.iloc[idx][self._action_feature_name])
                            for idx in range(len(all_seqs))
                        ],
                        start=[],
                    )[end_pos : end_pos + self._max_num_candidates]

                all_item_seq = item_seq + candidate_seq

            item_features.extend(all_item_seq)
            item_features_seqlen.append(len(all_item_seq))
            num_candidates.append(num_candidate)
            if with_ranking_labels:
                labels.extend(label_seq)

            action_features.extend(action_seq)
            action_features_seqlen.append(len(action_seq))

            packed_user_ids.append(uid)

        if len(packed_user_ids) == 0:
            return None

        feature_to_max_seqlen = {}
        for name in self._contextual_feature_names:
            feature_to_max_seqlen[name] = max(
                contextual_features_seqlen[name], default=0
            )

        ### Currently use clipped maxlen. check how this impacts the hstu results
        feature_to_max_seqlen[self._item_feature_name] = max(item_features_seqlen)
        feature_to_max_seqlen[self._action_feature_name] = max(action_features_seqlen)

        # if with_contextual_features:
        #     contextual_features_tensor = torch.tensor(
        #         [contextual_features[name] for name in self._contextual_feature_names],
        #     ).view(-1)
        #     contextual_features_lengths_tensor = torch.tensor(
        #         [
        #             contextual_features_seqlen[name]
        #             for name in self._contextual_feature_names
        #         ]
        #     ).view(-1)
        # else:
        #     contextual_features_tensor = torch.empty((0,), dtype=torch.int64)
        #     # contextual_features_lengths_tensor = torch.tensor(
        #     #     [0 for name in self._contextual_feature_names]
        #     # ).view(-1)
        #     # Create zero lengths for each contextual feature per batch sample to maintain tensor shape consistency
        #     contextual_features_lengths_tensor = torch.tensor(
        #         [0 for _ in range(len(user_ids) * len(self._contextual_feature_names))]
        #     ).view(-1)
        #     # features = KeyedJaggedTensor.from_lengths_sync(
        #     #     keys=[self._item_feature_name, self._action_feature_name],
        #     #     values=torch.concat(
        #     #         [
        #     #             torch.tensor(item_features, device=self._device),
        #     #             torch.tensor(action_features, device=self._device),
        #     #         ]
        #     #     ).long(),
        #     #     lengths=torch.concat(
        #     #         [
        #     #             torch.tensor(item_features_seqlen, device=self._device),
        #     #             torch.tensor(action_features_seqlen, device=self._device),
        #     #         ]
        #     #     ).long(),
        #     # )
        if any(len(contextual_features[name]) > 0 for name in self._contextual_feature_names):
            contextual_features_tensor = torch.tensor(
                [val for name in self._contextual_feature_names for val in contextual_features[name]]
            )
            contextual_features_lengths_tensor = torch.tensor(
                [length for name in self._contextual_feature_names for length in contextual_features_seqlen[name]]
            )
        else:
            contextual_features_tensor = torch.empty((0,), dtype=torch.int64)
            contextual_features_lengths_tensor = torch.tensor(
                [length for name in self._contextual_feature_names for length in contextual_features_seqlen[name]]
            )
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=self._contextual_feature_names
            + [self._item_feature_name, self._action_feature_name],
            values=torch.concat(
                [
                    contextual_features_tensor.to(device=self._device),
                    torch.tensor(item_features, device=self._device),
                    torch.tensor(action_features, device=self._device),
                ]
            ).long(),
            lengths=torch.concat(
                [
                    contextual_features_lengths_tensor.to(device=self._device),
                    torch.tensor(item_features_seqlen, device=self._device),
                    torch.tensor(action_features_seqlen, device=self._device),
                ]
            ).long(),
        )
        labels = torch.tensor(labels, dtype=torch.int64, device=self._device)
        batch_kwargs = dict(
            features=features,
            # batch_size=self._batch_size,
            batch_size=len(user_ids),
            feature_to_max_seqlen=feature_to_max_seqlen,
            contextual_feature_names=self._contextual_feature_names,
            item_feature_name=self._item_feature_name,
            action_feature_name=self._action_feature_name,
            max_num_candidates=self._max_num_candidates,
            num_candidates=torch.tensor(num_candidates, device=self._device)
            if self._max_num_candidates > 0
            else None,
        )
        if with_ranking_labels:
            return RankingBatch(labels=labels, **batch_kwargs)

        return Batch(**batch_kwargs)
    
    def get_regular_batch_user_ids(self) -> Optional[tuple]:
        batch_size = self._max_batch_size

        if not hasattr(self, "_current_user_id"):
            self._current_user_id = 0
            self._current_user_count = 0
            self._after_36_special = False
            self._final_special_done = False

        current_batch_start = self._current_user_id
        
        if self._after_36_special:
            if not self._final_special_done:
                # 生成特殊批次 [1, 5, 12, 36]
                user_ids = [1, 5, 12, 36]
                self._final_special_done = True
            else:
                # 特殊批次已生成，可以选择结束或重新开始
                return None  # 结束迭代
        else:
            if current_batch_start == 4:  # 批次 [4,5,6,7]
                max_count = 6
            elif current_batch_start == 36:  # 批次 [36,37,38,39]
                max_count = 18
            else:
                max_count = 32

            if self._current_user_count < max_count:
                self._current_user_count += 1
            else:
                if current_batch_start == 36:
                    self._current_user_id = 4
                    self._current_user_count = 1
                    self._after_36_special = True
                else:
                    self._current_user_id += batch_size
                    self._current_user_count = 1

            if not self._after_36_special:
                user_ids = [self._current_user_id + i for i in range(batch_size)]
            else:
                # 这种情况下会在下次调用时生成特殊批次
                user_ids = [self._current_user_id + i for i in range(batch_size)]
        
        dates = [20240101] * len(user_ids)
        
        seq_endptrs = []
        for uid in user_ids:
            if uid in self._item_history:
                current_len = len(self._item_history[uid])
            else:
                current_len = 0
            
            if self._full_mode:
                increment = self._max_incr_fea_len
            else:
                increment = torch.randint(1, self._max_incr_fea_len + 1, (1,)).item()
            
            new_end = min(current_len + increment, self._max_hist_len)
            seq_endptrs.append(new_end)

        user_ids_tensor = torch.tensor(user_ids, device=self._device).long()
        dates_tensor = torch.tensor(dates, device=self._device).long()
        seq_endptrs_tensor = torch.tensor(seq_endptrs, device=self._device).long()

        return (user_ids_tensor, dates_tensor, seq_endptrs_tensor)


    def get_regular_input_batch(
        self,
        user_ids,
        dates,
        sequence_endptrs,
        sequence_startptrs,
        with_contextual_features=False,
        with_ranking_labels=False,
    ):
        item_features: List[int] = []
        item_features_seqlen: List[int] = []
        action_features: List[int] = []
        action_features_seqlen: List[int] = []
        num_candidates_list: List[int] = []
        labels: List[int] = []

        batch_size = len(user_ids)
        if batch_size == 0:
            return None
        
        user_ids = user_ids.to(self._device)
        sequence_endptrs = sequence_endptrs.to(self._device)
        sequence_startptrs = sequence_startptrs.to(self._device)

        user_ids_list = user_ids.tolist()

        # 初始化或获取用户序列
        item_hists = [
            self._item_history[uid] if uid in self._item_history else torch.tensor([], device=self._device)
            for uid in user_ids_list
        ]
        action_hists = [
            self._action_history[uid]
            if uid in self._action_history
            else torch.tensor([], device=self._device)
            for uid in user_ids_list
        ]
        lengths = torch.tensor([len(hist_seq) for hist_seq in item_hists], device=self._device).long()

        target_lengths = sequence_endptrs.long()
        incr_lengths = torch.clamp(target_lengths - lengths, min=0, max=self._max_incr_fea_len)

        num_candidates = torch.randint(
            low=1, high=self._max_num_candidates + 1, size=(batch_size,), device=self._device
        )
        if self._full_mode:
            num_candidates = torch.full((batch_size,), self._max_num_candidates, device=self._device)
        if self._max_num_candidates == 0:
            num_candidates = torch.zeros((batch_size,), device=self._device, dtype=torch.long)

        item_start_positions = sequence_startptrs.to(torch.int32)
        action_start_positions = sequence_startptrs.to(torch.int32)

        if isinstance(with_contextual_features, bool):
            contextual_mask = [with_contextual_features] * len(user_ids)
        else:
            contextual_mask = with_contextual_features


        for idx, uid in enumerate(user_ids_list):
            if incr_lengths[idx] > 0:
                self._item_history[uid] = torch.cat([
                    item_hists[idx],
                    torch.randint(self._item_vocab_size, (incr_lengths[idx],), device=self._device)
                ], dim=0).long()
                self._action_history[uid] = torch.cat([
                    action_hists[idx],
                    torch.randint(128, (incr_lengths[idx],), device=self._device)
                ], dim=0).long()

            item_history = self._item_history[uid][item_start_positions[idx]:]
            action_history = self._action_history[uid][action_start_positions[idx]:]

            if self._max_num_candidates > 0:
                candidate_seq = torch.randint(
                    self._item_vocab_size, (num_candidates[idx].item(),), device=self._device
                )
                item_history = torch.cat([item_history, candidate_seq], dim=0)
                if with_ranking_labels:
                    label_seq = torch.randint(2, (num_candidates[idx].item(),), device=self._device)
                    labels.extend(label_seq.tolist())
                num_candidates_list.append(num_candidates[idx].item())
            else:
                num_candidates_list.append(0)

            item_features.extend(item_history.tolist())
            item_features_seqlen.append(len(item_history))

            action_features.extend(action_history.tolist())
            action_features_seqlen.append(len(action_history))

        contextual_features: Dict[str, List[int]] = defaultdict(list)
        contextual_features_seqlen: Dict[str, List[int]] = defaultdict(list)
        
        for idx, uid in enumerate(user_ids_list):
            if contextual_mask[idx]:
                # 为有上下文特征的用户生成特征值
                for contextual_feature_name in self._contextual_feature_names:
                    # 生成随机的上下文特征值（根据实际需求调整范围）
                    contextual_value = torch.randint(2, (1,), device=self._device).item()
                    contextual_features[contextual_feature_name].append(contextual_value)
                    contextual_features_seqlen[contextual_feature_name].append(1)
            else:
                # 为无上下文特征的用户添加空值
                for contextual_feature_name in self._contextual_feature_names:
                    contextual_features_seqlen[contextual_feature_name].append(0)

        feature_to_max_seqlen = {}
        for name in self._contextual_feature_names:
            feature_to_max_seqlen[name] = max(
                contextual_features_seqlen[name], default=0
            )

        ### Currently use clipped maxlen. check how this impacts the hstu results
        feature_to_max_seqlen[self._item_feature_name] = max(item_features_seqlen)
        feature_to_max_seqlen[self._action_feature_name] = max(action_features_seqlen)

        if any(len(contextual_features[name]) > 0 for name in self._contextual_feature_names):
            contextual_features_tensor = torch.tensor(
                [val for name in self._contextual_feature_names for val in contextual_features[name]]
            )
            contextual_features_lengths_tensor = torch.tensor(
                [length for name in self._contextual_feature_names for length in contextual_features_seqlen[name]]
            )
        else:
            contextual_features_tensor = torch.empty((0,), dtype=torch.int64)
            contextual_features_lengths_tensor = torch.tensor(
                [length for name in self._contextual_feature_names for length in contextual_features_seqlen[name]]
            )
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=self._contextual_feature_names
            + [self._item_feature_name, self._action_feature_name],
            values=torch.concat(
                [
                    contextual_features_tensor.to(device=self._device),
                    torch.tensor(item_features, device=self._device),
                    torch.tensor(action_features, device=self._device),
                ]
            ).long(),
            lengths=torch.concat(
                [
                    contextual_features_lengths_tensor.to(device=self._device),
                    torch.tensor(item_features_seqlen, device=self._device),
                    torch.tensor(action_features_seqlen, device=self._device),
                ]
            ).long(),
        )
        
        batch_kwargs = dict(
            features=features,
            batch_size=len(user_ids),
            feature_to_max_seqlen=feature_to_max_seqlen,
            contextual_feature_names=self._contextual_feature_names,
            item_feature_name=self._item_feature_name,
            action_feature_name=self._action_feature_name,
            max_num_candidates=self._max_num_candidates,
            num_candidates=torch.tensor(num_candidates, device=self._device)
            if self._max_num_candidates > 0
            else None,
        )
        if with_ranking_labels:
            labels = torch.tensor(labels, dtype=torch.int64, device=self._device)
            return RankingBatch(labels=labels, **batch_kwargs)

        return Batch(**batch_kwargs)