/******************************************************************************
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
#
# Implementation based on FlashInfer library.
# 
******************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <chrono>

#define cudaCheck(ans) { cudaSuccesAssert((ans), __FILE__, __LINE__); }
inline void cudaSuccesAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    //   if (abort) exit(code);
   }
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVCache(DType* k_data,
                               DType* v_data,
                               IdType* indices,
                               IdType* indptr,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* append_key, DType* append_value, IdType* batch_indices, 
                               IdType* positions, IdType* offsets, 
                               IdType* nnz_cuda, uint32_t nnz, 
                               size_t append_k_stride_n, size_t append_k_stride_h,
                               size_t append_v_stride_n, size_t append_v_stride_h,
                               cudaStream_t stream);

template <typename DType, typename IdType>
cudaError_t GatherPagedKVCache(DType* gather_kv,
                               IdType* page_ids,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_k2v,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* kv_cache,
                               uint32_t nnz,
                               cudaStream_t stream);

template <typename DType, typename IdType>
cudaError_t GatherPagedKVCacheAllLayers(DType* gather_kv,
                                        IdType* page_ids,
                                        uint32_t num_layers,
                                        uint32_t stride_gather,
                                        uint32_t stride_layer,
                                        uint32_t num_heads,
                                        uint32_t head_dim,
                                        uint32_t page_size,
                                        uint32_t stride_page,
                                        uint32_t stride_k2v,
                                        uint32_t stride_n,
                                        uint32_t stride_h,
                                        DType* kv_cache,
                                        uint32_t nnz,
                                        cudaStream_t stream);

cudaError_t GetPagedBatchIndicesPositions(
  int32_t batch_size,
  int32_t* append_indptr,
  int32_t* seq_lens_ptr,
  int32_t* batch_indices_ptr,
  int32_t* positions_ptr,
  cudaStream_t stream
);

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor seqlen_offsets, 
                           at::Tensor nnz_cuda, unsigned int nnz,
                           at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t kv_layout) {
  // unsigned int batch_size = kv_last_page_len.size(0);
  auto device = append_key.device();

  unsigned int num_heads, page_size, head_dim;
  head_dim = paged_k_cache.size(3);
  if (kv_layout == 1) {
    num_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_heads = paged_k_cache.size(2);
  }

  auto stride_page = paged_k_cache.stride(0);
  auto stride_n = (kv_layout == 1) ? head_dim : num_heads * head_dim;
  auto stride_h = (kv_layout == 1) ? page_size * head_dim : head_dim;

  // get kv_cache_strides
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");


  auto append_k_strides = append_key.strides();
  auto append_k_stride_n = append_k_strides[0];
  auto append_k_stride_h = append_k_strides[1];
  auto append_v_strides = append_value.strides();
  auto append_v_stride_n = append_v_strides[0];
  auto append_v_stride_h = append_v_strides[1];

  auto kv_scalar_dtype = paged_k_cache.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t status;
  switch (kv_scalar_dtype) {
    case at::ScalarType::BFloat16:
        status =
        AppendPagedKVCache(static_cast<nv_bfloat16*>(paged_k_cache.data_ptr()),
                           static_cast<nv_bfloat16*>(paged_v_cache.data_ptr()),
                           static_cast<int32_t*>(kv_indices.data_ptr()),
                           static_cast<int32_t*>(kv_indptr.data_ptr()),
                           num_heads, head_dim, page_size, stride_page, stride_n, stride_h,
                           static_cast<nv_bfloat16*>(append_key.data_ptr()),
                           static_cast<nv_bfloat16*>(append_value.data_ptr()),
                           static_cast<int32_t*>(batch_indices.data_ptr()),
                           static_cast<int32_t*>(positions.data_ptr()), 
                           static_cast<int32_t*>(seqlen_offsets.data_ptr()), 
                           static_cast<int32_t*>(nnz_cuda.data_ptr()), 
                           nnz, append_k_stride_n, append_k_stride_h, 
                           append_v_stride_n, append_v_stride_h, stream);
        break;
    case at::ScalarType::Half:
        status =
        AppendPagedKVCache(static_cast<nv_half*>(paged_k_cache.data_ptr()), 
                           static_cast<nv_half*>(paged_v_cache.data_ptr()),
                           static_cast<int32_t*>(kv_indices.data_ptr()),
                           static_cast<int32_t*>(kv_indptr.data_ptr()),
                           num_heads, head_dim, page_size, stride_page, stride_n, stride_h,
                           static_cast<nv_half*>(append_key.data_ptr()),
                           static_cast<nv_half*>(append_value.data_ptr()),
                           static_cast<int32_t*>(batch_indices.data_ptr()),
                           static_cast<int32_t*>(positions.data_ptr()), 
                           static_cast<int32_t*>(seqlen_offsets.data_ptr()), 
                           static_cast<int32_t*>(nnz_cuda.data_ptr()), 
                           nnz, append_k_stride_n, append_k_stride_h, 
                           append_v_stride_n, append_v_stride_h, stream);
        break;
    default:
        TORCH_CHECK(false, "AppendPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
  }
  TORCH_CHECK(status == cudaSuccess,
              "AppendPagedKVCache failed with error: ", cudaGetErrorString(status));
}

void gather_paged_kv_cache(at::Tensor gather_kv_gpu_buffer,
                           at::Tensor paged_kv_cache,
                           at::Tensor page_ids_to_offload,
                           unsigned int num_pages,
                           int64_t kv_layout) {
  auto device = paged_kv_cache.device();

  TORCH_CHECK(paged_kv_cache.ndimension() == 5, 
              "kv cache table must has 5 dimensions (num_pages, 2, page_size, num_head, head_dim).");
  
  unsigned int num_heads, page_size, head_dim;
  head_dim = paged_kv_cache.size(4);
  if (kv_layout == 1) {
    num_heads = paged_kv_cache.size(2);
    page_size = paged_kv_cache.size(3);
  } else {
    page_size = paged_kv_cache.size(2);
    num_heads = paged_kv_cache.size(3);
  }

  auto stride_page = paged_kv_cache.stride(0);
  auto stride_n = (kv_layout == 1) ? head_dim : num_heads * head_dim;
  auto stride_h = (kv_layout == 1) ? page_size * head_dim : head_dim;
  auto stride_k2v = paged_kv_cache.stride(1);

  // check input/output strides
  TORCH_CHECK(paged_kv_cache.strides() == gather_kv_gpu_buffer.strides(), 
              "input/output strides must be identical");
  TORCH_CHECK(paged_kv_cache.is_contiguous() && paged_kv_cache.is_contiguous(), 
              "buffer must be contiguous");
  
  auto kv_scalar_dtype = paged_kv_cache.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t status;
  switch (kv_scalar_dtype) {
    case at::ScalarType::BFloat16:
        status = GatherPagedKVCache(
            static_cast<nv_bfloat16*>(gather_kv_gpu_buffer.data_ptr()),
            static_cast<int32_t*>(page_ids_to_offload.data_ptr()),
            num_heads, head_dim, page_size, 
            stride_page, stride_k2v, stride_n, stride_h,
            static_cast<nv_bfloat16*>(paged_kv_cache.data_ptr()),
            num_pages * page_size, stream);
        break;
    case at::ScalarType::Half:
        status = GatherPagedKVCache(
            static_cast<nv_half*>(gather_kv_gpu_buffer.data_ptr()),
            static_cast<int32_t*>(page_ids_to_offload.data_ptr()),
            num_heads, head_dim, page_size, 
            stride_page, stride_k2v, stride_n, stride_h,
            static_cast<nv_half*>(paged_kv_cache.data_ptr()),
            num_pages * page_size, stream);
        break;
    default:
        TORCH_CHECK(false, "GatherPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
  }
  TORCH_CHECK(status == cudaSuccess,
              "GatherPagedKVCache failed with error: ", cudaGetErrorString(status));
}

void gather_paged_kv_cache_all_layers(uint16_t *gather_kv_gpu_buffer,
                                      uint16_t *paged_kv_cache,
                                      int *page_ids_to_offload,
                                      uint32_t num_layers,
                                      uint32_t stride_gather,
                                      uint32_t stride_layer,
                                      uint32_t num_heads,
                                      uint32_t head_dim,
                                      uint32_t page_size,
                                      uint32_t stride_page,
                                      uint32_t stride_k2v,
                                      uint32_t stride_n,
                                      uint32_t stride_h,
                                      uint32_t num_pages,
                                      cudaStream_t stream) {
  // auto device = paged_kv_cache.device();
  // const c10::cuda::OptionalCUDAGuard device_guard(device);

  cudaError_t status;
  status = GatherPagedKVCacheAllLayers(
      reinterpret_cast<nv_bfloat16*>(gather_kv_gpu_buffer),
      static_cast<int32_t*>(page_ids_to_offload),
      num_layers, stride_gather, stride_layer, 
      num_heads, head_dim, page_size, 
      stride_page, stride_k2v, stride_n, stride_h,
      reinterpret_cast<nv_bfloat16*>(paged_kv_cache),
      num_pages * page_size, stream);
  TORCH_CHECK(status == cudaSuccess,
              "GatherPagedKVCacheAllLayers failed with error: ", cudaGetErrorString(status));
}

namespace kvcache {

class HostKVStorageImpl
{
public:
    HostKVStorageImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int64_t num_tokens_per_chunk
    )
        : num_layers(num_layers)
        , num_kv_heads(num_kv_heads)
        , kv_headdim(kv_headdim)
        , page_size(num_tokens_per_page)
        , chunk_size(num_tokens_per_chunk)
        , _uid_to_chunk_id(num_layers, std::unordered_map<int64_t, std::vector<uintptr_t>>())
    {
        this->chunk_numel = num_tokens_per_chunk * 2 * num_kv_heads * kv_headdim;
        this->page_numel = 2 * page_size * num_kv_heads * kv_headdim;
        this->per_token_numel = 2 * num_kv_heads * kv_headdim;
    };

    ~HostKVStorageImpl()
    {}

    int64_t get_kvdata_length(int64_t user_id) {
        auto it = _uid_to_length.find(user_id);
        if (it ==  _uid_to_length.end()) return 0;
        return it->second;
    };

    void append_kvdata_v2(int64_t user_id, int64_t start_position, int64_t length, uint16_t *pinned_input_ptr, size_t gather_layer_stride) {
        assert(length % this->chunk_size == 0);
        if (start_position != 0) {
            assert(_uid_to_length[user_id] == start_position);
        }
        else {
            assert(_uid_to_length.find(user_id) == _uid_to_length.end());
            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
                _uid_to_chunk_id[layer_idx][user_id] = std::vector<uintptr_t>();
            _uid_to_mempool[user_id] = std::vector<uintptr_t>();
        }

            size_t num_chunks = length / chunk_size;
            size_t num_elem = length * per_token_numel;
            size_t kvdata_size = num_elem * sizeof(uint16_t);

        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            uint16_t* src_ptr = pinned_input_ptr + layer_idx * gather_layer_stride;
            
            for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
                _uid_to_chunk_id[layer_idx][user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr + chunk_idx * this->chunk_numel));
            }

            _uid_to_mempool[user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr));
        }
        _uid_to_length[user_id] = start_position + length;
    };

    std::vector<uint16_t*> get_kvdata_v2(int64_t user_id, int64_t length, int64_t layer_idx) {
        // int64_t offloaded_length = get_kvdata_length(user_id);
        // assert(offloaded_length >= length);

        std::vector<uint16_t*> chunk_ptrs;
        if (length == 0) {
            return chunk_ptrs;
        }
        // assert(length % this->chunk_size == 0);
        size_t num_chunks = length / chunk_size;
        const size_t chunk_bytesize = this->chunk_numel * sizeof(uint16_t);
        const auto &chunk_ptr_list = _uid_to_chunk_id[layer_idx][user_id];
        // assert(chunk_ptr_list.size() >= num_chunks);
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
            uint16_t* src_ptr = reinterpret_cast<uint16_t*>(chunk_ptr_list[chunk_idx]);
            chunk_ptrs.push_back(src_ptr);
        }
        return chunk_ptrs;
    };

    std::vector<std::pair<int64_t, int64_t>> get_cache_startpos_and_length(std::vector<int64_t>& user_ids) {
        int batch_size = user_ids.size();
        std::vector<std::pair<int64_t, int64_t>> result(batch_size);
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            int64_t uid = user_ids[seq_idx];
            auto it = _uid_to_length.find(uid);
            if (it != _uid_to_length.end()) {
                result[seq_idx] = std::make_pair(0, it->second); // 在host中起始位置总是0
            } else {
                result[seq_idx] = std::make_pair(0, 0);
            }
        }
        return result;
    };

    public:
        std::vector<std::unordered_map<int64_t, std::vector<uintptr_t>>> _uid_to_chunk_id;
        std::unordered_map<int64_t, int64_t> _uid_to_length;
        std::unordered_map<int64_t, std::vector<uintptr_t>> _uid_to_mempool;

        const int num_layers;
        const int num_kv_heads;
        const int kv_headdim;
        const int page_size;

        const int64_t chunk_size;
        size_t chunk_numel;
        size_t page_numel;
        size_t per_token_numel;
        size_t layer_numel;
};

class KVOnloadHandle {
public:
    KVOnloadHandle(
        int num_layers
    )
    : num_layers(num_layers)
    , event(std::vector<cudaEvent_t>(num_layers))
    , host_complete(num_layers, 0) {
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
            cudaEventCreate(&event[layer_idx]);
        }
    };

    ~KVOnloadHandle(){
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
            cudaEventDestroy(event[layer_idx]);
        }
    };

    // void record(int layer_idx, cudaStream_t stream) {
    //     cudaEventRecord(event[layer_idx], stream);
    // };

    // void wait(int layer_idx) {
    //     auto stream = at::cuda::getCurrentCUDAStream();
    //     cudaStreamWaitEvent(stream, event[layer_idx], 0);
    // };
    void reset() {
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
            host_complete[layer_idx] = 0;
        }
    }

    void complete_host(int layer_idx, cudaStream_t stream) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cudaEventRecord(event[layer_idx], stream);
            host_complete[layer_idx] = 1;
        }
        cv_.notify_one();
    };

    void wait_host(int layer_idx) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this, layer_idx](){ return host_complete[layer_idx] == 1; });
        }
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaStreamWaitEvent(stream, event[layer_idx], 0);
    };

    void wait_all(void);
public:
    int num_layers;
    std::vector<cudaEvent_t> event;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<int> host_complete;
};

class KVOffloadHandle {
public:

void record_ready(void) {
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaEventCreate(&ready_event);
        cudaEventRecord(ready_event, stream);
    };

    // void wait_ready(cudaStream_t stream) {
    //     cudaStreamWaitEvent(stream, ready_event);
    // };

public:
    cudaEvent_t ready_event;
};

class GPUKVCacheMangerImpl
{
public:
    GPUKVCacheMangerImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int num_primary_cache_pages,
        int num_onload_buffer_pages,
        int num_reserved_buffer_pages,
        int num_tokens_per_chunk,
        int max_num_sequences,
        int max_sequence_length,
        at::Tensor cache_table_tensor,
        HostKVStorageImpl& host_kv_mgr)
     : num_layers(num_layers)
     , num_kv_heads(num_kv_heads)
     , kv_headdim(kv_headdim)
     , num_tokens_per_page(num_tokens_per_page)
     , num_primary_cache_pages(num_primary_cache_pages)
     , num_onload_buffer_pages(num_onload_buffer_pages)
     , num_reserved_buffer_pages(num_reserved_buffer_pages)
     , num_tokens_per_chunk(num_tokens_per_chunk)
     , max_num_sequences(max_num_sequences)
     , max_sequence_length(max_sequence_length)
     , cache_table(static_cast<uint16_t*>(cache_table_tensor.data_ptr()))
     , device(cache_table_tensor.device())
     , onload_pinned_buffers(2)
     , onload_memcpy_event(2)
     , offload_pinned_buffers(2)
     , offload_memcpy_event(2)
    {
        const c10::cuda::OptionalCUDAGuard device_guard(this->device);

        for (int page_id = 0; page_id < num_primary_cache_pages; page_id++)
            _empty_pages.push(page_id);
    
        page_stride = 2 * num_tokens_per_page * num_kv_heads * kv_headdim;
        k2v_stride = num_tokens_per_page * num_kv_heads * kv_headdim;
        layer_stride = (num_primary_cache_pages + num_onload_buffer_pages) * page_stride;
        per_token_kv_stride = 2 * num_kv_heads * kv_headdim;

        cudaStreamCreate(&worker_stream);
        cudaStreamCreate(&onload_stream);
        cudaStreamCreate(&offload_stream);
        cudaStreamCreate(&sync_stream);

        this->host_kv_mgr = &host_kv_mgr;

        for (int i = 0; i < 2; i++) {
            cudaMallocHost((void**)&onload_pinned_buffers[i], host_kv_mgr.chunk_numel * sizeof(uint16_t));
            cudaEventCreate(&onload_memcpy_event[i]);
        }

        for (int i = 0; i < 2; i++) {
            cudaMallocHost((void**)&offload_pinned_buffers[i], host_kv_mgr.chunk_numel * sizeof(uint16_t));
            cudaEventCreate(&offload_memcpy_event[i]);
        }

        this->terminate_ = false;
        this->num_onload_memcpy_worker = 3;
        for (int i = 0; i < num_onload_memcpy_worker; i++) {
            this->onload_memcpy_worker.emplace_back(std::thread(
                &GPUKVCacheMangerImpl::onload_host_memcpy_loop, this 
            ));
        }

        this->num_offload_memcpy_worker = 3;
        for (int i = 0; i < num_offload_memcpy_worker; i++) {
            this->offload_memcpy_worker.emplace_back(std::thread(
                &GPUKVCacheMangerImpl::offload_host_memcpy_loop, this 
            ));
        }

        this->offload_busy_.store(false);
        this->offload_worker = std::thread(&GPUKVCacheMangerImpl::offload_loop, this);
    };

    ~GPUKVCacheMangerImpl() {
        {
            std::unique_lock<std::mutex> lock(onload_memcpy_task_mtx_);
            std::unique_lock<std::mutex> lock2(offload_memcpy_task_mtx_);
            std::unique_lock<std::mutex> lock3(offload_task_mutex_);
            this->terminate_ = true;
        }
        onload_memcpy_task_cv_.notify_all();
        offload_memcpy_task_cv_.notify_all();
        offload_task_cv_.notify_all();

        for (int i = 0; i < num_onload_memcpy_worker; i++) {
            this->onload_memcpy_worker[i].join();
        }

        for (int i = 0; i < num_offload_memcpy_worker; i++) {
            this->offload_memcpy_worker[i].join();
        }

        this->offload_worker.join();

        for (int i = 0; i < 2; i++) {
            cudaFree(onload_pinned_buffers[i]);
            cudaFree(offload_pinned_buffers[i]);
            cudaEventDestroy(onload_memcpy_event[i]);
            cudaEventDestroy(offload_memcpy_event[i]);
        }
    }

        int64_t getUIdToEvict(std::unordered_set<int64_t> freezed_uids) {
            if (freezed_uids.size() == 0)
                return _lru_list.back();
        
            for (auto it = std::rbegin(_lru_list); it != std::rend(_lru_list); ++it) {
                if (freezed_uids.find((int64_t)*it) != freezed_uids.end())
                    continue;
                return *it;
            }
            assert(false);
            return _lru_list.back();
        };

        std::vector<int32_t>& alloc(int64_t uid, int new_total_length, std::unordered_set<int64_t> freezed_uids) {
            int cur_cached_start = 0;
            int cur_cached_len = 0;
            // int padding_last_page = 0;

            bool found_in_cache = retain(uid);
            if (found_in_cache) {
                cur_cached_start = _uid_to_paged_cache_startpos[uid];
                cur_cached_len = _uid_to_paged_cache_length[uid];
            } else {
                _uid_to_page_id[uid] = std::vector<int32_t>();
                if (_uid_to_offloaded_length.find(uid) != _uid_to_offloaded_length.end()) {
                    _uid_to_paged_cache_startpos[uid] = _uid_to_offloaded_length[uid];
                    cur_cached_start = _uid_to_offloaded_length[uid];
                }
                else {
                    _uid_to_paged_cache_startpos[uid] = 0;
                }
            }

            // // 为onload部分分配page，并插入到最前面
            // if (cur_cached_start > 0) {
            //     int onload_num_pages = cur_cached_start / num_tokens_per_page;
            //     while ((size_t)onload_num_pages > _empty_pages.size()) {
            //         int64_t uid_to_evict = getUIdToEvict(freezed_uids);
            //         evict(uid_to_evict);
            //     }
            //     std::vector<int32_t> onload_pages;
            //     for (int i = 0; i < onload_num_pages; i++) {
            //         onload_pages.push_back(_empty_pages.front());
            //         _empty_pages.pop();
            //     }
            //     _uid_to_page_id[uid].insert(_uid_to_page_id[uid].begin(), onload_pages.begin(), onload_pages.end());
            //     _uid_to_paged_cache_startpos[uid] = 0;
            //     cur_cached_start = 0;
            //     cur_cached_len += cur_cached_start;
            // }

            int new_cached_len = new_total_length - cur_cached_start;
            int cur_num_pages = (cur_cached_len + num_tokens_per_page - 1) / num_tokens_per_page;
            int new_num_pages = (new_cached_len + num_tokens_per_page - 1) / num_tokens_per_page;

            int num_append_pages = new_num_pages - cur_num_pages;
            // std::cout << " *** " << cur_num_pages << " " << cur_cached_len << std::endl;
            // std::cout << " *** " << new_num_pages << " " << new_cached_len << std::endl;

            while ((size_t)num_append_pages > _empty_pages.size()) {
                int64_t uid_to_evict = getUIdToEvict(freezed_uids);
                evict(uid_to_evict);
                // std::cout << "evict " << uid_to_evict << std::endl;
            }

        std::vector<int32_t>& page_ids = _uid_to_page_id[uid];
        // for (auto pid : page_ids) {
        //     // std::cout << " - " << pid << std::endl;
        // }
        for (int i = 0; i < num_append_pages; i++) {
            page_ids.push_back(_empty_pages.front());
            _empty_pages.pop();
        }
        _uid_to_paged_cache_length[uid] = new_cached_len;

            return page_ids;
        };

        std::vector<int32_t> get_total_cache_length(std::vector<int64_t>& uids) {
            int batch_size = uids.size();
            std::vector<int32_t> total_cached_lengths(batch_size);
            for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
                int64_t uid = uids[seq_idx];
                if (_uid_to_paged_cache_startpos.find(uid) != _uid_to_paged_cache_startpos.end()) {
                    total_cached_lengths[seq_idx] = _uid_to_paged_cache_startpos[uid] + _uid_to_paged_cache_length[uid];
                } else if (_uid_to_offloaded_length.find(uid) != _uid_to_offloaded_length.end())
                    total_cached_lengths[seq_idx] = _uid_to_offloaded_length[uid];
                else {
                    total_cached_lengths[seq_idx] = 0;
                }
            }
            return total_cached_lengths;
        };
        
        void evict(int64_t uid)
        {
            auto const tableIt = _lru_lookup_table.find(uid);
            assert(_lru_lookup_table.end() != tableIt);
            // if (_lru_lookup_table.end() != tableIt) {
                _lru_list.erase(tableIt->second);
                _lru_lookup_table.erase(tableIt);
                // assert(_uid_to_page_id[uid].size() > 0);

                for (auto page_id : _uid_to_page_id[uid]) {
                    _empty_pages.push(page_id);
                }

            _uid_to_page_id.erase(uid);
            _uid_to_paged_cache_startpos.erase(uid);
            _uid_to_paged_cache_length.erase(uid);
        // }
    };

    void evict_all()
    {
        std::queue<int64_t> empty_pages;
        std::swap(_empty_pages, empty_pages);
        _lru_list.clear();
        _lru_lookup_table.clear();
        _uid_to_page_id.clear();
        _uid_to_paged_cache_startpos.clear();
        _uid_to_paged_cache_length.clear();

        for (int page_id = 0; page_id < this->num_primary_cache_pages; page_id++)
            _empty_pages.push(page_id);
    };

        void invalid(int64_t uid) {
            auto const tableIt = _lru_lookup_table.find(uid);
            if (_lru_lookup_table.end() != tableIt) {
                _lru_list.erase(tableIt->second);
                _lru_lookup_table.erase(tableIt);

                for (auto page_id : _uid_to_page_id[uid]) {
                    _empty_pages.push(page_id);
                }

                _uid_to_page_id.erase(uid);
                _uid_to_paged_cache_startpos.erase(uid);
                _uid_to_paged_cache_length.erase(uid);
                _uid_to_offloaded_length.erase(uid);
            }
        };

        bool retain(int64_t uid)
        {
            auto const tableIt = _lru_lookup_table.find(uid);
            bool found = (_lru_lookup_table.end() != tableIt);
            if (found) {
                _lru_list.erase(tableIt->second);
            }
            _lru_list.push_front(uid);
            _lru_lookup_table[uid] = _lru_list.begin();
            return found;
        };

        uint16_t *get_cache_table(void) {
            return cache_table;
        };

        uint16_t *get_cache_table_by_layer(int layer_idx) {
            return cache_table + layer_idx * layer_stride;
        };

        std::vector<std::pair<int32_t, int32_t>> get_cache_startpos_and_length(std::vector<int64_t>& user_ids) {
            int batch_size = user_ids.size();
            std::vector<std::pair<int32_t, int32_t>> result(batch_size);
            for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
                int64_t uid = user_ids[seq_idx];
                int32_t start_pos = 0;
                int32_t gpu_cache_length = 0;
                
                if (_uid_to_paged_cache_startpos.find(uid) != _uid_to_paged_cache_startpos.end()) {
                    start_pos = _uid_to_paged_cache_startpos[uid];
                    gpu_cache_length = _uid_to_paged_cache_length[uid];  // 只返回GPU缓存长度
                } else if (_uid_to_offloaded_length.find(uid) != _uid_to_offloaded_length.end()) {
                    start_pos = _uid_to_offloaded_length[uid];
                    gpu_cache_length = 0;  // 只有卸载数据，GPU缓存长度为0
                }
                
                result[seq_idx] = std::make_pair(start_pos, gpu_cache_length);
            }
            return result;
        };

    public:
    void onload_kvcache(
        std::vector<int64_t>& user_ids, 
        KVOnloadHandle& onloadhandle) {
        const c10::cuda::OptionalCUDAGuard device_guard(this->device);

        // std::cout << "onload_kvcache start" << std::endl << std::flush;
        const int batch_size = user_ids.size();

        std::vector<size_t> onload_length(batch_size);
        std::vector<size_t> onload_offsets(batch_size + 1);
        onload_offsets[0] = 0;
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            auto uid = user_ids[seq_idx];
            if (this->_uid_to_paged_cache_startpos.find(uid) != this->_uid_to_paged_cache_startpos.end())
                onload_length[seq_idx] = this->_uid_to_paged_cache_startpos[uid];
            else if (this->_uid_to_offloaded_length.find(uid) != this->_uid_to_offloaded_length.end())
                onload_length[seq_idx] = this->_uid_to_offloaded_length[uid];
            else
                onload_length[seq_idx] = 0;

            onload_offsets[seq_idx + 1] = onload_offsets[seq_idx] + onload_length[seq_idx];
        }
        size_t total_onload_length = onload_offsets[batch_size];
        if (total_onload_length == 0) {
            for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++)
                onloadhandle.complete_host(layer_idx, this->onload_stream);
            // std::cout << "onload_kvcache empty" << std::endl << std::flush;
            return;
        }

        const size_t chunk_numel_part = host_kv_mgr->chunk_numel / (this->num_onload_memcpy_worker + 1);

        for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
            uint16_t *gpu_onload_buffer = this->get_cache_table_by_layer(layer_idx) + this->num_primary_cache_pages * this->page_stride;

            int task_idx = 0;
            for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
                std::vector<uint16_t *> chunk_ptrs = host_kv_mgr->get_kvdata_v2(user_ids[0], onload_length[0], layer_idx);
                
                // std::cout << "[Onload] uid: " << user_ids[seq_idx] << " - " << chunk_ptrs.size() << std::endl;
                // std::cout << "\t" << host_kv_mgr->chunk_numel << " - " << host_kv_mgr->chunk_size * this->per_token_kv_stride << std::endl;
                for (int chunk_idx = 0; chunk_idx < chunk_ptrs.size(); chunk_idx++) {
                    // std::cout << "\t" << reinterpret_cast<uintptr_t>(chunk_ptrs[chunk_idx]) << " - " << (chunk_ptrs[chunk_idx] - chunk_ptrs[0]) << std::endl;

                    onload_host_memcpy(onload_pinned_buffers[task_idx%2], chunk_ptrs[chunk_idx], host_kv_mgr->chunk_numel * sizeof(uint16_t), chunk_numel_part * sizeof(uint16_t));
                    
                    cudaCheck(cudaMemcpyAsync(gpu_onload_buffer + onload_offsets[seq_idx] * this->per_token_kv_stride + chunk_idx * host_kv_mgr->chunk_numel, 
                        onload_pinned_buffers[task_idx%2], host_kv_mgr->chunk_numel * sizeof(uint16_t), cudaMemcpyHostToDevice, this->onload_stream));
                    cudaCheck(cudaEventRecord(offload_memcpy_event[task_idx%2], this->onload_stream));

                    if (task_idx > 0) {
                        cudaCheck(cudaEventSynchronize(offload_memcpy_event[(task_idx - 1)%2]));
                    }


                    task_idx++;
                }
            }

            onloadhandle.complete_host(layer_idx, this->onload_stream);
        }
        // std::cout << "onload_kvcache end" << std::endl << std::flush;
    };

    void offload_kvcache(
        KVOffloadHandle& offload_handle,
        at::Tensor offload_user_ids,      // host
        at::Tensor offload_page_ids,      // gpu
        at::Tensor gather_kv_gpu_buffer,  // gpu
        at::Tensor new_offload_startpos,  // host
        at::Tensor new_offload_lengths)   // host
    {
        {
            std::unique_lock<std::mutex> lock(offload_task_mutex_);
            offload_task_queue.push(std::make_tuple(
                offload_user_ids, 
                offload_page_ids, 
                gather_kv_gpu_buffer, 
                new_offload_startpos, 
                new_offload_lengths, 
                offload_handle.ready_event
            ));
        }
        offload_task_cv_.notify_one();
    };

    bool is_busy_offloading() {
        return !offload_task_queue.empty() || this->offload_busy_.load();
    }

private:
    void onload_host_memcpy(void* dst, void* src, size_t bytes, size_t bytes_part) {
        {
            std::unique_lock<std::mutex> lock(onload_memcpy_task_mtx_);
            this->onload_memcpy_cmplt_flag = 0;
            for (int i = 1; i < (this->num_onload_memcpy_worker + 1); i++)
                onload_memcpy_task_queue.push(std::make_tuple(
                    (reinterpret_cast<char *>(dst) + i * bytes_part),
                    (reinterpret_cast<char *>(src) + i * bytes_part),
                    bytes_part
                ));
        }
        onload_memcpy_task_cv_.notify_all();
        std::memcpy(dst, src, bytes_part);

        {
            std::unique_lock<std::mutex> lock(onload_memcpy_cmplt_mtx_);
            onload_memcpy_cmplt_cv_.wait(lock, [this] {
                return this->onload_memcpy_cmplt_flag == this->num_onload_memcpy_worker || this->terminate_;
            });
        }
    }

    void onload_host_memcpy_loop() {
        while (true) {
            void *dst, *src;
            size_t bytes;
            {
                std::unique_lock<std::mutex> lock(onload_memcpy_task_mtx_);
                onload_memcpy_task_cv_.wait(lock, [this]{
                    return !this->onload_memcpy_task_queue.empty() || this->terminate_;
                });
                if (terminate_) return;
                std::tie(dst, src, bytes) = onload_memcpy_task_queue.front();
                onload_memcpy_task_queue.pop();
            }
            std::memcpy(dst, src, bytes);
            {
                std::unique_lock<std::mutex> lock(onload_memcpy_cmplt_mtx_);
                onload_memcpy_cmplt_flag += 1;
                if (onload_memcpy_cmplt_flag == this->num_onload_memcpy_worker)
                    onload_memcpy_cmplt_cv_.notify_one();
            }
        }
    }

    void offload_host_memcpy(void* dst, void* src, size_t bytes, size_t bytes_part) {
        {
            std::unique_lock<std::mutex> lock(offload_memcpy_task_mtx_);
            this->offload_memcpy_cmplt_flag = 0;
            for (int i = 1; i < (this->num_offload_memcpy_worker + 1); i++)
                offload_memcpy_task_queue.push(std::make_tuple(
                    (reinterpret_cast<char *>(dst) + i * bytes_part),
                    (reinterpret_cast<char *>(src) + i * bytes_part),
                    bytes_part
                ));
        }
        offload_memcpy_task_cv_.notify_all();
        std::memcpy(dst, src, bytes_part);

        {
            std::unique_lock<std::mutex> lock(offload_memcpy_cmplt_mtx_);
            offload_memcpy_cmplt_cv_.wait(lock, [this] {
                return this->offload_memcpy_cmplt_flag == this->num_offload_memcpy_worker || this->terminate_;
            });
        }
    }

    void offload_host_memcpy_loop() {
        while (true) {
            void *dst, *src;
            size_t bytes;
            {
                std::unique_lock<std::mutex> lock(offload_memcpy_task_mtx_);
                offload_memcpy_task_cv_.wait(lock, [this]{
                    return !this->offload_memcpy_task_queue.empty() || this->terminate_;
                });
                if (terminate_) return;
                std::tie(dst, src, bytes) = offload_memcpy_task_queue.front();
                offload_memcpy_task_queue.pop();
            }
            std::memcpy(dst, src, bytes);
            {
                std::unique_lock<std::mutex> lock(offload_memcpy_cmplt_mtx_);
                offload_memcpy_cmplt_flag += 1;
                if (offload_memcpy_cmplt_flag == this->num_offload_memcpy_worker)
                    offload_memcpy_cmplt_cv_.notify_one();
            }
        }
    }

    void offload_loop()
    {
        const c10::cuda::OptionalCUDAGuard device_guard(this->device);

        while (true) {
            
            at::Tensor offload_user_ids, offload_page_ids, gather_kv_gpu_buffer, new_offload_startpos, new_offload_lengths;
            cudaEvent_t offload_gpu_acq_event;

            {
                std::unique_lock<std::mutex> lock(offload_task_mutex_);
                offload_task_cv_.wait(lock, [this] {
                    return !offload_task_queue.empty() || this->terminate_;
                });
                if (this->terminate_) {
                    return;
                }

                std::tie(
                    offload_user_ids, offload_page_ids, gather_kv_gpu_buffer, new_offload_startpos, new_offload_lengths, offload_gpu_acq_event
                ) = offload_task_queue.front();
                offload_task_queue.pop();

                this->offload_busy_.store(true);
            }
            uint16_t *gather_kv_gpu_buffer_data_ptr = static_cast<uint16_t*>(gather_kv_gpu_buffer.data_ptr());
            int num_offload_pages = offload_page_ids.numel();
            size_t gather_layer_stride = num_offload_pages * this->page_stride;

            // std::cout << "[Offload Launch]: waiting event" << std::endl;
            cudaStreamWaitEvent(this->offload_stream, offload_gpu_acq_event);
            cudaEventDestroy(offload_gpu_acq_event);
            // std::cout << "[Offload Launch]: waited event" << std::endl;

            // gather
            gather_paged_kv_cache_all_layers(
                gather_kv_gpu_buffer_data_ptr,
                this->get_cache_table(),
                static_cast<int*>(offload_page_ids.data_ptr()),
                this->num_layers,
                gather_layer_stride,
                this->layer_stride,
                this->num_kv_heads,
                this->kv_headdim,
                this->num_tokens_per_page,
                this->page_stride,
                this->k2v_stride,
                this->num_kv_heads * this->kv_headdim,
                this->kv_headdim,
                num_offload_pages,
                this->offload_stream);
            // release on gpu kvcache
            // skipped

            size_t pinned_bytes = this->num_layers * gather_layer_stride * sizeof(uint16_t);
            uint16_t *host_kv_ptr = static_cast<uint16_t *>(aligned_alloc(sysconf(_SC_PAGESIZE), pinned_bytes));

            size_t chunk_numel_part = host_kv_mgr->chunk_numel / (this->num_offload_memcpy_worker + 1);

            int num_chunks = (num_offload_pages * this->num_tokens_per_page) / this->num_tokens_per_chunk * this->num_layers;
            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                cudaMemcpyAsync(
                    offload_pinned_buffers[chunk_idx%2],
                    gather_kv_gpu_buffer_data_ptr + chunk_idx * host_kv_mgr->chunk_numel,
                    host_kv_mgr->chunk_numel * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost, 
                    this->offload_stream);
                cudaEventRecord(offload_memcpy_event[chunk_idx%2], this->offload_stream);

                if (chunk_idx > 0) {
                    cudaEventSynchronize(offload_memcpy_event[(chunk_idx-1)%2]);
                    offload_host_memcpy(
                        host_kv_ptr + (chunk_idx-1) * host_kv_mgr->chunk_numel,
                        offload_pinned_buffers[(chunk_idx-1)%2],
                        host_kv_mgr->chunk_numel * sizeof(uint16_t),
                        chunk_numel_part * sizeof(uint16_t));
                }
            }
            {
                cudaEventSynchronize(offload_memcpy_event[(num_chunks-1)%2]);
                offload_host_memcpy(
                    host_kv_ptr + (num_chunks-1) * host_kv_mgr->chunk_numel,
                    offload_pinned_buffers[(num_chunks-1)%2],
                    host_kv_mgr->chunk_numel * sizeof(uint16_t),
                    chunk_numel_part * sizeof(uint16_t));
            }

            // bookkeepping int host kv storage
            size_t page_offset = 0;
            int* offload_startpos = static_cast<int *>(new_offload_startpos.data_ptr());
            int* offload_lengths = static_cast<int *>(new_offload_lengths.data_ptr());

            const int offload_batch_size = offload_user_ids.numel();
            for (int seq_idx = 0; seq_idx < offload_batch_size; seq_idx++) {
                int64_t uid = static_cast<int64_t*>(offload_user_ids.data_ptr())[seq_idx];
                uint16_t *input_ptr = host_kv_ptr + page_offset * this->page_stride;
                host_kv_mgr->append_kvdata_v2(uid, offload_startpos[seq_idx], offload_lengths[seq_idx], input_ptr, gather_layer_stride);
                this->_uid_to_offloaded_length[uid] = offload_startpos[seq_idx] + offload_lengths[seq_idx];
                page_offset += offload_lengths[seq_idx] / this->num_tokens_per_page;
            }

            this->offload_busy_.store(false);
            // std::cout << "offload kvcache finish one" << std::endl << std::flush;
        }
    }

public:
        int num_layers;
        int num_kv_heads;
        int kv_headdim;
        int num_tokens_per_page;
        int num_primary_cache_pages;
        int num_onload_buffer_pages;
        int num_reserved_buffer_pages;
        int num_tokens_per_chunk;
        int max_num_sequences;
        int max_sequence_length;

        size_t layer_stride;
        size_t k2v_stride;
        size_t page_stride;
        size_t per_token_kv_stride;

    public:
        std::list<int64_t> _lru_list;
        std::unordered_map<int64_t, 
                        typename std::list<int64_t>::iterator> _lru_lookup_table;
        std::queue<int64_t> _empty_pages;
        std::unordered_map<int64_t, std::vector<int32_t>> _uid_to_page_id;
        std::unordered_map<int64_t, int32_t> _uid_to_paged_cache_startpos;
        std::unordered_map<int64_t, int32_t> _uid_to_paged_cache_length;
        std::unordered_map<int64_t, int32_t> _uid_to_offloaded_length;

    int num_onload_memcpy_worker;
    std::vector<std::thread> onload_memcpy_worker;
    std::vector<uint16_t*> onload_pinned_buffers;
    std::vector<cudaEvent_t> onload_memcpy_event;

    std::queue<std::tuple<void*, void*, size_t>> onload_memcpy_task_queue;
    std::mutex onload_memcpy_task_mtx_;
    std::condition_variable onload_memcpy_task_cv_;

    int onload_memcpy_cmplt_flag;
    std::mutex onload_memcpy_cmplt_mtx_;
    std::condition_variable onload_memcpy_cmplt_cv_;

    std::thread offload_worker;

    std::queue<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, cudaEvent_t>> offload_task_queue;
    std::mutex offload_task_mutex_;
    std::condition_variable offload_task_cv_;

    int num_offload_memcpy_worker;
    std::vector<std::thread> offload_memcpy_worker;
    std::vector<uint16_t*> offload_pinned_buffers;
    std::vector<cudaEvent_t> offload_memcpy_event;

    std::queue<std::tuple<void*, void*, size_t>> offload_memcpy_task_queue;
    std::mutex offload_memcpy_task_mtx_;
    std::condition_variable offload_memcpy_task_cv_;

    int offload_memcpy_cmplt_flag;
    std::mutex offload_memcpy_cmplt_mtx_;
    std::condition_variable offload_memcpy_cmplt_cv_;

    bool terminate_;
    std::atomic<bool> offload_busy_;

    cudaStream_t worker_stream;
    cudaStream_t onload_stream;
    cudaStream_t offload_stream;
    cudaStream_t sync_stream;

    HostKVStorageImpl *host_kv_mgr;

public:
    uint16_t *cache_table;
    c10::Device device;
};


void prepare_kvcache(
    GPUKVCacheMangerImpl& gpu_mgr,
    HostKVStorageImpl& host_mgr,
    std::vector<int64_t>& user_ids,
    std::vector<int64_t>& total_hist_lens, // all histo w/o candi
    at::Tensor page_ids_gpu_buffer,
    at::Tensor offload_page_ids_gpu_buffer,
    at::Tensor offload_uids_buffer,
    at::Tensor metadata_host_buffer,
    at::Tensor metadata_gpu_buffer) {

        const c10::cuda::OptionalCUDAGuard device_guard(gpu_mgr.device);

        // std::cout << "prepare_kvcache start" << std::endl << std::flush;

        int batch_size = user_ids.size();

        std::vector<int32_t> old_history_lengths = gpu_mgr.get_total_cache_length(user_ids);

        std::vector<int> page_indices;
        std::vector<int> offload_page_ids;
        int64_t *offload_user_ids = static_cast<int64_t*>(offload_uids_buffer.data_ptr());

        int *host_bufptr = static_cast<int*>(metadata_host_buffer.data_ptr());

        int *page_indptr = host_bufptr + 0;
        int *last_page_len = host_bufptr + batch_size + 1;
        int *total_history_lengths = host_bufptr + batch_size * 2 + 1;
        int *total_history_offsets = host_bufptr + batch_size * 3 + 1;
        int *new_history_nnz_cuda = host_bufptr + batch_size * 4 + 2;
        int *new_history_offsets = host_bufptr + batch_size * 4 + 3;
        // === ^ GPU === v Host ===
        int *new_offload_startpos = host_bufptr + batch_size * 5 + 4;
        int *new_offload_lengths = host_bufptr + batch_size * 6 + 4;

        int *num_page_ids = host_bufptr + batch_size * 7 + 4;
        int *num_offload_page_ids = host_bufptr + batch_size * 7 + 5;
        int *num_offload_user_ids = host_bufptr + batch_size * 7 + 6;

        size_t onload_page_offset = gpu_mgr.num_primary_cache_pages;
        size_t num_offload_uids = 0;
        size_t num_offload_pages = 0;

        page_indptr[0] = 0;
        total_history_offsets[0] = 0;
        new_history_offsets[0] = 0;

        const std::unordered_set<int64_t> freezed_uids(user_ids.begin(), user_ids.end());
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            int64_t uid = user_ids[seq_idx];
            int total_history_length = total_hist_lens[seq_idx];
            // std::cout << "[DEBUG] User " << uid << " total_history_length=" << total_history_length << std::endl;

            std::vector<int32_t>& page_ids = gpu_mgr.alloc(uid, total_history_length, freezed_uids);
            int gpu_cache_startpos = gpu_mgr._uid_to_paged_cache_startpos[uid];
            int gpu_cache_length = gpu_mgr._uid_to_paged_cache_length[uid];
            // std::cout << "[DEBUG] User " << uid << " gpu_cache_startpos=" << gpu_cache_startpos
            //       << " gpu_cache_length=" << gpu_cache_length
            //       << " page_ids.size=" << page_ids.size() << std::endl;

            int num_onload_pages = gpu_cache_startpos / gpu_mgr.num_tokens_per_page;
            // std::cout << "[DEBUG] User " << uid << " num_onload_pages=" << num_onload_pages << std::endl;
            for (int i = 0; i < num_onload_pages; i++) page_indices.push_back(onload_page_offset+i);
            page_indices.insert(page_indices.end(), page_ids.begin(), page_ids.end());
            page_indptr[seq_idx + 1] = page_indptr[seq_idx] + page_ids.size();
            last_page_len[seq_idx] = gpu_cache_length % gpu_mgr.num_tokens_per_page;
            onload_page_offset += num_onload_pages;

            total_history_lengths[seq_idx] = total_history_length;
            total_history_offsets[seq_idx + 1] = total_history_offsets[seq_idx] + total_history_length;
            new_history_offsets[seq_idx + 1] = new_history_offsets[seq_idx] + total_history_length - old_history_lengths[seq_idx];
            
            // // 直接用 page_ids，顺序已保证
            // page_indices.insert(page_indices.end(), page_ids.begin(), page_ids.end());
            // page_indptr[seq_idx + 1] = page_indptr[seq_idx] + page_ids.size();
            // last_page_len[seq_idx] = gpu_cache_length % gpu_mgr.num_tokens_per_page;
            // total_history_lengths[seq_idx] = total_history_length;
            // total_history_offsets[seq_idx + 1] = total_history_offsets[seq_idx] + total_history_length;
            // new_history_offsets[seq_idx + 1] = new_history_offsets[seq_idx] + total_history_length - old_history_lengths[seq_idx];
            

            auto offloaded_length = 0;
            if (gpu_mgr._uid_to_offloaded_length.find(uid) != gpu_mgr._uid_to_offloaded_length.end())
                offloaded_length = gpu_mgr._uid_to_offloaded_length[uid];
            if (total_history_length - offloaded_length >= gpu_mgr.num_tokens_per_chunk) {
                auto chunked_length = total_history_length - total_history_length % gpu_mgr.num_tokens_per_chunk;
                auto new_offload_page_start = (offloaded_length - gpu_cache_startpos) / gpu_mgr.num_tokens_per_page;
                // std::cout << "[DEBUG] User " << uid << " offload: offloaded_length=" << offloaded_length
                //       << " chunked_length=" << chunked_length
                //       << " new_offload_page_start=" << new_offload_page_start << std::endl;

                offload_user_ids[num_offload_uids] = uid;
                new_offload_startpos[num_offload_uids] = offloaded_length;
                new_offload_lengths[num_offload_uids] = chunked_length - offloaded_length;
                auto num_pages = new_offload_lengths[num_offload_uids] / gpu_mgr.num_tokens_per_page;
                offload_page_ids.insert(offload_page_ids.end(), page_ids.begin() + new_offload_page_start, page_ids.begin() + new_offload_page_start + num_pages);
                
                num_offload_uids += 1;
                num_offload_pages += num_pages;
            }
        }

        auto new_tokens = new_history_offsets[batch_size];
        *new_history_nnz_cuda = new_tokens;
        *num_page_ids = page_indptr[batch_size];
        *num_offload_page_ids = num_offload_pages;
        *num_offload_user_ids = num_offload_uids;
        // std::cout << "num_offload_pages: " << num_offload_pages << std::endl << std::flush;
        // std::cout << "[DEBUG] new_tokens=" << new_tokens
        //     << " num_page_ids=" << *num_page_ids
        //     << " num_offload_page_ids=" << *num_offload_page_ids
        //     << " num_offload_user_ids=" << *num_offload_user_ids << std::endl;

        // std::cout << "[DEBUG] cudaMemcpyAsync page_ids_gpu_buffer, size=" << page_indptr[batch_size] << std::endl;
        // std::cout << "[DEBUG] cudaMemcpyAsync offload_page_ids_gpu_buffer, size=" << num_offload_pages << std::endl;

        cudaMemcpyAsync(page_ids_gpu_buffer.data_ptr(), page_indices.data(), page_indptr[batch_size] * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_mgr.worker_stream);
        cudaMemcpyAsync(offload_page_ids_gpu_buffer.data_ptr(), offload_page_ids.data(), num_offload_pages * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_mgr.worker_stream);

        size_t host_buffer_d2h_size = (batch_size * 5 + 4) * sizeof(int32_t);
        cudaMemcpyAsync(metadata_gpu_buffer.data_ptr(), metadata_host_buffer.data_ptr(), host_buffer_d2h_size, cudaMemcpyHostToDevice, gpu_mgr.worker_stream);
        
        int *gpu_bufptr = static_cast<int*>(metadata_gpu_buffer.data_ptr());

        int *total_history_lengths_dev = gpu_bufptr + batch_size * 2 + 1;
        int *new_history_offsets_dev = gpu_bufptr + batch_size * 4 + 3;
        int *batch_indices_dev = gpu_bufptr + batch_size * 5 + 4;
        int *position_dev = gpu_bufptr + batch_size * 5 + 4 + new_tokens;

        GetPagedBatchIndicesPositions(
            batch_size,
            new_history_offsets_dev,
            total_history_lengths_dev,
            batch_indices_dev,
            position_dev,
            gpu_mgr.worker_stream
        );

        cudaStreamSynchronize(gpu_mgr.worker_stream);
        // std::cout << "prepare_kvcache stop" << std::endl << std::flush;
    }


    // void onload_kvcache(
    //     GPUKVCacheMangerImpl& gpu_mgr,
    //     HostKVStorageImpl& host_mgr,
    //     std::vector<int64_t>& user_ids, 
    //     std::vector<at::Tensor>& host_pinned_kv_buffer, 
    //     KVOnloadHandle& onloadhandle) {
    //     // std::cout << "onload_kvcache start" << std::endl << std::flush;
    //     const int batch_size = user_ids.size();

    //     std::vector<size_t> onload_length(batch_size);
    //     std::vector<size_t> onload_offsets(batch_size + 1);
    //     onload_offsets[0] = 0;
    //     for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
    //         auto uid = user_ids[seq_idx];
    //         if (gpu_mgr._uid_to_paged_cache_startpos.find(uid) != gpu_mgr._uid_to_paged_cache_startpos.end())
    //             onload_length[seq_idx] = gpu_mgr._uid_to_paged_cache_startpos[uid];
    //         else if (gpu_mgr._uid_to_offloaded_length.find(uid) != gpu_mgr._uid_to_offloaded_length.end())
    //             onload_length[seq_idx] = gpu_mgr._uid_to_offloaded_length[uid];
    //         else
    //             onload_length[seq_idx] = 0;

    //         onload_offsets[seq_idx + 1] = onload_offsets[seq_idx] + onload_length[seq_idx];
    //     }
    //     size_t total_onload_length = onload_offsets[batch_size];
    //     if (total_onload_length == 0) return;

    //     for (int layer_idx = 0; layer_idx < gpu_mgr.num_layers; layer_idx++) {
    //         uint16_t *pinned_onload_buffer = static_cast<uint16_t*>(host_pinned_kv_buffer[layer_idx].data_ptr());
    //         uint16_t *gpu_onload_buffer = gpu_mgr.get_cache_table_by_layer(layer_idx) + gpu_mgr.num_primary_cache_pages * gpu_mgr.page_stride;

    //         for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
    //             host_mgr.get_kvdata(user_ids[seq_idx], onload_length[seq_idx], layer_idx, pinned_onload_buffer + onload_offsets[seq_idx] * gpu_mgr.per_token_kv_stride);
    //         }

    //         // // 测量 GPU 内存拷贝带宽
    //         // size_t gpu_copy_bytes = total_onload_length * gpu_mgr.per_token_kv_stride * sizeof(uint16_t);
    //         // auto gpu_copy_start = std::chrono::high_resolution_clock::now();
            
    //         // cudaMemcpyAsync(gpu_onload_buffer, pinned_onload_buffer, gpu_copy_bytes, 
    //         //                cudaMemcpyHostToDevice, gpu_mgr.onload_stream);
            
    //         // // 强制同步以测量准确时间
    //         // cudaStreamSynchronize(gpu_mgr.onload_stream);
            
    //         // auto gpu_copy_end = std::chrono::high_resolution_clock::now();
    //         // auto gpu_copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_copy_end - gpu_copy_start).count();
            
    //         // double gpu_copy_bandwidth = (gpu_copy_bytes / (1024.0 * 1024.0)) / (gpu_copy_duration / 1000000.0); // MB/s
    //         // std::cout << "DEBUG: Layer " << layer_idx << " GPU Copy Bandwidth: " << gpu_copy_bandwidth << " MB/s" << std::endl;
    //         cudaMemcpyAsync(gpu_onload_buffer, pinned_onload_buffer, total_onload_length * gpu_mgr.per_token_kv_stride * sizeof(uint16_t), cudaMemcpyHostToDevice, gpu_mgr.onload_stream);
    //         onloadhandle.record(layer_idx, gpu_mgr.onload_stream);
    //     }
    //     // std::cout << "onload_kvcache end" << std::endl << std::flush;
    // };

    // void offload_kvcache(
    //     GPUKVCacheMangerImpl& gpu_mgr,
    //     HostKVStorageImpl& host_mgr, 
    //     KVOffloadHandle& offload_handle,
    //     // int num_offload_pages,
    //     at::Tensor offload_user_ids,      // host
    //     at::Tensor offload_page_ids,      // gpu
    //     at::Tensor gather_kv_gpu_buffer,  // gpu
    //     // at::Tensor pinned_input_kvdata,   // host
    //     at::Tensor new_offload_startpos,  // host
    //     at::Tensor new_offload_lengths)   // host
    // {
    //     size_t num_offload_pages = offload_page_ids.numel();
    //     if (num_offload_pages == 0) return;

    //     // std::cout << "offload kvcache start" << std::endl << std::flush;
    //     offload_handle.wait_ready(gpu_mgr.offload_stream);
    //     size_t gather_layer_stride = num_offload_pages * gpu_mgr.page_stride;

    //     // gather
    //     gather_paged_kv_cache_all_layers(
    //                         static_cast<uint16_t*>(gather_kv_gpu_buffer.data_ptr()),
    //                         gpu_mgr.get_cache_table(),
    //                         static_cast<int*>(offload_page_ids.data_ptr()),
    //                         gpu_mgr.num_layers,
    //                         gather_layer_stride,
    //                         gpu_mgr.layer_stride,
    //                         gpu_mgr.num_kv_heads,
    //                         gpu_mgr.kv_headdim,
    //                         gpu_mgr.num_tokens_per_page,
    //                         gpu_mgr.page_stride,
    //                         gpu_mgr.k2v_stride,
    //                         gpu_mgr.num_kv_heads * gpu_mgr.kv_headdim,
    //                         gpu_mgr.kv_headdim,
    //                         num_offload_pages,
    //                         gpu_mgr.offload_stream);
        
    //     uint16_t *pinned_input_kvdata_ptr;
    //     cudaMallocHost((void**)(&pinned_input_kvdata_ptr), gpu_mgr.num_layers * gather_layer_stride * sizeof(uint16_t));
    //     // d2h
    //     cudaMemcpyAsync(
    //         pinned_input_kvdata_ptr,  // [#layers, #num_offload_page, 2, page_size, num_heads, headdim]
    //         gather_kv_gpu_buffer.data_ptr(), 
    //         gpu_mgr.num_layers * gather_layer_stride * sizeof(uint16_t),
    //         cudaMemcpyDeviceToHost, 
    //         gpu_mgr.offload_stream);
    //     cudaStreamSynchronize(gpu_mgr.offload_stream);
    //     // std::cout << "offload kvcache gpu cache release" << std::endl << std::flush;

    //     // append to host
    //     const int batch_size = offload_user_ids.numel();
    //     size_t page_offset = 0;
    //     int* _offload_startpos = static_cast<int *>(new_offload_startpos.data_ptr());
    //     int* _offload_lengths = static_cast<int *>(new_offload_lengths.data_ptr());
    //     for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
    //         int64_t uid = static_cast<int64_t*>(offload_user_ids.data_ptr())[seq_idx];
    //         uint16_t *input_ptr = pinned_input_kvdata_ptr + page_offset * gpu_mgr.page_stride;
    //         host_mgr.append_kvdata(uid, _offload_startpos[seq_idx], _offload_lengths[seq_idx], input_ptr, gather_layer_stride);
    //         gpu_mgr._uid_to_offloaded_length[uid] = _offload_startpos[seq_idx] + _offload_lengths[seq_idx];
    //         page_offset += _offload_lengths[seq_idx] / gpu_mgr.num_tokens_per_page;
    //     }
    //     cudaFreeHost(pinned_input_kvdata_ptr);

    //     // std::cout << "offload kvcache end" << std::endl << std::flush;
    // };

    // void sync_onload_buffer_to_cache(
    //     GPUKVCacheMangerImpl& gpu_mgr,
    //     std::vector<int64_t>& user_ids
    // )
    // {
    //     int batch_size = user_ids.size();
    //     std::unordered_set<int64_t> freezed_uids(user_ids.begin(), user_ids.end());
    //     // 1. 计算每个用户的onload长度（即分配前的startpos）
    //     std::vector<size_t> onload_lengths(batch_size);
    //     std::vector<size_t> onload_offsets(batch_size + 1, 0);

    //     // // 打印初始状态
    //     // std::cout << "[DEBUG] sync_onload_buffer_to_cache: before" << std::endl;
    //     // for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    //     //     int64_t uid = user_ids[seq_idx];
    //     //     int32_t start = 0, len = 0;
    //     //     if (gpu_mgr._uid_to_paged_cache_startpos.find(uid) != gpu_mgr._uid_to_paged_cache_startpos.end())
    //     //         start = gpu_mgr._uid_to_paged_cache_startpos[uid];
    //     //     if (gpu_mgr._uid_to_paged_cache_length.find(uid) != gpu_mgr._uid_to_paged_cache_length.end())
    //     //         len = gpu_mgr._uid_to_paged_cache_length[uid];
    //     //     std::cout << "  [uid=" << uid << "] gpu start=" << start << " length=" << len << std::endl;
    //     // }

    //     for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    //         int64_t uid = user_ids[seq_idx];
    //         if (gpu_mgr._uid_to_paged_cache_startpos.find(uid) != gpu_mgr._uid_to_paged_cache_startpos.end())
    //             onload_lengths[seq_idx] = gpu_mgr._uid_to_paged_cache_startpos[uid];
    //         else if (gpu_mgr._uid_to_offloaded_length.find(uid) != gpu_mgr._uid_to_offloaded_length.end())
    //             onload_lengths[seq_idx] = gpu_mgr._uid_to_offloaded_length[uid];
    //         else
    //             onload_lengths[seq_idx] = 0;
    //         onload_offsets[seq_idx + 1] = onload_offsets[seq_idx] + onload_lengths[seq_idx];
    //     }
    //     size_t total_onload_length = onload_offsets[batch_size];
    //     if (total_onload_length == 0) return;
    //     // if (total_onload_length == 0) {
    //     //     std::cout << "[DEBUG] sync_onload_buffer_to_cache: total_onload_length==0, return" << std::endl;
    //     //     return;
    //     // }

    //     for (int layer_idx = 0; layer_idx < gpu_mgr.num_layers; ++layer_idx) {
    //         uint16_t* gpu_onload_buffer = gpu_mgr.get_cache_table_by_layer(layer_idx) + gpu_mgr.num_primary_cache_pages * gpu_mgr.page_stride;
    //         for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    //             int64_t uid = user_ids[seq_idx];
    //             size_t len = onload_lengths[seq_idx];
    //             if (len == 0) continue;
    //             int num_pages = len / gpu_mgr.num_tokens_per_page;

    //             if (layer_idx == 0) {
    //                 std::vector<int32_t> onload_page_ids;
    //                 // std::cout << "[DEBUG] [uid=" << uid << "] layer=0, need onload len=" << len << " num_pages=" << num_pages << std::endl;
    //                 // std::cout << "[DEBUG] empty_pages.size()=" << gpu_mgr._empty_pages.size() << std::endl; 
    //                 while ((size_t)num_pages > gpu_mgr._empty_pages.size()) {
    //                     int64_t uid_to_evict = gpu_mgr.getUIdToEvict(freezed_uids);
    //                     // std::cout << "[DEBUG] [uid=" << uid << "] evict uid=" << uid_to_evict << "page_id.size=" << gpu_mgr._uid_to_page_id[uid_to_evict].size()<< std::endl;
    //                     gpu_mgr.evict(uid_to_evict);
    //                 }
    //                 for (int i = 0; i < num_pages; ++i) {
    //                     onload_page_ids.push_back(gpu_mgr._empty_pages.front());
    //                     gpu_mgr._empty_pages.pop();
    //                 }
    //                 // std::cout << "[DEBUG] [uid=" << uid << "] onload_page_ids: ";
    //                 // for (auto pid : onload_page_ids) std::cout << pid << " ";
    //                 // std::cout << std::endl;
    //                 gpu_mgr._uid_to_page_id[uid].insert(gpu_mgr._uid_to_page_id[uid].begin(), onload_page_ids.begin(), onload_page_ids.end());
    //                 // std::cout << "[DEBUG] [uid=" << uid << "] page_id.size(after)=" << gpu_mgr._uid_to_page_id[uid].size() << std::endl;
    //                 // std::cout << "[DEBUG] [uid=" << uid << "] meta before: start=" << gpu_mgr._uid_to_paged_cache_startpos[uid]
    //                         // << " length=" << gpu_mgr._uid_to_paged_cache_length[uid] << std::endl;
    //                 gpu_mgr._uid_to_paged_cache_startpos[uid] = 0;
    //                 gpu_mgr._uid_to_paged_cache_length[uid] += len;
    //                 // std::cout << "[DEBUG] [uid=" << uid << "] meta after: start=" << gpu_mgr._uid_to_paged_cache_startpos[uid]
    //                         // << " length=" << gpu_mgr._uid_to_paged_cache_length[uid] << std::endl;
    //             }

    //             // 逐页拷贝，每层都要
    //             for (int page_i = 0; page_i < num_pages; ++page_i) {
    //                 uint16_t* src = gpu_onload_buffer + (onload_offsets[seq_idx] / gpu_mgr.num_tokens_per_page + page_i) * gpu_mgr.page_stride;
    //                 uint16_t* dst = gpu_mgr.get_cache_table_by_layer(layer_idx) + gpu_mgr._uid_to_page_id[uid][page_i] * gpu_mgr.page_stride;
    //                 cudaMemcpyAsync(dst, src, gpu_mgr.page_stride * sizeof(uint16_t), cudaMemcpyDeviceToDevice, gpu_mgr.sync_stream);
    //             }
    //         }
    //     }
    //     cudaStreamSynchronize(gpu_mgr.sync_stream);
    //     // // 打印最终状态
    //     // std::cout << "[DEBUG] sync_onload_buffer_to_cache: after" << std::endl;
    //     // for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    //     //     int64_t uid = user_ids[seq_idx];
    //     //     int32_t start = 0, len = 0;
    //     //     if (gpu_mgr._uid_to_paged_cache_startpos.find(uid) != gpu_mgr._uid_to_paged_cache_startpos.end())
    //     //         start = gpu_mgr._uid_to_paged_cache_startpos[uid];
    //     //     if (gpu_mgr._uid_to_paged_cache_length.find(uid) != gpu_mgr._uid_to_paged_cache_length.end())
    //     //         len = gpu_mgr._uid_to_paged_cache_length[uid];
    //     //         size_t page_count = 0;
    //     //     if (gpu_mgr._uid_to_page_id.find(uid) != gpu_mgr._uid_to_page_id.end())
    //     //         page_count = gpu_mgr._uid_to_page_id[uid].size();
    //     //     std::cout << "  [uid=" << uid << "] gpu start=" << start
    //     //             << " length=" << len
    //     //             << " page_count=" << page_count 
    //     //             // << " page=" << gpu_mgr._uid_to_page_id[uid] 
    //     //             << std::endl;
    //     // }
    // };
    void sync_onload_buffer_to_cache(
        GPUKVCacheMangerImpl& gpu_mgr,
        std::vector<int64_t>& user_ids
    )
    {
        int batch_size = user_ids.size();
        std::unordered_set<int64_t> freezed_uids(user_ids.begin(), user_ids.end());
        std::vector<size_t> onload_lengths(batch_size);
        std::vector<size_t> onload_offsets(batch_size + 1, 0);

        for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
            int64_t uid = user_ids[seq_idx];
            if (gpu_mgr._uid_to_paged_cache_startpos.find(uid) != gpu_mgr._uid_to_paged_cache_startpos.end())
                onload_lengths[seq_idx] = gpu_mgr._uid_to_paged_cache_startpos[uid];
            else if (gpu_mgr._uid_to_offloaded_length.find(uid) != gpu_mgr._uid_to_offloaded_length.end())
                onload_lengths[seq_idx] = gpu_mgr._uid_to_offloaded_length[uid];
            else
                onload_lengths[seq_idx] = 0;
            onload_offsets[seq_idx + 1] = onload_offsets[seq_idx] + onload_lengths[seq_idx];
        }
        size_t total_onload_length = onload_offsets[batch_size];
        if (total_onload_length == 0) return;

        for (int layer_idx = 0; layer_idx < gpu_mgr.num_layers; ++layer_idx) {
            uint16_t* gpu_onload_buffer = gpu_mgr.get_cache_table_by_layer(layer_idx) + gpu_mgr.num_primary_cache_pages * gpu_mgr.page_stride;
            for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
                int64_t uid = user_ids[seq_idx];
                size_t len = onload_lengths[seq_idx];
                if (len == 0) continue;
                int num_pages = len / gpu_mgr.num_tokens_per_page;

                if (layer_idx == 0) {
                    std::vector<int32_t> onload_page_ids;
                    while ((size_t)num_pages > gpu_mgr._empty_pages.size()) {
                        int64_t uid_to_evict = gpu_mgr.getUIdToEvict(freezed_uids);
                        gpu_mgr.evict(uid_to_evict);
                    }
                    for (int i = 0; i < num_pages; ++i) {
                        onload_page_ids.push_back(gpu_mgr._empty_pages.front());
                        gpu_mgr._empty_pages.pop();
                    }

                    gpu_mgr._uid_to_page_id[uid].insert(gpu_mgr._uid_to_page_id[uid].begin(), onload_page_ids.begin(), onload_page_ids.end());

                    gpu_mgr._uid_to_paged_cache_startpos[uid] = 0;
                    gpu_mgr._uid_to_paged_cache_length[uid] += len;
                }

                for (int page_i = 0; page_i < num_pages; ++page_i) {
                    uint16_t* src = gpu_onload_buffer + (onload_offsets[seq_idx] / gpu_mgr.num_tokens_per_page + page_i) * gpu_mgr.page_stride;
                    uint16_t* dst = gpu_mgr.get_cache_table_by_layer(layer_idx) + gpu_mgr._uid_to_page_id[uid][page_i] * gpu_mgr.page_stride;
                    cudaMemcpyAsync(dst, src, gpu_mgr.page_stride * sizeof(uint16_t), cudaMemcpyDeviceToDevice, gpu_mgr.sync_stream);
                }
            }
        }
        // cudaStreamSynchronize(gpu_mgr.sync_stream);
    };
    
    void synchronize_sync_stream(
        GPUKVCacheMangerImpl& gpu_mgr
    ) 
    {
        cudaStreamSynchronize(gpu_mgr.sync_stream);
    };

}

PYBIND11_MODULE(paged_kvcache_ops, m) {
  m.def("append_kvcache", &append_paged_kv_cache, "append paged kv cache on GPU", py::call_guard<py::gil_scoped_release>());
  m.def("gather_kvcache", &gather_paged_kv_cache, "gather paged kv cache on GPU", py::call_guard<py::gil_scoped_release>());

  py::class_<kvcache::HostKVStorageImpl>(m, "HostKVStorageImpl")
    .def(py::init<int, int, int, int, int64_t>(), 
         py::arg("num_layers"),
         py::arg("num_kv_heads"),
         py::arg("kv_headdim"),
         py::arg("num_tokens_per_page"),
         py::arg("num_tokens_per_chunk"))
    .def("get_cache_startpos_and_length", &kvcache::HostKVStorageImpl::get_cache_startpos_and_length)
  ;

  py::class_<kvcache::GPUKVCacheMangerImpl>(m, "GPUKVCacheMangerImpl")
    .def(py::init<int, int, int, int, int, int, int, int, int, int, at::Tensor, kvcache::HostKVStorageImpl&>(),
         py::arg("num_layers"),
         py::arg("num_kv_heads"),
         py::arg("kv_headdim"),
         py::arg("num_tokens_per_page"),
         py::arg("num_primary_cache_pages"),
         py::arg("num_onload_buffer_pages"),
         py::arg("num_reserved_buffer_pages"),
         py::arg("num_tokens_per_chunk"), 
         py::arg("max_num_sequences"),
         py::arg("max_sequence_length"),
         py::arg("cache_table"),
         py::arg("host_kv_mgr"))
    .def("get_total_cache_length", &kvcache::GPUKVCacheMangerImpl::get_total_cache_length)
    .def("get_cache_startpos_and_length", &kvcache::GPUKVCacheMangerImpl::get_cache_startpos_and_length)
    .def("evict_all", &kvcache::GPUKVCacheMangerImpl::evict_all)
    .def("onload_kvcache", &kvcache::GPUKVCacheMangerImpl::onload_kvcache, py::call_guard<py::gil_scoped_release>())
    .def("offload_kvcache", &kvcache::GPUKVCacheMangerImpl::offload_kvcache, py::call_guard<py::gil_scoped_release>())
    .def("is_busy_offloading", &kvcache::GPUKVCacheMangerImpl::is_busy_offloading)
  ;

  py::class_<kvcache::KVOnloadHandle>(m, "KVOnloadHandle")
    .def(py::init<int>(), py::arg("num_layers"))
    // .def("wait", &kvcache::KVOnloadHandle::wait)
    .def("wait_host", &kvcache::KVOnloadHandle::wait_host)
    .def("reset", &kvcache::KVOnloadHandle::reset)
  ;

  py::class_<kvcache::KVOffloadHandle>(m, "KVOffloadHandle")
    .def(py::init())
    .def("record_ready", &kvcache::KVOffloadHandle::record_ready)
  ;

  m.def("prepare_kvcache", &kvcache::prepare_kvcache, "prepare_kvcache", py::call_guard<py::gil_scoped_release>());
  m.def("synchronize_sync_stream", &kvcache::synchronize_sync_stream, "synchronize sync stream", py::call_guard<py::gil_scoped_release>());
  m.def("sync_onload_buffer_to_cache", &kvcache::sync_onload_buffer_to_cache, "sync onload buffer to main cache", py::call_guard<py::gil_scoped_release>());
}