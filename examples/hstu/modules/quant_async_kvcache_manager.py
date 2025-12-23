import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import math
from concurrent.futures import ThreadPoolExecutor
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from configs import KVCacheMetadata
import os

from .async_kvcache_manager import AsyncHSTUKVCacheManager
import paged_kvcache_ops

class RandomRotationLVQ4BitQuantizer:
    """
    随机旋转 + LVQ 4-bit量化器
    用于KV Cache的高效压缩
    """
    def __init__(
        self, 
        num_prototypes: int = 16,  # 4-bit对应16个原型
        rotation_dim: int = 64,    # 旋转矩阵的维度
        seed: int = 42
    ):
        # 4-bit量化：16个原型向量
        self.num_prototypes = num_prototypes
        self.rotation_dim = rotation_dim
        
        # 初始化随机旋转矩阵 (保持正交性)
        torch.manual_seed(seed)
        self.rotation_matrix = self._init_random_rotation(rotation_dim)
        
        # 每个头单独维护LVQ原型码本
        self.prototype_codebooks = {}  # layer_idx -> (key_prototypes, value_prototypes)
        
    def _init_random_rotation(self, dim: int) -> torch.Tensor:
        """初始化随机正交旋转矩阵"""
        # 使用QR分解确保正交性
        random_matrix = torch.randn(dim, dim)
        q, _ = torch.linalg.qr(random_matrix)
        return q.to(torch.bfloat16)
    
    def _apply_random_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用随机旋转到特征维度
        x: [..., head_dim] -> [..., head_dim]
        """
        original_shape = x.shape
        head_dim = original_shape[-1]
        
        # 如果head_dim不是rotation_dim的倍数，需要填充
        if head_dim % self.rotation_dim != 0:
            # 简单实现：只旋转前rotation_dim个维度
            x_rotated = x[..., :self.rotation_dim] @ self.rotation_matrix.T
            x = torch.cat([x_rotated, x[..., self.rotation_dim:]], dim=-1)
        else:
            # 重塑并应用旋转
            x_flat = x.view(-1, head_dim)
            # 分块旋转
            chunks = []
            for i in range(0, head_dim, self.rotation_dim):
                chunk = x_flat[:, i:i+self.rotation_dim]
                rotated_chunk = chunk @ self.rotation_matrix.T
                chunks.append(rotated_chunk)
            x_rotated = torch.cat(chunks, dim=1)
            x = x_rotated.view(original_shape)
            
        return x
    
    def _inverse_random_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """逆旋转：恢复原始特征空间"""
        original_shape = x.shape
        head_dim = original_shape[-1]
        
        if head_dim % self.rotation_dim != 0:
            x_inv = x[..., :self.rotation_dim] @ self.rotation_matrix
            x = torch.cat([x_inv, x[..., self.rotation_dim:]], dim=-1)
        else:
            x_flat = x.view(-1, head_dim)
            chunks = []
            for i in range(0, head_dim, self.rotation_dim):
                chunk = x_flat[:, i:i+self.rotation_dim]
                inv_chunk = chunk @ self.rotation_matrix
                chunks.append(inv_chunk)
            x_inv = torch.cat(chunks, dim=1)
            x = x_inv.view(original_shape)
            
        return x
    
    def _lvq_quantize(self, x: torch.Tensor, layer_idx: int, is_key: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LVQ量化：学习原型并量化
        返回：量化后的索引和更新后的原型
        """
        # 确保码本存在
        if layer_idx not in self.prototype_codebooks:
            self.prototype_codebooks[layer_idx] = (
                torch.randn(self.num_prototypes, self.rotation_dim, dtype=torch.bfloat16) * 0.1,
                torch.randn(self.num_prototypes, self.rotation_dim, dtype=torch.bfloat16) * 0.1
            )
        
        # 获取当前层的原型
        key_prototypes, value_prototypes = self.prototype_codebooks[layer_idx]
        prototypes = key_prototypes if is_key else value_prototypes
        
        # 重塑输入：只处理旋转后的维度
        x_flat = x.view(-1, x.shape[-1])
        if x.shape[-1] > self.rotation_dim:
            x_flat = x_flat[:, :self.rotation_dim]
        
        # 计算距离并找到最近的原型索引
        distances = torch.cdist(x_flat, prototypes)  # [batch*seq, num_prototypes]
        indices = torch.argmin(distances, dim=-1)  # [batch*seq]
        
        # 学习：更新原型（简单在线学习）
        learning_rate = 0.01
        for i in range(self.num_prototypes):
            mask = (indices == i)
            if mask.sum() > 0:
                assigned_vectors = x_flat[mask]
                prototypes[i] = (1 - learning_rate) * prototypes[i] + \
                               learning_rate * assigned_vectors.mean(dim=0)
        
        # 更新码本
        if is_key:
            self.prototype_codebooks[layer_idx] = (prototypes, value_prototypes)
        else:
            self.prototype_codebooks[layer_idx] = (key_prototypes, prototypes)
        
        return indices, prototypes[indices].view(x.shape[:-1] + (prototypes.shape[-1],))
    
    def quantize_kv_cache(self, kv_cache: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, dict]:
        """
        量化KV Cache
        kv_cache: [num_pages, 2, page_size, num_heads, head_dim]
        返回：量化后的数据和量化元数据
        """
        num_pages, _, page_size, num_heads, head_dim = kv_cache.shape
        
        # 分离key和value
        keys = kv_cache[:, 0]  # [num_pages, page_size, num_heads, head_dim]
        values = kv_cache[:, 1]
        
        # 应用随机旋转
        keys_rotated = self._apply_random_rotation(keys)
        values_rotated = self._apply_random_rotation(values)
        
        # LVQ量化
        key_indices_list = []
        value_indices_list = []
        key_quantized_list = []
        value_quantized_list = []
        
        for h in range(num_heads):
            # 量化每个头的key
            key_indices, key_quantized = self._lvq_quantize(
                keys_rotated[..., h, :], layer_idx, is_key=True
            )
            # 量化每个头的value
            value_indices, value_quantized = self._lvq_quantize(
                values_rotated[..., h, :], layer_idx, is_key=False
            )
            
            key_indices_list.append(key_indices)
            value_indices_list.append(value_indices)
            key_quantized_list.append(key_quantized)
            value_quantized_list.append(value_quantized)
        
        # 合并结果
        key_indices = torch.stack(key_indices_list, dim=-2)  # [num_pages, page_size, num_heads]
        value_indices = torch.stack(value_indices_list, dim=-2)
        
        # 4-bit打包：每两个索引打包成一个字节
        def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
            indices_uint8 = indices.to(torch.uint8)
            # 确保索引在0-15范围内
            indices_uint8 = indices_uint8 & 0x0F
            # 打包：每两个4-bit打包成一个8-bit
            packed = torch.zeros(indices_uint8.shape[:-1] + (indices_uint8.shape[-1] // 2,), 
                               dtype=torch.uint8)
            for i in range(0, indices_uint8.shape[-1], 2):
                high = indices_uint8[..., i] << 4
                low = indices_uint8[..., i+1] if i+1 < indices_uint8.shape[-1] else 0
                packed[..., i//2] = high | low
            return packed
        
        key_indices_packed = pack_4bit(key_indices)
        value_indices_packed = pack_4bit(value_indices)
        
        # 准备量化元数据
        quant_metadata = {
            'layer_idx': layer_idx,
            'original_shape': kv_cache.shape,
            'rotation_matrix': self.rotation_matrix,
            'key_prototypes': self.prototype_codebooks[layer_idx][0],
            'value_prototypes': self.prototype_codebooks[layer_idx][1],
            'key_indices_shape': key_indices.shape,
            'value_indices_shape': value_indices.shape,
        }
        
        # 返回打包后的索引和元数据（压缩率：32倍，从bfloat16到4-bit）
        quantized_data = {
            'key_indices': key_indices_packed,
            'value_indices': value_indices_packed
        }
        
        return quantized_data, quant_metadata
    
    def dequantize_kv_cache(self, quantized_data: dict, quant_metadata: dict) -> torch.Tensor:
        """
        反量化KV Cache
        """
        # 解包4-bit索引
        def unpack_4bit(packed: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
            indices = torch.zeros(original_shape, dtype=torch.long)
            packed_flat = packed.view(-1)
            
            for i in range(0, len(packed_flat)):
                byte = packed_flat[i].item()
                indices_flat = indices.view(-1)
                idx = i * 2
                if idx < len(indices_flat):
                    indices_flat[idx] = (byte >> 4) & 0x0F
                if idx + 1 < len(indices_flat):
                    indices_flat[idx + 1] = byte & 0x0F
            
            return indices.view(original_shape)
        
        key_indices = unpack_4bit(quantized_data['key_indices'], 
                                 quant_metadata['key_indices_shape'])
        value_indices = unpack_4bit(quantized_data['value_indices'], 
                                   quant_metadata['value_indices_shape'])
        
        # 从LVQ码本中查找原型
        layer_idx = quant_metadata['layer_idx']
        key_prototypes, value_prototypes = self.prototype_codebooks[layer_idx]
        
        # 恢复量化后的值
        num_pages, page_size, num_heads = key_indices.shape[:3]
        head_dim = key_prototypes.shape[-1]
        
        keys_dequantized = key_prototypes[key_indices.view(-1)].view(
            num_pages, page_size, num_heads, head_dim)
        values_dequantized = value_prototypes[value_indices.view(-1)].view(
            num_pages, page_size, num_heads, head_dim)
        
        # 应用逆旋转
        keys_restored = self._inverse_random_rotation(keys_dequantized)
        values_restored = self._inverse_random_rotation(values_dequantized)
        
        # 重新组合为KV Cache格式
        kv_cache_dequantized = torch.stack([keys_restored, values_restored], dim=1)
        
        # 如果原始head_dim更大，需要填充
        original_head_dim = quant_metadata['original_shape'][-1]
        if head_dim < original_head_dim:
            padding = torch.zeros(kv_cache_dequantized.shape[:-1] + 
                                 (original_head_dim - head_dim,),
                                 dtype=kv_cache_dequantized.dtype)
            kv_cache_dequantized = torch.cat([kv_cache_dequantized, padding], dim=-1)
        
        return kv_cache_dequantized


# ============ 修改AsyncHSTUKVCacheManager以集成量化 ============

class QuantizedAsyncHSTUKVCacheManager(AsyncHSTUKVCacheManager):
    """支持2-bit量化的KV Cache管理器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 获取head_dim（从父类或参数）
        head_dim = getattr(self, 'head_dim', kwargs.get('kv_headdim', 128))
        num_layers = getattr(self, 'num_layers', kwargs.get('num_layers', 32))

        # 为每一层初始化2-bit量化器
        self.quantizers = [
            RandomRotationLVQ4BitQuantizer(
                num_prototypes=4,  # 2-bit: 4个原型
                rotation_dim=min(64, head_dim),
                seed=42 + i
            )
            for i in range(num_layers)
        ]

        self.offload_worker = ThreadPoolExecutor(max_workers=4)

    def offload_kvcache_quant(self, kvcache_metadata):

        print("debug in quant_xx.py: function offload_kvcache_launch")
        """修改：在offload时进行2-bit量化"""
        num_offload_pages = len(kvcache_metadata.offload_page_ids)
        if num_offload_pages == 0:
            return None

        if not hasattr(kvcache_metadata, 'gather_kv_gpu_buffer'):
            # 分配GPU缓冲区（原始大小）
            kvcache_metadata.gather_kv_gpu_buffer = torch.empty(
                [self.num_layers * num_offload_pages, 2, self.page_size, self.num_heads, self.head_dim],
                dtype=torch.bfloat16, device=torch.cuda.current_device(),
            )

        self.gpu_kvcache_mgr.offload_kvcache(
            kvcache_metadata.kv_offload_handle,
            kvcache_metadata.offload_user_ids,
            kvcache_metadata.offload_page_ids,
            kvcache_metadata.gather_kv_gpu_buffer,
            kvcache_metadata.new_offload_startpos,
            kvcache_metadata.new_offload_lengths,
        )

       # 异步量化并传输到CPU
        def async_quantize_and_store():
            for layer_idx in range(self.num_layers):
                # 提取该层的KV Cache
                layer_kv = kvcache_metadata.gather_kv_gpu_buffer[
                    layer_idx * num_offload_pages:(layer_idx + 1) * num_offload_pages
                ]

                # 使用该层的量化器进行2-bit量化
                quantizer = self.quantizers[layer_idx]
                quantized_data, quant_metadata = quantizer.quantize_kv_cache(
                    layer_kv, layer_idx
                )

                cpu_data = {
                    'key_indices': quantized_data['key_indices'].cpu(),
                    'value_indices': quantized_data['value_indices'].cpu()
                }

                self.host_kv_mgr.store_quantized_pages(
                    kvcache_metadata.offload_user_ids,
                    kvcache_metadata.offload_page_ids,
                    layer_idx,
                    cpu_data,
                    quant_metadata
                )

            # 释放GPU缓冲区
            if hasattr(kvcache_metadata, 'gather_kv_gpu_buffer'):
                del kvcache_metadata.gather_kvcache_gpu_buffer
                torch.cuda.empty_cache()

        # 在单独的线程中执行量化
        self.offload_worker.submit(async_quantize_and_store)


    def prepare_kvcache_async_quant(self,
        batch_size,
        user_ids,
        total_history_lengths,
        static_page_ids_gpu_buffer,
        static_offload_page_ids_gpu_buffer,
        static_onload_handle,
    ):
        origin_cached_lengths = self.gpu_kvcache_mgr.get_total_cache_length(user_ids)
        new_tokens = sum([ total_history_lengths[idx] - origin_cached_lengths[idx] for idx in range(batch_size) ])
        if new_tokens <= 0:
            print(total_history_lengths)
            print(origin_cached_lengths)

        offload_uids_buffer = torch.empty([batch_size,], dtype=torch.int64)
        metadata_host_buffer = torch.empty([batch_size * 7 + 7,], dtype=torch.int, pin_memory=True)
        metadata_gpu_buffer = torch.empty([batch_size * 5 + 4 + new_tokens * 2,], dtype=torch.int, device = torch.cuda.current_device())

        kvcache_metadata_fut = self.executor.submit(paged_kvcache_ops.prepare_kvcache,
            self.gpu_kvcache_mgr, self.host_kv_mgr,
            user_ids, total_history_lengths,
            static_page_ids_gpu_buffer, static_offload_page_ids_gpu_buffer,
            offload_uids_buffer,
            metadata_host_buffer, metadata_gpu_buffer)

        static_onload_handle.reset()

        # 修改：创建包含反量化的onload函数
        def onload_with_dequantization(user_ids, static_onload_handle):
            """执行onload并包含KV Cache反量化"""
            # 1. 先执行原始的onload操作
            self.gpu_kvcache_mgr.onload_kvcache(user_ids, static_onload_handle)

            # 2. 对每个用户进行反量化处理
            print("debug in quant_xx.py: function onload_kvcache_finalize")

            # 为每个层反量化
            for layer_idx in range(len(self.quantizers)):
                quantizer = self.quantizers[layer_idx]

            # 对每个用户ID处理
            for user_id in user_ids:
                # 获取该用户在该层的所有页面ID
                page_ids = self.host_kv_mgr.get_page_ids_for_user(user_id, layer_idx)

                if not page_ids:
                    continue

                # 从主机存储获取量化数据
                retrieved_data = self.host_kv_mgr.retrieve_quantized_pages(
                    user_id, page_ids, layer_idx
                )

                if retrieved_data:
                    # 提取数据指针
                    key_indices_ptr = retrieved_data['key_indices']
                    value_indices_ptr = retrieved_data['value_indices']
                    scales_ptr = retrieved_data['scales']
                    zeros_ptr = retrieved_data['zeros']
                    num_pages = retrieved_data['num_pages']

                    # 将数据转换为PyTorch张量
                    quant_page_numel = self.host_kv_mgr.quant_page_numel
                    num_kv_heads = self.host_kv_mgr.num_kv_heads

                    # 创建量化数据字典
                    quantized_data = {
                        'key_indices': torch.from_numpy(
                            np.ctypeslib.as_array(
                                ctypes.cast(key_indices_ptr, ctypes.POINTER(ctypes.c_uint16)),
                                shape=(num_pages, quant_page_numel)
                            )
                        ).cuda(),
                        'value_indices': torch.from_numpy(
                            np.ctypeslib.as_array(
                                ctypes.cast(value_indices_ptr, ctypes.POINTER(ctypes.c_uint16)),
                                shape=(num_pages, quant_page_numel)
                            )
                        ).cuda()
                    }

                    # 创建量化元数据
                    quant_metadata = {
                        'scales': torch.from_numpy(
                            np.ctypeslib.as_array(
                                ctypes.cast(scales_ptr, ctypes.POINTER(ctypes.c_float)),
                                shape=(num_pages, num_kv_heads)
                            )
                        ).cuda(),
                        'zeros': torch.from_numpy(
                            np.ctypeslib.as_array(
                                ctypes.cast(zeros_ptr, ctypes.POINTER(ctypes.c_float)),
                                shape=(num_pages, num_kv_heads)
                            )
                        ).cuda(),
                        'bits': 2,  # 假设是2-bit量化
                        'group_size': 4  # 假设分组大小为4
                    }

                    # 反量化
                    dequantized_kv = quantizer.dequantize_kv_cache(quantized_data, quant_metadata)

                    # 将反量化后的数据放回GPU缓存表
                    # 这里需要根据实际的页ID更新缓存表
                    # 假设cache_table的格式是: cache_table[layer_idx][user_id][page_id]
                    for i, page_id in enumerate(page_ids):
                        if page_id < len(self.cache_table[layer_idx][user_id]):
                            self.cache_table[layer_idx][user_id][page_id] = dequantized_kv[i]

                    # 释放检索到的数据内存
                    self.host_kv_mgr.free_retrieved_pages(retrieved_data)

                     # 修改：提交包含反量化的onload任务
        onload_fut = self.onload_worker.submit(onload_with_dequantization,
            user_ids, static_onload_handle)

        return origin_cached_lengths, new_tokens, offload_uids_buffer, metadata_host_buffer, metadata_gpu_buffer, kvcache_metadata_fut, onload_fut
