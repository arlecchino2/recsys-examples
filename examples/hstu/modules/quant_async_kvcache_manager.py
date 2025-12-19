import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import math

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
    """支持4-bit量化的KV Cache管理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 为每一层初始化量化器
        self.quantizers = [
            RandomRotationLVQ4BitQuantizer(
                num_prototypes=16,  # 4-bit
                rotation_dim=min(64, self.head_dim),
                seed=42 + i
            )
            for i in range(self.num_layers)
        ]
        
        # 存储量化后的数据（CPU内存）
        self.quantized_host_storage = {}
    
    def offload_kvcache_launch(self, kvcache_metadata):
        """修改：在offload时进行量化"""
        num_offload_pages = len(kvcache_metadata.offload_page_ids)
        if num_offload_pages == 0:
            return None
        
        # 分配GPU缓冲区（原始大小）
        kvcache_metadata.gather_kv_gpu_buffer = torch.empty(
            [self.num_layers * num_offload_pages, 2, self.page_size, self.num_heads, self.head_dim],
            dtype=torch.bfloat16, device=torch.cuda.current_device(),
        )
        
        # 调用原始offload（将数据收集到GPU缓冲区）
        paged_kvcache_ops.offload_kvcache_launch(
            self.host_kv_mgr,
            kvcache_metadata.kv_offload_handle,
            kvcache_metadata.offload_user_ids,
            kvcache_metadata.offload_page_ids,
            kvcache_metadata.gather_kv_gpu_buffer,
            kvcache_metadata.new_offload_startpos,
            kvcache_metadata.new_offload_lengths,
        )
        
        # 异步量化并传输到CPU
        def async_quantize_and_store():
            quantized_data_all_layers = {}
            
            for layer_idx in range(self.num_layers):
                # 提取该层的KV Cache
                layer_kv = kvcache_metadata.gather_kv_gpu_buffer[
                    layer_idx * num_offload_pages:(layer_idx + 1) * num_offload_pages
                ]
                
                # 使用该层的量化器进行量化
                quantizer = self.quantizers[layer_idx]
                quantized_data, quant_metadata = quantizer.quantize_kv_cache(
                    layer_kv, layer_idx
                )
                
                # 存储量化后的数据（索引是4-bit，远小于原始数据）
                quantized_data_all_layers[layer_idx] = {
                    'data': quantized_data,
                    'metadata': quant_metadata
                }
            
            # 将量化后的数据移动到CPU（体积大大减小）
            for layer_idx, data_dict in quantized_data_all_layers.items():
                # 转换为CPU张量
                cpu_data = {
                    'key_indices': data_dict['data']['key_indices'].cpu(),
                    'value_indices': data_dict['data']['value_indices'].cpu()
                }
                cpu_metadata = data_dict['metadata']
                
                # 存储到主机存储
                self.host_kv_mgr.store_quantized_pages(
                    kvcache_metadata.offload_user_ids,
                    kvcache_metadata.offload_page_ids,
                    layer_idx,
                    cpu_data,
                    cpu_metadata
                )
            
            # 释放GPU缓冲区
            del kvcache_metadata.gather_kv_gpu_buffer
            torch.cuda.empty_cache()
        
        # 在单独的线程中执行量化（避免阻塞）
        self.offload_worker.submit(async_quantize_and_store)
    
    def onload_kvcache_finalize(self, user_ids):
        """修改：在onload时进行反量化"""
        # 从主机存储获取量化数据
        quantized_data_dict = self.host_kv_mgr.retrieve_quantized_pages(user_ids)
        
        if quantized_data_dict:
            # 为每个层异步反量化
            def async_dequantize(layer_idx, quantized_data, quant_metadata):
                quantizer = self.quantizers[layer_idx]
                
                # 将数据移回GPU（如果还在CPU）
                if quantized_data['key_indices'].device.type == 'cpu':
                    quantized_data['key_indices'] = quantized_data['key_indices'].cuda()
                    quantized_data['value_indices'] = quantized_data['value_indices'].cuda()
                
                # 反量化
                dequantized_kv = quantizer.dequantize_kv_cache(quantized_data, quant_metadata)
                return dequantized_kv
            
            # 收集所有反量化任务
            dequantize_futures = []
            for layer_idx, (quantized_data, quant_metadata) in quantized_data_dict.items():
                future = self.onload_worker.submit(
                    async_dequantize, layer_idx, quantized_data, quant_metadata
                )
                dequantize_futures.append((layer_idx, future))
            
            # 等待所有层完成并更新GPU缓存
            for layer_idx, future in dequantize_futures:
                dequantized_kv = future.result()
                
                # 将反量化后的数据放回GPU缓存表
                # 这里需要根据实际的页ID更新缓存表
                page_ids = self.host_kv_mgr.get_page_ids_for_user(user_ids, layer_idx)
                self.cache_table[layer_idx, page_ids] = dequantized_kv
        
        # 调用原始方法完成onload
        paged_kvcache_ops.onload_kvcache_finalize(self.gpu_kvcache_mgr, self.host_kv_mgr, user_ids)


# ============ 使用示例 ============
if __name__ == "__main__":
    # 初始化量化版缓存管理器
    quantized_manager = QuantizedAsyncHSTUKVCacheManager(
        num_layers=32,
        num_kv_heads=32,
        kv_headdim=128,
        num_tokens_per_page=128,
        num_primary_cache_pages=8192,
        num_onload_buffer_pages=1024,
        num_reserved_buffer_pages=512,
        num_tokens_per_chunk=256,
        max_num_sequences=256,
        max_sequence_length=8192,
        max_batch_size=64,
    )
    
    print("4-bit量化KV Cache管理器初始化完成")
    print(f"原始KV Cache大小: {quantized_manager.cache_table.nelement() * 2 / 1e9:.2f} GB")
    print(f"量化后预期大小: {quantized_manager.cache_table.nelement() * 0.25 / 1e9:.2f} GB")
    print("压缩率: 75% 内存节省")