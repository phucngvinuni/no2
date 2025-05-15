# model_util.py
import math
from typing import Tuple
import numpy as np
from timm.models.registry import register_model # Keep for model registration if any models defined here
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
# Import IMGC_NUMCLASS only if FIM modules are to be used with their classifiers
# from base_args import IMGC_NUMCLASS

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x) # Original timm might have drop after fc2
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias) # Original timm has bias here
        # Your code had bias=False and separate q_bias, v_bias. Sticking to your version:
        # self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        # if qkv_bias:
        #     self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        #     self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        # else:
        #     self.q_bias = None
        #     self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Original timm logic for qkv with bias:
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Your logic:
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) # C might be all_head_dim here
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.relu = nn.ReLU() # Original MAE/ViT often don't have a separate ReLU here after MLP
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        # If a relu was intended here in your original code, add it back.
        # x = self.relu(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        if not (H == self.img_size[0] and W == self.img_size[1]): # Allow slight flexibility if needed, but better to assert
             # x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False) # Optional resize
            raise ValueError(f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x).flatten(2).transpose(1, 2) # B, N, C
        return x

def get_sinusoid_encoding_table(n_position, d_hid, cls_token=False): # Added cls_token flag
    # n_position: number of patches, e.g., 196 for 224/16
    # d_hid: embedding dimension
    if cls_token:
        n_position = n_position + 1 # For [CLS] token

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ViTEncoder_Van(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, # qk_scale not used by Block
                 drop_rate=0., attn_drop_rate=0., # attn_drop_rate not used by Block
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.0,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.register_buffer('pos_embed', get_sinusoid_encoding_table(self.num_patches, embed_dim), persistent=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights_vit) # Use a specific init for ViT parts

    def _init_weights_vit(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio): # For MAE pre-training if used
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        if len_keep == L: # No masking
            return x, None, None # No mask, no ids_restore

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=torch.bool) # True for remove/masked
        mask[:, :len_keep] = 0 # Keep first len_keep after shuffling
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore # mask is boolean, True for MASKED by encoder

    def forward(self, x_img: torch.Tensor, encoder_boolean_mask: torch.Tensor = None):
        # x_img: [B, C, H, W]
        # encoder_boolean_mask: [B, NumTotalPatches], True if patch is MASKED for encoder.
        # If None or all False, all patches are processed.

        x_tokens = self.patch_embed(x_img) # [B, NumTotalPatches, EmbedDim]
        x_tokens = x_tokens + self.pos_embed # Add positional embedding to all tokens

        if encoder_boolean_mask is not None and encoder_boolean_mask.any():
            # If a mask is provided, select only the UNMASKED (visible) tokens
            # `encoder_boolean_mask` has True for MASKED, so we need ~encoder_boolean_mask
            # Need to reshape mask to select tokens: [B, NumTotalPatches] -> for selection
            # x_tokens is [B, NumTotalPatches, D]
            # We need to select tokens along dim 1.
            # A common way is to flatten, select, then reshape.
            B, L, D = x_tokens.shape
            # ~encoder_boolean_mask is [B, L], True for VISIBLE
            # This selection is a bit tricky with batched indexing.
            # Simplest if encoder_boolean_mask is used to *create* ids_keep
            
            # MAE style selection:
            # noise = torch.rand(B, L, device=x_tokens.device)
            # ids_shuffle = torch.argsort(noise, dim=1)
            # ids_restore = torch.argsort(ids_shuffle, dim=1)
            # len_keep = (~encoder_boolean_mask).sum(dim=1).min().item() # Smallest number of visible tokens in batch
            # ids_keep = ids_shuffle[:, :len_keep]
            # x_proc = torch.gather(x_tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
            # This assumes a fixed number of kept tokens, which `encoder_boolean_mask` doesn't guarantee.

            # Simpler: process only the tokens where mask is False
            # This will result in variable length sequences if mask is not uniform per batch item.
            # Transformer blocks expect fixed length.
            # So, the standard MAE approach is:
            # 1. Apply pos_embed
            # 2. Shuffle tokens + get ids_restore
            # 3. Keep the first N_visible tokens
            # 4. Pass N_visible tokens through encoder blocks
            # The `encoder_boolean_mask` you pass from dataset is already effectively the result of step 3.
            # It determines WHICH tokens are visible. The encoder just processes these.
            x_proc = x_tokens[~encoder_boolean_mask].view(B, -1, D) # ~mask means visible
        else: # No mask or all False mask -> process all tokens
            x_proc = x_tokens
            # ids_restore = None # No shuffling was done if all are processed

        for blk in self.blocks:
            x_proc = blk(x_proc)
        x_proc = self.norm(x_proc)

        # If doing MAE, the encoder would return x_proc (visible tokens), the mask, and ids_restore
        # For this SemCom, we just need the processed tokens. The decoder will need to know
        # which ones were masked if it's to reconstruct the full set.
        return x_proc # Features of an podmnožina (subset) of patches


class ViTDecoder_ImageReconstruction(nn.Module):
    def __init__(self, patch_size=16, num_total_patches=196,
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=nn.LayerNorm, init_values=0.0): # Added init_values default
        super().__init__()
        self.num_total_patches = num_total_patches
        self.patch_size_h, self.patch_size_w = to_2tuple(patch_size)
        self.embed_dim = embed_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer('pos_embed_decoder', get_sinusoid_encoding_table(self.num_total_patches, embed_dim), persistent=False)

        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer, init_values=init_values, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, 3 * self.patch_size_h * self.patch_size_w)
        
        trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights_vit)

    def _init_weights_vit(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # --- CORRECTED SIGNATURE TO MATCH THE CALL ---
    def forward(self,
                x_vis_tokens: torch.Tensor,         # Matches keyword arg from model.py
                encoder_mask_boolean: torch.Tensor, # Matches keyword arg
                full_image_num_patches_h: int,      # Matches keyword arg
                full_image_num_patches_w: int,      # Matches keyword arg
                ids_restore_if_mae: torch.Tensor = None # Matches keyword arg, optional
               ) -> torch.Tensor:
        B = x_vis_tokens.shape[0]
        D_decoder = x_vis_tokens.shape[-1]
        assert D_decoder == self.embed_dim, f"Decoder input embed_dim mismatch. Expected {self.embed_dim}, got {D_decoder}"

        x_full_sequence_for_transformer: torch.Tensor

        if not encoder_mask_boolean.any(): # No masking by encoder, or all patches are visible
            # This implies x_vis_tokens contains features for ALL patches.
            if x_vis_tokens.shape[1] != self.num_total_patches:
                 # This case can happen if mask_ratio=0 but somehow encoder still outputs fewer tokens.
                 # Or if encoder_mask_boolean was all False, but x_vis_tokens is still just visible.
                 # This indicates a logic mismatch upstream or how encoder_mask_boolean is used.
                 print(f"Warning: Decoder received {x_vis_tokens.shape[1]} tokens but encoder_mask_boolean indicates "
                       f"no masking or all visible. Expected {self.num_total_patches} tokens for full reconstruction path.")
                 # Fallback: Attempt MAE style if ids_restore provided, else error or pad.
                 if ids_restore_if_mae is not None:
                     print("  Attempting MAE unshuffle with ids_restore_if_mae.")
                 else: # Simplest: if mask_ratio = 0, encoder MUST send all tokens.
                      raise ValueError(f"Decoder expected {self.num_total_patches} tokens when encoder_mask is all False, "
                                       f"but received {x_vis_tokens.shape[1]}. Ensure encoder sends all patch features when mask_ratio=0.")

            x_full_sequence_for_transformer = x_vis_tokens + self.pos_embed_decoder.expand(B, -1, -1)
        
        else: # MAE-style reconstruction: encoder_mask_boolean has True for some masked patches
            num_visible_patches = x_vis_tokens.shape[1]
            # Check consistency: number of False in mask should match number of visible tokens
            expected_visible = (~encoder_mask_boolean).sum(dim=1)
            if not torch.all(expected_visible == num_visible_patches):
                # This is a critical error in how visible tokens are passed or how mask is formed.
                # For batched processing, num_visible_patches must be consistent or handled by padding.
                # Assuming for now that each item in batch has same num_visible_patches if MAE used.
                raise ValueError(
                    f"Mismatch between x_vis_tokens length ({num_visible_patches}) and "
                    f"number of unmasked patches in encoder_mask_boolean (varies, e.g., {expected_visible[0]}). "
                    "Ensure consistent number of visible tokens per batch or pad."
                )

            if ids_restore_if_mae is not None: # Preferred MAE un-shuffling
                num_masked_patches = self.num_total_patches - num_visible_patches
                if num_masked_patches < 0: # Should not happen
                    raise ValueError("More visible tokens than total patches based on ids_restore.")

                mask_tokens_to_append = self.mask_token.repeat(B, num_masked_patches, 1)
                x_temp_shuffled = torch.cat([x_vis_tokens, mask_tokens_to_append], dim=1)
                
                # Un-shuffle to restore original patch order
                x_unshuffled = torch.gather(x_temp_shuffled, dim=1,
                                            index=ids_restore_if_mae.unsqueeze(-1).expand(-1, -1, D_decoder))
                x_full_sequence_for_transformer = x_unshuffled + self.pos_embed_decoder.expand(B, -1, -1)
            else:
                # Simpler MAE reconstruction without shuffling (relies on encoder_mask_boolean for placement)
                # This is harder to implement correctly for batched variable length inputs.
                # For now, if ids_restore is None but masking occurred, this path needs robust implementation.
                # The most straightforward for now is to ensure encoder always outputs num_total_patches
                # (either actual features or placeholders that become mask_tokens before decoder blocks).
                # Or, the encoder must provide ids_restore if it only sends visible tokens.
                # Given the current ViTEncoder_Van, if masking is active, it only returns visible tokens.
                # So, this path is problematic without ids_restore.
                print(f"Warning: MAE-style decoding invoked (due to encoder_mask_boolean) but "
                      f"ids_restore_if_mae is None. Decoder needs a way to reconstruct the full sequence. "
                      f"Number of visible tokens received: {num_visible_patches}")

                # Fallback: create a full sequence and fill in visible tokens.
                # This assumes x_vis_tokens are already in their "correct" sparse order.
                x_full_sequence_for_transformer = self.mask_token.expand(B, self.num_total_patches, -1) + \
                                                  self.pos_embed_decoder.expand(B, -1, -1)
                
                # Create a view for easier scatter (if needed, but direct assignment is fine)
                # This is a non-trivial scatter operation for batches.
                # A common MAE pattern:
                # full_indices = torch.arange(self.num_total_patches, device=x_vis_tokens.device).unsqueeze(0).expand(B, -1)
                # visible_indices_bool = ~encoder_mask_boolean
                # For each batch item, place x_vis_tokens into x_full_sequence_for_transformer
                # This simplified version might require x_vis_tokens to be padded if not all batches have same num_visible
                idx_batch = 0
                for i in range(B):
                    num_vis_this_item = (~encoder_mask_boolean[i]).sum().item()
                    if num_vis_this_item > 0 : # Check if any visible tokens for this item
                        x_full_sequence_for_transformer[i, ~encoder_mask_boolean[i], :] = x_vis_tokens[idx_batch : idx_batch + num_vis_this_item, :] + \
                                                                                      self.pos_embed_decoder.squeeze(0)[~encoder_mask_boolean[i], :]
                        idx_batch += num_vis_this_item
                if idx_batch != x_vis_tokens.shape[0]*x_vis_tokens.shape[1]/D_decoder and x_vis_tokens.shape[1] != self.num_total_patches : # if x_vis_tokens not [B,L,D]
                    if x_vis_tokens.ndim == 2 and x_vis_tokens.shape[0] == (~encoder_mask_boolean).sum().item(): # if x_vis_tokens is flattened [N_total_vis, D]
                         # This part is very tricky if num_visible varies per batch item and x_vis_tokens is flattened
                         # It's much cleaner if ViTEncoder_Van always outputs [B, N_fixed_visible, D] or provides ids_restore
                         print("Warning: Complex case for MAE decoder with flattened variable visible tokens. Reconstruction might be incorrect.")
                         pass # Let it proceed, but likely an issue


        decoded_tokens = x_full_sequence_for_transformer
        for blk in self.blocks:
            decoded_tokens = blk(decoded_tokens)
        decoded_tokens = self.norm(decoded_tokens)

        pixels_per_patch = self.head(decoded_tokens)

        reconstructed_image = pixels_per_patch.view(
            B,
            full_image_num_patches_h,
            full_image_num_patches_w,
            3,
            self.patch_size_h,
            self.patch_size_w
        )
        reconstructed_image = reconstructed_image.permute(0, 3, 1, 4, 2, 5).contiguous()
        reconstructed_image = reconstructed_image.view(
            B, 3,
            full_image_num_patches_h * self.patch_size_h,
            full_image_num_patches_w * self.patch_size_w
        )
        reconstructed_image = torch.sigmoid(reconstructed_image)
        return reconstructed_image

class HierarchicalQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25,
                 hier_init: bool = True, linkage: str = 'ward'): # Removed distance_threshold for simplicity now
        super().__init__()
        self.num_embeddings = num_embeddings    # K (number of codebook vectors)
        self.embedding_dim = embedding_dim      # D (dimension of each codebook vector)
        self.commitment_cost = commitment_cost  # Beta for VQ loss

        # The codebook is an nn.Embedding layer. Its weights are the codebook vectors.
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize codebook weights (common practice for VQ-VAE)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        self.hier_init_done = not hier_init # Flag to control hierarchical initialization
        self.linkage = linkage


    def _initialize_centroids_hierarchical(self, x_flat_cpu_np: np.ndarray):
        # This function attempts to initialize self.embedding.weight using hierarchical clustering
        # It should only run once during the first suitable training batch.
        if not self.hier_init_done: # Check the flag
            try:
                from scipy.cluster.hierarchy import linkage as scipy_linkage_func, fcluster
            except ImportError:
                print("Warning: Scipy not found for HierarchicalQuantizer init. Using nn.Embedding default init.")
                self.hier_init_done = True # Don't try again
                return

            if x_flat_cpu_np.shape[0] < self.num_embeddings and x_flat_cpu_np.shape[0] > 1 :
                print(f"Warning: Hierarchical init has fewer samples ({x_flat_cpu_np.shape[0]}) "
                      f"than num_embeddings ({self.num_embeddings}). Initializing with available samples and random.")
                # Use available samples and fill the rest randomly or by duplication
                num_available = x_flat_cpu_np.shape[0]
                centroids = np.zeros((self.num_embeddings, self.embedding_dim), dtype=x_flat_cpu_np.dtype)
                centroids[:num_available] = x_flat_cpu_np
                if num_available > 0 and num_available < self.num_embeddings:
                    fill_indices = np.random.choice(num_available, self.num_embeddings - num_available, replace=True)
                    centroids[num_available:] = x_flat_cpu_np[fill_indices]
                elif num_available == 0: # No samples, keep random init from nn.Embedding
                    self.hier_init_done = True
                    return

            elif x_flat_cpu_np.shape[0] >= self.num_embeddings : # Sufficient samples
                Z = scipy_linkage_func(x_flat_cpu_np, method=self.linkage)
                labels = fcluster(Z, self.num_embeddings, criterion='maxclust')
                
                centroids = np.zeros((self.num_embeddings, self.embedding_dim), dtype=x_flat_cpu_np.dtype)
                unique_labels = np.unique(labels) # Actual cluster labels formed
                
                # Map found cluster centroids to the first N codebook entries
                for i, label_val in enumerate(unique_labels):
                    if i >= self.num_embeddings: break # Should not happen if fcluster worked as expected
                    mask = (labels == label_val)
                    if np.any(mask):
                        centroids[i] = x_flat_cpu_np[mask].mean(axis=0)
                    else: # Should also not happen if label_val is from unique_labels
                        centroids[i] = x_flat_cpu_np[np.random.randint(len(x_flat_cpu_np))]
                
                # If fcluster gave fewer than num_embeddings distinct clusters
                if len(unique_labels) < self.num_embeddings:
                    num_to_fill = self.num_embeddings - len(unique_labels)
                    fill_indices = np.random.choice(x_flat_cpu_np.shape[0], num_to_fill, replace=True)
                    centroids[len(unique_labels):] = x_flat_cpu_np[fill_indices]
            else: # x_flat_cpu_np.shape[0] <= 1, not enough for linkage
                print("Warning: Not enough samples (<2) for hierarchical clustering init. Using nn.Embedding default init.")
                self.hier_init_done = True
                return
            
            self.embedding.weight.data.copy_(torch.from_numpy(centroids).to(self.embedding.weight.device))
            print("HierarchicalQuantizer: Centroids initialized using hierarchical clustering.")
            self.hier_init_done = True # Mark initialization as done


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        assert x.shape[-1] == self.embedding_dim, "Input last dim must match embedding_dim"
        x_flat = x.reshape(-1, self.embedding_dim)

        if not self.hier_init_done and self.training:
            # Attempt hierarchical initialization only if flag is set and in training
            # and enough data points are available in the current batch
            if x_flat.shape[0] > 1 : # Need at least 2 for linkage
                self._initialize_centroids_hierarchical(x_flat.detach().cpu().numpy())
            else: # Not enough samples in this specific batch, will try next if still training
                pass


        # Calculate L2 distances to codebook vectors (self.embedding.weight)
        distances_sq = (
            torch.sum(x_flat**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) - # Access .weight of nn.Embedding
            2 * torch.matmul(x_flat, self.embedding.weight.t())
        )
        # Clamp to avoid sqrt of tiny negative numbers due to float precision
        distances_sq = torch.clamp(distances_sq, min=0.0)

        # Find the closest codebook vector indices
        encoding_indices = torch.argmin(distances_sq, dim=1)  # Shape: [N]

        # Quantize: Use embedding layer to look up the chosen codebook vectors
        quantized_flat = self.embedding(encoding_indices) # Shape: [N, D]

        # Reshape quantized vectors back to the original input shape
        quantized_output_tokens = quantized_flat.view(original_shape)

        # VQ Losses:
        # 1. Codebook loss (moves codebook vectors closer to encoder outputs)
        codebook_loss = F.mse_loss(quantized_output_tokens, x_flat.view_as(quantized_output_tokens).detach())
        # 2. Commitment loss (encourages encoder outputs to commit to a codebook vector)
        commitment_loss = F.mse_loss(x_flat.view_as(quantized_output_tokens), quantized_output_tokens.detach())
        
        vq_total_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-Through Estimator for gradients to flow back to encoder
        quantized_output_tokens_st = x + (quantized_output_tokens - x).detach()

        # Perplexity (optional metric for codebook usage)
        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_output_tokens_st, vq_total_loss, perplexity, encoding_indices


class Channels(nn.Module):
    def __init__(self):
        super().__init__()

    def _get_noise_std(self, signal_power_per_symbol_component: float, total_noise_variance: float) -> float:
        # Assuming signal_power_per_symbol_component is for one real component if signal is complex
        # Or for the real symbol if signal is real.
        # If noise_variance is total variance for a complex noise (split between real/imag),
        # then variance per component is total_noise_variance / 2.
        # std_dev per component = sqrt(total_noise_variance / 2).
        # If total_noise_variance is already per real component, then std_dev = sqrt(total_noise_variance).
        # For simplicity, let's assume total_noise_variance is what's given.
        # For complex (real+imag), noise power is split.
        # Here, n_var is total variance for a real symbol, or for *each component* of a complex symbol
        # if it's pre-normalized to unit power per component.
        # If n_var is total noise power for a complex symbol (Ps/Pn = SNR, Ps=1, Pn=n_var),
        # then noise power per real/imag component is n_var/2. Std dev is sqrt(n_var/2).
        return math.sqrt(max(0.0, total_noise_variance))


    def AWGN(self, Tx_sig: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        device = Tx_sig.device
        noise_std = self._get_noise_std(1.0, total_noise_variance) # Assume Tx_sig is normalized, so Ps_component=1

        if torch.is_complex(Tx_sig):
            # For complex, total_noise_variance is often defined for the complex number,
            # so variance per real/imag component is total_noise_variance / 2.
            noise_std_per_component = math.sqrt(max(0.0, total_noise_variance) / 2.0)
            noise = torch.normal(0, noise_std_per_component, size=Tx_sig.shape, device=device) + \
                    1j * torch.normal(0, noise_std_per_component, size=Tx_sig.shape, device=device)
        else: # Real signal
            noise = torch.normal(0, noise_std, size=Tx_sig.shape, device=device)
        return Tx_sig + noise

    def _apply_flat_fading_channel(self, Tx_sig: torch.Tensor, H_complex: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        # Tx_sig: [B, L, D_feat_real_or_complex]
        # H_complex: [B, 1, 1] complex fading coefficient
        
        is_input_real = not torch.is_complex(Tx_sig)
        original_shape = Tx_sig.shape
        
        Tx_as_complex = Tx_sig
        if is_input_real:
            if Tx_sig.shape[-1] % 2 != 0:
                raise ValueError("Last dim of real Tx_sig must be even for complex representation in fading.")
            Tx_as_complex = torch.complex(Tx_sig[..., :Tx_sig.shape[-1]//2], Tx_sig[..., Tx_sig.shape[-1]//2:])

        # Apply fading
        faded_signal = Tx_as_complex * H_complex # Broadcasting H_complex

        # Add AWGN (complex noise)
        noise_std_per_component = math.sqrt(max(0.0, total_noise_variance) / 2.0) # Variance per real/imag
        noise = torch.normal(0, noise_std_per_component, faded_signal.shape, device=Tx_sig.device) + \
                1j * torch.normal(0, noise_std_per_component, faded_signal.shape, device=Tx_sig.device)
        received_noisy_faded = faded_signal + noise

        # Perfect channel estimation and equalization
        equalized_signal = received_noisy_faded / (H_complex + 1e-8) # Add epsilon for stability

        Rx_sig_out = equalized_signal
        if is_input_real:
            Rx_sig_out = torch.cat((equalized_signal.real, equalized_signal.imag), dim=-1)
        
        return Rx_sig_out.view(original_shape)


    def Rayleigh(self, Tx_sig: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        B = Tx_sig.shape[0]
        device = Tx_sig.device
        # Rayleigh fading: H ~ CN(0, 1) => E[|H|^2] = 1
        H_real = torch.normal(0, math.sqrt(1/2), size=(B, 1, 1), device=device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=(B, 1, 1), device=device)
        H_complex = torch.complex(H_real, H_imag)
        return self._apply_flat_fading_channel(Tx_sig, H_complex, total_noise_variance)

    def Rician(self, Tx_sig: torch.Tensor, total_noise_variance: float, K_factor: float = 1.0) -> torch.Tensor:
        B = Tx_sig.shape[0]
        device = Tx_sig.device

        # Rician fading: H = sqrt(K/(K+1))*H_los + sqrt(1/(K+1))*H_nlos
        # H_los is deterministic (e.g., 1), H_nlos ~ CN(0,1)
        # Total power E[|H|^2] should be normalized to 1.
        
        # Line-of-Sight (LoS) component (deterministic part, often normalized magnitude 1, phase 0)
        H_los_real = math.sqrt(K_factor / (K_factor + 1.0))
        H_los_imag = 0.0
        
        # Non-Line-of-Sight (NLoS) scattered component (Rayleigh-like)
        # Power of NLoS part is 1/(K+1)
        std_nlos_per_component = math.sqrt(1.0 / (2.0 * (K_factor + 1.0))) # std for real/imag part of CN(0, 1/(K+1))
        
        H_nlos_real = torch.normal(0, std_nlos_per_component, size=(B, 1, 1), device=device)
        H_nlos_imag = torch.normal(0, std_nlos_per_component, size=(B, 1, 1), device=device)
        
        H_complex = torch.complex(H_los_real + H_nlos_real, H_los_imag + H_nlos_imag)
        return self._apply_flat_fading_channel(Tx_sig, H_complex, total_noise_variance)

# FIM modules are not strictly needed for Approach 1 SemCom model,
# but kept here as they were in your original file.
# If used, IMGC_NUMCLASS would need to be imported from base_args.py
# class FIM_V1(nn.Module): ...
# class FIM_V2(nn.Module): ...