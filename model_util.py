# model_util.py
import math
from typing import Tuple, Optional
import numpy as np
# from timm.models.registry import register_model # Not registering here, but in model.py
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

# --- (DropPath, Mlp, Attention, Block, PatchEmbed, get_sinusoid_encoding_table, ViTEncoder_Van
#       HierarchicalQuantizer, Channels - should be the same as the last correct versions) ---
# For brevity, I'm only showing ViTDecoder_ImageReconstruction with the CNN head.
# Ensure other classes are correctly defined as in the previous "full code for model_util.py"
# where HierarchicalQuantizer used nn.Embedding.

def _cfg(url='', **kwargs): # Keep _cfg if model.py uses it
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), **kwargs
    }

class DropPath(nn.Module): # ... (as before) ...
    def __init__(self, drop_prob=None): super(DropPath, self).__init__(); self.drop_prob = drop_prob
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)
    def extra_repr(self) -> str: return f'p={self.drop_prob}'

class Mlp(nn.Module): # ... (as before) ...
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(); out_features = out_features or in_features; hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features); self.act = act_layer(); self.fc2 = nn.Linear(hidden_features, out_features); self.drop = nn.Dropout(drop)
    def forward(self, x): x = self.fc1(x); x = self.act(x); x = self.fc2(x); x = self.drop(x); return x

class Attention(nn.Module): # ... (as before, with self.all_head_dim fix) ...
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__(); self.num_heads = num_heads; head_dim = dim // num_heads
        if attn_head_dim is not None: head_dim = attn_head_dim
        self.all_head_dim = head_dim * self.num_heads; self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop); self.proj = nn.Linear(self.all_head_dim, dim); self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape; head_dim_calc = self.all_head_dim // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim_calc).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0); q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.all_head_dim)
        x = self.proj(x); x = self.proj_drop(x); return x

class Block(nn.Module): # ... (as before) ...
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., init_values=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__(); self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(); self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio); self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0: self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)); self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else: self.gamma_1, self.gamma_2 = None, None
    def forward(self, x):
        if self.gamma_1 is None: x = x + self.drop_path(self.attn(self.norm1(x))); x = x + self.drop_path(self.mlp(self.norm2(x)))
        else: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x))); x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module): # ... (as before) ...
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__(); img_size = to_2tuple(img_size); patch_size = to_2tuple(patch_size)
        self.img_size = img_size; self.patch_size = patch_size
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        if not (H == self.img_size[0] and W == self.img_size[1]): raise ValueError(f"Input image size ({H}*{W}) doesn't match model PatchEmbed size ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x).flatten(2).transpose(1, 2); return x

def get_sinusoid_encoding_table(n_position, d_hid, cls_token=False): # ... (as before) ...
    if cls_token: n_position = n_position + 1
    def get_position_angle_vec(position): return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]); sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ViTEncoder_Van(nn.Module): # ... (as before) ...
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.0, use_learnable_pos_emb=False):
        super().__init__(); self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim); self.num_patches = self.patch_embed.num_patches
        if use_learnable_pos_emb: self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim)); trunc_normal_(self.pos_embed, std=.02)
        else: self.register_buffer('pos_embed', get_sinusoid_encoding_table(self.num_patches, embed_dim), persistent=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]; self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values) for i in range(depth)])
        self.norm = norm_layer(embed_dim); self.apply(self._init_weights_vit)
    def _init_weights_vit(self, m):
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
    def forward(self, x_img: torch.Tensor, encoder_boolean_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_tokens = self.patch_embed(x_img); x_tokens = x_tokens + self.pos_embed
        x_proc = x_tokens
        if encoder_boolean_mask is not None and encoder_boolean_mask.any(): # True in mask means MASKED for encoder
            # Assuming B, L, D for x_tokens and B, L for mask
            # This simple selection requires all items in batch to have same number of visible tokens
            # or for x_tokens[~encoder_boolean_mask] to be followed by a padding/unpadding mechanism
            # For mask_ratio = 0, encoder_boolean_mask is all False, ~encoder_boolean_mask is all True, all tokens are kept.
            x_proc = x_tokens[~encoder_boolean_mask].view(x_tokens.shape[0], -1, self.embed_dim)
        for blk in self.blocks: x_proc = blk(x_proc)
        x_proc = self.norm(x_proc); return x_proc


class ViTDecoder_ImageReconstruction(nn.Module):
    def __init__(self, patch_size: int = 16, num_total_patches: int = 196,
                 embed_dim: int = 192, # Input token dimension for decoder's ViT blocks
                 depth: int = 4, num_heads: int = 4, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, norm_layer=nn.LayerNorm, init_values: float = 0.0,
                 out_chans: int = 3):
        super().__init__()
        self.num_total_patches = num_total_patches
        self.patch_size_h, self.patch_size_w = to_2tuple(patch_size)
        self.embed_dim = embed_dim # For ViT blocks

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer('pos_embed_decoder', get_sinusoid_encoding_table(self.num_total_patches, embed_dim), persistent=False)

        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer, init_values=init_values, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # CNN Head
        # Calculate number of upsampling stages (assuming 2x upsampling per stage)
        # e.g., patch_size 16 -> 16x upsample -> log2(16) = 4 stages
        if patch_size == 0: raise ValueError("patch_size cannot be zero.")
        num_upsample_stages = int(math.log2(patch_size))
        if 2**num_upsample_stages != patch_size:
            raise ValueError(f"patch_size ({patch_size}) must be a power of 2 for this simple CNN head design.")

        cnn_layers = []
        current_channels = embed_dim
        
        for i in range(num_upsample_stages):
            out_c = current_channels // 2 if i < num_upsample_stages - 1 else out_chans
            if out_c < out_chans and i < num_upsample_stages -1 : # Ensure intermediate channels don't go below out_chans
                out_c = max(out_chans, current_channels // 2, 16) # Ensure some minimum channels
            
            cnn_layers.append(
                nn.ConvTranspose2d(current_channels, out_c, kernel_size=4, stride=2, padding=1)
            )
            if i < num_upsample_stages - 1: # No activation/norm before final output if it's sigmoid/tanh
                cnn_layers.append(nn.BatchNorm2d(out_c)) # Using BatchNorm for CNNs
                cnn_layers.append(nn.GELU())
            current_channels = out_c
        
        cnn_layers.append(nn.Sigmoid())
        self.cnn_pixel_head = nn.Sequential(*cnn_layers)
        
        trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights_vit)

    def _init_weights_vit(self, m):
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
             trunc_normal_(m.weight, std=.02)
             if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_vis_tokens: torch.Tensor, encoder_mask_boolean: torch.Tensor,
                full_image_num_patches_h: int, full_image_num_patches_w: int,
                ids_restore_if_mae: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N_processed_tokens, D_in = x_vis_tokens.shape
        assert D_in == self.embed_dim, f"Decoder input embed_dim mismatch. Expected {self.embed_dim}, got {D_in}"

        x_full_sequence: torch.Tensor
        pos_embed_expanded = self.pos_embed_decoder.expand(B, -1, -1)

        if not encoder_mask_boolean.any(): # Case 1: No masking by encoder (mask_ratio=0.0)
            if N_processed_tokens != self.num_total_patches:
                raise ValueError(f"Decoder received {N_processed_tokens} tokens but encoder_mask_boolean indicates no masking. Expected {self.num_total_patches} tokens.")
            x_full_sequence = x_vis_tokens + pos_embed_expanded
        
        else: # Case 2: MAE-style masking was done by encoder
            if ids_restore_if_mae is None:
                # This fallback is complex and error-prone if N_vis varies per batch item.
                # It assumes x_vis_tokens are the visible tokens IN THEIR ORIGINAL UNSHUFFLED POSITIONS.
                # And encoder_mask_boolean correctly identifies these.
                print("Warning: MAE-style decoding without 'ids_restore_if_mae'. "
                      "Relying on 'encoder_mask_boolean' to place visible tokens and mask_tokens. "
                      "This path requires encoder to output fixed N_vis tokens or careful handling.")
                
                x_full_sequence = torch.zeros(B, self.num_total_patches, self.embed_dim,
                                              device=x_vis_tokens.device, dtype=x_vis_tokens.dtype)
                
                # Add positional embeddings to all positions first
                x_full_sequence += pos_embed_expanded
                
                # Place mask_tokens at MASKED positions (where encoder_mask_boolean is True)
                x_full_sequence[encoder_mask_boolean] = self.mask_token.squeeze(0).squeeze(0) # Broadcasting mask_token
                
                # Place visible tokens at UNMASKED positions
                # This needs to handle cases where x_vis_tokens might be flattened if num_visible varies
                # Assuming x_vis_tokens is [B, N_vis, D] and N_vis is consistent for the batch
                x_full_sequence[~encoder_mask_boolean] = x_vis_tokens.reshape(-1, self.embed_dim) # Flatten x_vis for assignment

            else: # Proper MAE un-shuffling with ids_restore
                num_masked_patches = self.num_total_patches - N_processed_tokens
                if num_masked_patches < 0: raise ValueError("More visible tokens than total patches based on ids_restore.")

                mask_tokens_to_append = self.mask_token.repeat(B, num_masked_patches, 1)
                x_temp_shuffled_or_partial = torch.cat([x_vis_tokens, mask_tokens_to_append], dim=1)
                
                x_unshuffled = torch.gather(x_temp_shuffled_or_partial, dim=1,
                                            index=ids_restore_if_mae.unsqueeze(-1).expand(-1, -1, D_in))
                x_full_sequence = x_unshuffled + pos_embed_expanded

        decoded_tokens = x_full_sequence
        for blk in self.blocks: decoded_tokens = blk(decoded_tokens)
        decoded_tokens = self.norm(decoded_tokens)

        x_feat_map = decoded_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, full_image_num_patches_h, full_image_num_patches_w
        )
        reconstructed_image = self.cnn_pixel_head(x_feat_map)
        return reconstructed_image


class HierarchicalQuantizer(nn.Module):
    # ... (Keep the corrected HierarchicalQuantizer class from the previous response that uses nn.Embedding) ...
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, hier_init: bool = True, linkage: str = 'ward'):
        super().__init__(); self.num_embeddings = num_embeddings; self.embedding_dim = embedding_dim; self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim); self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.hier_init_done = not hier_init; self.linkage = linkage
    def _initialize_centroids_hierarchical(self, x_flat_cpu_np: np.ndarray):
        if self.hier_init_done: return
        try: from scipy.cluster.hierarchy import linkage as scipy_linkage_func, fcluster
        except ImportError: print("Warning: Scipy not found for HierarchicalQuantizer init."); self.hier_init_done = True; return
        num_samples = x_flat_cpu_np.shape[0]
        if num_samples < 2: print(f"Warning: Not enough samples ({num_samples}) for hierarchical clustering init."); self.hier_init_done = True; return
        num_clusters_for_fcluster = min(num_samples, self.num_embeddings); num_clusters_for_fcluster = max(1, num_clusters_for_fcluster)
        # print(f"HierarchicalQuantizer: Initializing with {num_samples} samples, targeting {self.num_embeddings} embeddings, will form {num_clusters_for_fcluster} initial clusters via fcluster.")
        Z = scipy_linkage_func(x_flat_cpu_np, method=self.linkage)
        cluster_labels_from_fcluster = fcluster(Z, num_clusters_for_fcluster, criterion='maxclust')
        centroids = np.zeros((self.num_embeddings, self.embedding_dim), dtype=x_flat_cpu_np.dtype); unique_fcluster_labels = np.unique(cluster_labels_from_fcluster); num_actual_clusters_formed = len(unique_fcluster_labels)
        for i in range(num_actual_clusters_formed):
            centroid_idx = i; current_fcluster_label_val = unique_fcluster_labels[i]; current_cluster_mask = (cluster_labels_from_fcluster == current_fcluster_label_val)
            if np.any(current_cluster_mask): centroids[centroid_idx] = x_flat_cpu_np[current_cluster_mask].mean(axis=0)
            else:
                if num_samples > 0: centroids[centroid_idx] = x_flat_cpu_np[np.random.randint(num_samples)]
        if num_actual_clusters_formed < self.num_embeddings:
            # print(f"  Hierarchical clustering formed {num_actual_clusters_formed} distinct centroids. Expected {self.num_embeddings}. Filling remaining...")
            num_to_fill_additionally = self.num_embeddings - num_actual_clusters_formed
            if num_samples > 0: fill_indices = np.random.choice(num_samples, num_to_fill_additionally, replace=True); centroids[num_actual_clusters_formed:] = x_flat_cpu_np[fill_indices]
        self.embedding.weight.data.copy_(torch.from_numpy(centroids).to(self.embedding.weight.device))
        print("HierarchicalQuantizer: Centroids initialized using hierarchical clustering.")
        self.hier_init_done = True
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape; assert x.shape[-1] == self.embedding_dim; x_flat = x.reshape(-1, self.embedding_dim)
        if not self.hier_init_done and self.training and x_flat.shape[0] > 1 : self._initialize_centroids_hierarchical(x_flat.detach().cpu().numpy())
        distances_sq = (torch.sum(x_flat**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(x_flat, self.embedding.weight.t()))
        distances_sq = torch.clamp(distances_sq, min=0.0); encoding_indices = torch.argmin(distances_sq, dim=1)
        quantized_flat = self.embedding(encoding_indices); quantized_output_tokens = quantized_flat.view(original_shape)
        codebook_loss = F.mse_loss(quantized_output_tokens.detach(), x) # Corrected VQVAE loss (orig DeepMind)
        commitment_loss = F.mse_loss(x, quantized_output_tokens.detach()) # Corrected VQVAE loss
        vq_total_loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized_output_tokens_st = x + (quantized_output_tokens - x).detach()
        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float(); avg_probs = torch.mean(encodings_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))); return quantized_output_tokens_st, vq_total_loss, perplexity, encoding_indices

class Channels(nn.Module):
    # ... (Keep the Channels class exactly as it was in the previous full code response) ...
    def __init__(self): super().__init__()
    def _get_noise_std(self, total_noise_variance: float) -> float: return math.sqrt(max(0.0, total_noise_variance))
    def AWGN(self, Tx_sig: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        device = Tx_sig.device; noise_std = self._get_noise_std(total_noise_variance)
        if torch.is_complex(Tx_sig): noise_std_per_component = math.sqrt(max(0.0, total_noise_variance) / 2.0); noise = torch.normal(0, noise_std_per_component, size=Tx_sig.shape, device=device) + 1j * torch.normal(0, noise_std_per_component, size=Tx_sig.shape, device=device)
        else: noise = torch.normal(0, noise_std, size=Tx_sig.shape, device=device)
        return Tx_sig + noise
    def _apply_flat_fading_channel(self, Tx_sig: torch.Tensor, H_complex: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        is_input_real = not torch.is_complex(Tx_sig); original_shape = Tx_sig.shape; Tx_as_complex = Tx_sig
        if is_input_real:
            if Tx_sig.shape[-1] % 2 != 0: Tx_sig = F.pad(Tx_sig, (0,1)); Tx_as_complex = torch.complex(Tx_sig[...,:Tx_sig.shape[-1]//2], Tx_sig[...,Tx_sig.shape[-1]//2:]) # Pad if odd for complex conv
            else: Tx_as_complex = torch.complex(Tx_sig[..., :Tx_sig.shape[-1]//2], Tx_sig[..., Tx_sig.shape[-1]//2:])
        faded_signal = Tx_as_complex * H_complex
        noise_std_per_component = math.sqrt(max(0.0, total_noise_variance) / 2.0)
        noise = torch.normal(0, noise_std_per_component, faded_signal.shape, device=Tx_sig.device) + 1j * torch.normal(0, noise_std_per_component, faded_signal.shape, device=Tx_sig.device)
        received_noisy_faded = faded_signal + noise
        equalized_signal = received_noisy_faded / (H_complex + 1e-8)
        Rx_sig_out = equalized_signal
        if is_input_real: Rx_sig_out = torch.cat((equalized_signal.real, equalized_signal.imag), dim=-1)
        return Rx_sig_out.view(original_shape) # Ensure original shape, esp if padding was done
    def Rayleigh(self, Tx_sig: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        B = Tx_sig.shape[0]; device = Tx_sig.device; H_real = torch.normal(0, math.sqrt(1/2), size=(B, 1, 1), device=device); H_imag = torch.normal(0, math.sqrt(1/2), size=(B, 1, 1), device=device)
        H_complex = torch.complex(H_real, H_imag); return self._apply_flat_fading_channel(Tx_sig, H_complex, total_noise_variance)
    def Rician(self, Tx_sig: torch.Tensor, total_noise_variance: float, K_factor: float = 1.0) -> torch.Tensor:
        B = Tx_sig.shape[0]; device = Tx_sig.device; H_los_real = math.sqrt(K_factor / (K_factor + 1.0)); H_los_imag = 0.0
        std_nlos_per_component = math.sqrt(1.0 / (2.0 * (K_factor + 1.0)))
        H_nlos_real = torch.normal(0, std_nlos_per_component, size=(B, 1, 1), device=device); H_nlos_imag = torch.normal(0, std_nlos_per_component, size=(B, 1, 1), device=device) # Centered at 0
        H_complex = torch.complex(H_los_real + H_nlos_real, H_los_imag + H_nlos_imag); return self._apply_flat_fading_channel(Tx_sig, H_complex, total_noise_variance)