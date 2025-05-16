# model.py
import math
from typing import Dict # For type hinting
import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model
# import inspect # Removed inspect

from model_util import (
    ViTEncoder_Van, ViTDecoder_ImageReconstruction,
    HierarchicalQuantizer, Channels, _cfg # Assuming PatchEmbed is in model_util if ViTEncoder_Van uses it
)

# print("!!! MODEL.PY BEING EXECUTED - CLEAN VERSION !!!") # Optional: Keep for one last check, then remove

class ViT_Reconstruction_Model(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 encoder_in_chans: int = 3,
                 encoder_embed_dim: int = 768,
                 encoder_depth: int = 12,
                 encoder_num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 init_values: float = 0.0,
                 use_learnable_pos_emb: bool = False,
                 quantizer_dim: int = 256,
                 bits_for_quantizer: int = 8,
                 quantizer_commitment_cost: float = 0.25,
                 **kwargs):
        super().__init__()

        effective_patch_size = patch_size

        self.img_encoder = ViTEncoder_Van(
            img_size=img_size,
            patch_size=effective_patch_size,
            in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.full_image_num_patches_h = self.img_encoder.patch_embed.patch_shape[0]
        self.full_image_num_patches_w = self.img_encoder.patch_embed.patch_shape[1]
        num_total_patches_in_image = self.img_encoder.patch_embed.num_patches

        self.img_decoder = ViTDecoder_ImageReconstruction(
            patch_size=effective_patch_size,
            num_total_patches=num_total_patches_in_image,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            init_values=init_values
        )

        self.encoder_to_channel_proj = nn.Linear(encoder_embed_dim, quantizer_dim)
        self.channel_simulator = Channels()
        self.channel_to_decoder_proj = nn.Linear(quantizer_dim, decoder_embed_dim)

        self.num_bits_for_vq = bits_for_quantizer
        self.quantizer = HierarchicalQuantizer(
            num_embeddings=2**self.num_bits_for_vq,
            embedding_dim=quantizer_dim,
            commitment_cost=quantizer_commitment_cost
        )
        self.current_vq_loss = torch.tensor(0.0, dtype=torch.float32)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self,
                img: torch.Tensor,
                bm_pos: torch.Tensor = None,
                targets=None, # Not used by this specific forward for reconstruction
                _eval: bool = False,
                eval_snr_db: float = 30.0,
                train_snr_db_min: float = 30,
                train_snr_db_max: float = 30.0,
                **kwargs # Catch any other unexpected kwargs
               ) -> Dict[str, torch.Tensor]:

        if kwargs: # Log if any unexpected kwargs are still passed
            print(f"Warning: ViT_Reconstruction_Model.forward() received unexpected kwargs: {kwargs}")

        is_currently_training = self.training and not _eval

        if bm_pos is None:
            encoder_input_mask_bool = torch.zeros(
                img.shape[0], self.img_encoder.patch_embed.num_patches,
                dtype=torch.bool, device=img.device
            )
        else:
            encoder_input_mask_bool = bm_pos

        x_encoded_tokens = self.img_encoder(img, encoder_input_mask_bool)
        x_for_quantizer = self.encoder_to_channel_proj(x_encoded_tokens)
        quantized_tokens_st, vq_c_loss, _, _ = self.quantizer(x_for_quantizer)
        self.current_vq_loss = vq_c_loss

        if is_currently_training:
            current_snr_db_tensor = torch.rand(1, device=img.device) * \
                                    (train_snr_db_max - train_snr_db_min) + train_snr_db_min
        else:
            current_snr_db_tensor = torch.tensor(eval_snr_db, device=img.device)
        
        noise_power_variance = 10**(-current_snr_db_tensor / 10.0)
        tokens_after_channel = self.channel_simulator.Rayleigh(quantized_tokens_st, noise_power_variance.item())
        x_for_decoder_input = self.channel_to_decoder_proj(tokens_after_channel)
        
        reconstructed_image = self.img_decoder(
            x_vis_tokens=x_for_decoder_input,
            encoder_mask_boolean=encoder_input_mask_bool,
            full_image_num_patches_h=self.full_image_num_patches_h,
            full_image_num_patches_w=self.full_image_num_patches_w,
            ids_restore_if_mae=None
        )

        output_dict = {
            'reconstructed_image': reconstructed_image,
            'vq_loss': self.current_vq_loss
        }
        return output_dict


@register_model
def ViT_Reconstruction_Model_Default(pretrained: bool = False, **kwargs) -> ViT_Reconstruction_Model:
    model_defaults = dict(
        patch_size=16,
        encoder_in_chans=3,
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3,
        mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        quantizer_dim=256, bits_for_quantizer=8, quantizer_commitment_cost=0.25,
        init_values=0.0, use_learnable_pos_emb=False,
        drop_rate=0.0, drop_path_rate=0.1
    )
    
    final_model_constructor_args = model_defaults.copy()
    if 'img_size' in kwargs: resolved_img_size = kwargs['img_size']
    elif 'input_size' in kwargs: resolved_img_size = kwargs['input_size']
    else: resolved_img_size = 224 # Fallback if create_model doesn't pass it
    final_model_constructor_args['img_size'] = resolved_img_size

    # Override defaults with relevant keys from kwargs passed by timm.create_model
    relevant_kwargs_for_model_init = list(model_defaults.keys()) + ['img_size'] # include img_size
    for key in relevant_kwargs_for_model_init:
        if key in kwargs: # If timm.create_model passes it as a kwarg
            final_model_constructor_args[key] = kwargs[key]
    
    model = ViT_Reconstruction_Model(**final_model_constructor_args)
    model.default_cfg = _cfg()
    if pretrained:
        print("Warning: `pretrained=True` for ViT_Reconstruction_Model_Default, but no loading logic implemented.")
    return model