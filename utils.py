import io
import os
import csv
import math
from typing import Iterable
inf = math.inf
import time
import json
import thop
import torch
import datetime
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from timm.utils import get_state_dict
from timm.models import create_model
from collections import OrderedDict
from pytorch_msssim import ms_ssim, ssim
from collections import defaultdict, deque
from timm.loss import LabelSmoothingCrossEntropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, f1_score
# model.py
import math
import torch
# import pickle # Not used
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from channel import * # Using Channels from model_util
from model_util import (
    ViTEncoder_Van, ViTDecoder_ImageReconstruction, HierarchicalQuantizer, Channels,_cfg, # Keep FIM parts if you might add an auxiliary FIM loss
    PatchEmbed # ViTDecoder_Van was removed, using ViTDecoder_ImageReconstruction
)
from functools import partial
from timm.models.registry import register_model  # Import register_model from timm.models.registry
# from timm.models.layers import trunc_normal_ as __call_trunc_normal_ # Already in model_util
# from typing import List, Callable, Union, Any, TypeVar, Tuple # Not strictly needed here
from base_args import IMGC_NUMCLASS # Still needed if FIM aux loss is used
from pytorch_msssim import ms_ssim, ssim # For SSIM loss if needed
# def trunc_normal_(tensor, mean=0., std=1.): # Already in model_util
#     __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

# __all__ = ['ViT_Reconstruction_Model', 'ViT_FIM_model_for_AuxLoss'] # Example new names

class ViT_Reconstruction_Model(nn.Module): # Renamed from ViT_Van_CLS
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
                 use_learnable_pos_emb=False, num_classes=0, # num_classes not used for head
                 quantizer_dim=32, # Dimension for features going into/out of quantizer
                 **kwargs):
        super().__init__()
        self.img_encoder = ViTEncoder_Van(
            img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        # Determine total number of patches from the encoder's patch_embed
        num_total_patches = self.img_encoder.patch_embed.num_patches
        self.full_image_num_patches_h = self.img_encoder.patch_embed.patch_shape[0]
        self.full_image_num_patches_w = self.img_encoder.patch_embed.patch_shape[1]


        self.img_decoder = ViTDecoder_ImageReconstruction(
            patch_size=patch_size, num_total_patches=num_total_patches,
            embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, init_values=init_values
        )

        self.encoder_to_channel = nn.Linear(encoder_embed_dim, quantizer_dim)
        self.channel = Channels() # For AWGN, Rayleigh etc.
        self.channel_to_decoder = nn.Linear(quantizer_dim, decoder_embed_dim)

        self.bit_per_digit = 8 # For quantizer
        self.vq_layer = HierarchicalQuantizer(
            num_embeddings=2**self.bit_per_digit,
            embedding_dim=quantizer_dim, # Must match output of encoder_to_channel
            commitment_cost=0.25
        )
        self.vq_loss_value = None # To store vq_loss from forward

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use Timm's trunc_normal_ for consistency if preferred, or xavier_uniform_
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, bm_pos=None, targets=None, _eval=False, test_snr=10.0): # bm_pos might be optional
        # bm_pos: [B, NumTotalPatches], True if masked at ENCODER input
        # If bm_pos is None, assume all patches are processed by encoder
        is_training = self.training

        # 1. Encode visible patches
        # For reconstruction, if not doing MAE-style pretraining, bm_pos might be all False
        # or encoder processes full image.
        if bm_pos is None: # If no mask provided, create a "no mask"
            bm_pos_encoder = torch.zeros(img.shape[0], self.img_encoder.patch_embed.num_patches,
                                dtype=torch.bool, device=img.device)
        else:
            bm_pos_encoder = bm_pos # Use provided mask for encoder

        # x_vis_encoded: features of *only* the patches processed by encoder
        x_vis_encoded = self.img_encoder(img, bm_pos_encoder) # [B, NumVisiblePatches, EncoderEmbedDim]

        # 2. Project to channel dimension
        x_for_channel = self.encoder_to_channel(x_vis_encoded) # [B, NumVisiblePatches, QuantizerDim]

        # 3. Quantize
        quantized_x_for_channel, vq_loss, _, _ = self.vq_layer(x_for_channel)
        self.vq_loss_value = vq_loss # Store for loss calculation

        # 4. Simulate Channel
        # Note: Channel simulation on quantized_x_for_channel.
        # The SNR and channel type should be managed in engine.py more flexibly.
        if is_training:
            # Simplified: use a fixed SNR or a random one during training from a range
            # noise_var = torch.FloatTensor([1]).to(img.device) * 10**(-10.0/20) # Example: 10dB SNR
            snr_db_train = torch.rand(1).to(img.device) * 20 - 5 # SNR from -5 to 15 dB
            noise_var = torch.FloatTensor([1]).to(img.device) * 10**(-snr_db_train/20)
        else: # Evaluation
            noise_var = torch.FloatTensor([1]).to(img.device) * 10**(-test_snr/20)

        # Assuming self.channel.Rayleigh expects real-valued tensor where last dim is even for complex
        # Or directly handles complex tensors. Let's assume it handles real for now.
        if quantized_x_for_channel.shape[-1] % 2 !=0 and not torch.is_complex(quantized_x_for_channel):
             # Pad if odd for simple real to complex split, or ensure quantizer_dim is even
             # This is a simplification. Proper complex handling is better.
             padded_quantized_x = F.pad(quantized_x_for_channel, (0,1)) # Pad last dim
             x_after_channel_real = self.channel.Rayleigh(padded_quantized_x, noise_var.item())
             x_after_channel = x_after_channel_real[..., :-1] # Remove padding
        else:
            x_after_channel = self.channel.Rayleigh(quantized_x_for_channel, noise_var.item())


        # 5. Project to decoder dimension
        x_for_decoder = self.channel_to_decoder(x_after_channel) # [B, NumVisiblePatches, DecoderEmbedDim]

        # 6. Decode to image
        # The decoder needs to know which patches were originally masked at the encoder to insert mask tokens
        # and the total grid size.
        reconstructed_image = self.img_decoder(
            x_for_decoder, # Features of visible patches
            bm_pos_encoder,  # Mask used by the encoder
            self.full_image_num_patches_h,
            self.full_image_num_patches_w
        )

        out = {}
        out['reconstructed_image'] = reconstructed_image
        out['vq_loss'] = self.vq_loss_value # Pass VQ loss out
        return out

# Keep ViT_FIM_CLS if you plan to use it for some auxiliary task or comparison
# but ensure its forward method is compatible or adapt it.
# For Approach 1, ViT_Reconstruction_Model is the primary one.
class ViT_FIM_CLS(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0,
                 encoder_embed_dim=768, encoder_depth=12,encoder_num_heads=12, decoder_num_classes=768, # decoder_num_classes here is for the ViT decoder block, not final cls
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, # This num_classes is for final head
                 quantizer_dim=32,
                 **kwargs):
        super().__init__()
        self.img_encoder = ViTEncoder_FIM(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans,
                                num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb) # ViTEncoder_FIM will produce cls_out

        # This decoder is a generic ViT decoder, not necessarily for image reconstruction.
        # If this model is also meant for reconstruction, it needs ViTDecoder_ImageReconstruction
        self.img_decoder_generic = ViTDecoder_Van( # Using the original ViTDecoder_Van
            patch_size=patch_size,
            num_patches=self.img_encoder.patch_embed.num_patches, # num_patches from encoder
            embed_dim=decoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,init_values=init_values,
            num_classes=decoder_num_classes # Output dim of this decoder block
        )


        self.encoder_to_channel = nn.Linear(encoder_embed_dim, quantizer_dim)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(quantizer_dim, decoder_embed_dim)
        self.classification_head = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS) # CLASSIFICATION head
        self.bit_per_digit = 8
        self.vq_layer = HierarchicalQuantizer(
            num_embeddings=2**self.bit_per_digit,
            embedding_dim=quantizer_dim,
            commitment_cost=0.25
        )
        self.vq_loss_value = None
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, img, bm_pos, targets=None, _eval=False, test_snr=10.0):
        is_training = self.training
        out = {}

        # Encoder produces main features (x_encoded) and auxiliary FIM classifier outputs (cls_out_fim)
        x_encoded, cls_out_fim = self.img_encoder(img, bm_pos, targets if is_training else None) # Pass targets only for training FIM

        x_for_channel = self.encoder_to_channel(x_encoded)
        quantized_x, vq_loss, _, _ = self.vq_layer(x_for_channel)
        self.vq_loss_value = vq_loss

        if is_training:
            snr_db_train = torch.rand(1).to(img.device) * 20 - 5
            noise_var = torch.FloatTensor([1]).to(img.device) * 10**(-snr_db_train/20)
        else:
            noise_var = torch.FloatTensor([1]).to(img.device) * 10**(-test_snr/20)
        
        if quantized_x.shape[-1] % 2 !=0 and not torch.is_complex(quantized_x):
             padded_quantized_x = F.pad(quantized_x, (0,1))
             x_after_channel_real = self.channel.Rayleigh(padded_quantized_x, noise_var.item())
             x_after_channel = x_after_channel_real[..., :-1]
        else:
            x_after_channel = self.channel.Rayleigh(quantized_x, noise_var.item())

        x_for_decoder = self.channel_to_decoder(x_after_channel)
        
        # This generic decoder outputs features, then classification head
        # For approach 1, you would NOT use this path for the YOLO input.
        # This path is for if ViT_FIM_CLS is still doing some classification.
        decoded_features = self.img_decoder_generic(x_for_decoder) # Pass bm_pos if MAE style for decoder
        classification_logits = self.classification_head(decoded_features.mean(1)) # Example: average and classify

        out['out_x'] = classification_logits # Main classification output
        out['out_c'] = cls_out_fim         # Auxiliary FIM classification outputs
        out['vq_loss'] = self.vq_loss_value
        # For Approach 1, you'd also want to output something that can be turned into an image
        # This would require adding an ViTDecoder_ImageReconstruction path here as well, or
        # making this model purely for classification and using ViT_Reconstruction_Model for Approach 1.
        # Let's assume for clarity, ViT_FIM_CLS is *not* the model used for Approach 1's image reconstruction.
        return out

# Registration functions - update model names if you renamed them
@register_model
def ViT_Reconstruction_Model_Default(pretrained: bool = False, **kwargs) -> 'ViT_Reconstruction_Model': # Forward ref if class below
    print(f"--- [DEBUG REG FN] Inside ViT_Reconstruction_Model_Default registration function ---")
    print(f"  [DEBUG REG FN] Received **kwargs from timm.create_model: {kwargs}")

    # These are the base defaults for your model's architecture parameters
    # These can be overridden by what's passed in **kwargs from timm.create_model
    model_constructor_args = {
        'patch_size': 16,
        'encoder_in_chans': 3,
        'encoder_embed_dim': 384, 'encoder_depth': 6, 'encoder_num_heads': 6,
        'decoder_embed_dim': 192, 'decoder_depth': 3, 'decoder_num_heads': 3,
        'mlp_ratio': 4.0, 'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'quantizer_dim': 64, 'bits_for_quantizer': 8, 'quantizer_commitment_cost': 0.25,
        'init_values': 0.0, 'use_learnable_pos_emb': False,
        'drop_rate': 0.0, 'drop_path_rate': 0.1 # Base defaults for these
    }

    # Determine the image size
    # Prioritize 'img_size' if directly passed in kwargs,
    # then 'input_size' from kwargs, then a hardcoded default.
    if 'img_size' in kwargs:
        resolved_img_size = kwargs['img_size']
        print(f"  [DEBUG REG FN] Using 'img_size' from kwargs: {resolved_img_size}")
    elif 'input_size' in kwargs:
        resolved_img_size = kwargs['input_size']
        print(f"  [DEBUG REG FN] Using 'input_size' from kwargs for img_size: {resolved_img_size}")
    else:
        resolved_img_size = 224 # Fallback default
        print(f"  [DEBUG REG FN] Warning: Neither 'img_size' nor 'input_size' in kwargs. Defaulting img_size to {resolved_img_size}")
    
    model_constructor_args['img_size'] = resolved_img_size

    # Override other defaults with values from kwargs if they exist
    # For example, if 'drop_rate' is in kwargs, it will override the default 0.0
    for key in model_constructor_args.keys():
        if key in kwargs:
            model_constructor_args[key] = kwargs[key]
    
    # Add any other kwargs that are not model constructor args but timm might pass (like pretrained_cfg)
    # model_constructor_args.update(kwargs) # This was too broad and could re-introduce conflicts
    # Instead, only pass known args to your model constructor.
    # If your ViT_Reconstruction_Model.__init__ takes **kwargs, this is less of an issue.
    # Let's assume ViT_Reconstruction_Model.__init__ explicitly lists all its params.

    print(f"  [DEBUG REG FN] Final args for ViT_Reconstruction_Model constructor: {model_constructor_args}")
    
    # Assuming ViT_Reconstruction_Model is defined in the same file or imported
    model = ViT_Reconstruction_Model(**model_constructor_args)
    model.default_cfg = _cfg() # For timm compatibility
    if pretrained:
        print("Warning: `pretrained=True` for ViT_Reconstruction_Model_Default, but no loading logic.")
    return model

@register_model
def ViT_FIM_model_S(pretrained = False,**kwargs): # This remains a classification model
    model = ViT_FIM_CLS(
        img_size=kwargs.get('input_size', 224),
        patch_size=16,
        encoder_embed_dim=384, encoder_depth=4, encoder_num_heads=6,
        decoder_embed_dim=192, decoder_depth=1, decoder_num_heads=4, # Decoder for features for classification head
        decoder_num_classes=192, # output dim of the ViT decoder block
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        quantizer_dim=32,
        **kwargs)
    model.default_cfg = _cfg()
    # Pretrained loading for FIM model
    return model

## Including pakages
def sel_criterion(args): # NO 'task_type' argument here
    # For Approach 1, the SemCom model is trained for image reconstruction
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss() # Alternative reconstruction loss
    print("Criterion for Image RECONSTRUCTION (SemCom Training) = %s" % (str(criterion)))
    return criterion

def get_model(args):
    print(f"Creating model: {args.model} with input_size: {args.input_size}")
    from timm.models import create_model as timm_create_model

    model_kwargs = {
        'input_size': args.input_size,
        'img_size': args.input_size,
        # 'drop_rate': args.drop_rate,       # OLD - incorrect argument name
        'drop_rate': args.drop_path,         # NEW - use args.drop_path for the model's drop_rate parameter
                                             # OR if your model specifically needs a separate drop_rate,
                                             # add --drop_rate to base_args.py
                                             # Many timm models use 'drop_rate' for MLP dropout and
                                             # 'drop_path_rate' for stochastic depth.
                                             # Let's assume your model expects drop_rate for MLP/Proj dropout
                                             # and you want to control it with args.drop_path for now.
                                             # A cleaner way is to have distinct args if they control distinct things.
                                             # For now, let's assume args.drop_path controls the general dropout for blocks.
                                             # ViT_Reconstruction_Model takes 'drop_rate' and 'drop_path_rate'.
        'drop_path_rate': args.drop_path, # This is correct as per base_args.py
        # If you want a separate MLP/Attention dropout, add a --mlp_drop_rate arg to base_args.py
        # and pass it as 'drop_rate': args.mlp_drop_rate
        # For simplicity, let's assume you want to use args.drop_path for the general drop_rate as well for now.
        # If your model has specific `drop_rate` and `drop_path_rate` parameters in its __init__,
        # and you want to control them separately:
        # Add --drop_rate to base_args.py:
        # parser.add_argument('--drop_rate', type=float, default=0.0, help='Dropout rate for MLP/Attention')
        # Then in model_kwargs:
        # 'drop_rate': args.drop_rate,
        # 'drop_path_rate': args.drop_path,
    }
    # To be safe and explicit for ViT_Reconstruction_Model_Default which takes both:
    if not hasattr(args, 'drop_rate'): # If --drop_rate wasn't added to base_args.py
        args.drop_rate = 0.0 # Provide a default if the model expects it
        print(f"  Note: 'args.drop_rate' not found, defaulting model's 'drop_rate' to {args.drop_rate}")


    model_kwargs['drop_rate'] = args.drop_rate # Use the (potentially new) args.drop_rate
    # model_kwargs['drop_path_rate'] is already args.drop_path


    print(f"  kwargs for timm.create_model: {model_kwargs}")

    model = timm_create_model(
        args.model,
        pretrained=False,
        **model_kwargs
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'=> Number of params: {n_parameters / 1e6:.2f} M')
    return model

def load_checkpoint(model,args):
    
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

    print("Load ckpt from the place")
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('encoder.'):
            new_dict['img_'+key] = checkpoint_model[key]
        # elif key.startswith('img_encoder.blocks.3'):
        #     new_dict['img_encoder.blocks_cas.0'+key[20:]] = checkpoint_model[key]
        # elif key.startswith('img_encoder.blocks.3'):
        #     new_dict['img_encoder.blocks_cas.1'+key[20:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict
    return checkpoint_model


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

class NativeScalerWithGradNormCount: # For AMP
    state_dict_key = "amp_scaler"

    def __init__(self): # REMOVE device_type argument here
        # Use the non-deprecated torch.amp.GradScaler
        # Enable it only if cuda is available and intended.
        # The scaler object itself should be created regardless,
        # but its `enabled` flag controls its operation.
        self._scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    def __call__(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                 parameters: Iterable[torch.nn.Parameter],
                 clip_grad: float = None,
                 update_grad: bool = True,
                 create_graph: bool = False
                ):
        
        # Only scale/unscale if scaler is enabled (i.e., on CUDA and AMP is active)
        if self._scaler.is_enabled():
            self._scaler.scale(loss).backward(create_graph=create_graph)
        else: # If not enabled (e.g. running on CPU), just do a normal backward pass
            loss.backward(create_graph=create_graph)
        
        norm = None
        if update_grad:
            if self._scaler.is_enabled():
                self._scaler.unscale_(optimizer) # Unscale before clipping, only if enabled

            if clip_grad is not None and clip_grad > 0:
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            # else:
                # norm = get_grad_norm_(parameters) # Optional: compute norm even if not clipping

            if self._scaler.is_enabled():
                self._scaler.step(optimizer)
                self._scaler.update()
            else: # If not enabled, just a normal optimizer step
                optimizer.step()
            
            optimizer.zero_grad()

        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

# Definition for get_grad_norm_ (if you want to use it for logging grad_norm)
# def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = [p for p in parameters if p.grad is not None]
#     norm_type = float(norm_type)
#     if len(parameters) == 0:
#         return torch.tensor(0.)
#     device = parameters[0].grad.device
#     if norm_type == inf: # Make sure inf is defined (from math or torch._six)
#         import math
#         total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
#     else:
#         total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
#     return total_norm


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def path_exists_make(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    # output_dir = Path(args.output_dir+'/ckpt_'+args.train_type)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_exists_make(output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor
    
    
def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2.0, dtype=torch.float32)

    if mse == 0:
        return torch.tensor([100.0])

    PIXEL_MAX = 255.0

    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def get_imagenet_list(path):
    fns = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fns.append(row[0])
    
    return fns

def complex_sig(shape, device):
        sig_real = torch.randn(*shape)
        sig_imag = torch.randn(*shape)
        return (torch.complex(sig_real, sig_imag)/np.sqrt(2)).to(device)

def pwr_normalize(sig):
    _, num_ele = sig.shape[0], torch.numel(sig[0])
    pwr_sig = torch.sum(torch.abs(sig)**2, dim=-1)/num_ele
    sig = sig/torch.sqrt(pwr_sig.unsqueeze(-1))
    return sig

def np_to_torch(img):
    img = np.swapaxes(img, 0, 1)  # w, h, c
    img = np.swapaxes(img, 0, 2)  # c, h, w
    return torch.from_numpy(img).float()

def to_chan_last(img):
    img = img.transpose(1, 2)
    img = img.transpose(2, 3)
    return img

def as_img_array(image: torch.Tensor) -> torch.Tensor:
    # This function should output tensors in [0, 255] range, float32 for ssim
    # The input image from your reconstruction is likely [0,1] float.
    # If it's already float, ensure it is, then multiply.
    if image.dtype != torch.float32:
        image = image.float() # Ensure float32
    image = torch.clamp(image * 255.0, 0, 255) # Scale to 0-255 and clamp
    return torch.round(image) # Round to nearest integer, still float

def calc_psnr(predictions: torch.Tensor, targets: torch.Tensor) -> list:
    # Assuming predictions and targets are already detached and on CPU
    # and are in [0,1] float range initially from model output
    if predictions.ndim == 3: predictions = predictions.unsqueeze(0)
    if targets.ndim == 3: targets = targets.unsqueeze(0)

    predictions_uint8 = as_img_array(predictions.float()) # Ensure float32 before scaling
    targets_uint8 = as_img_array(targets.float())     # Ensure float32 before scaling
    
    mse = torch.mean((predictions_uint8 - targets_uint8) ** 2.0, dim=(1, 2, 3)) # Keep as float for calculation
    # Handle pure black images or perfect matches to avoid log(0) or division by zero
    psnr_val = torch.where(mse == 0, torch.tensor(100.0, device=mse.device), 20 * torch.log10(255.0 / torch.sqrt(mse)))
    return psnr_val.tolist()


def calc_ssim(predictions: torch.Tensor, targets: torch.Tensor) -> list:
    # Assuming predictions and targets are already detached and on CPU
    # and are in [0,1] float range initially from model output

    # Ensure inputs to ssim are B, C, H, W and float32, range [0, 255] for data_range=255
    if predictions.ndim == 3: predictions = predictions.unsqueeze(0) # Add batch dim if missing
    if targets.ndim == 3: targets = targets.unsqueeze(0)

    # Convert to float32, scale to [0, 255]
    # The ssim function expects floating point numbers.
    pred_float32 = predictions.float()
    targ_float32 = targets.float()

    # The pytorch_msssim library internally handles scaling if data_range is set
    # and expects inputs to be in a typical image range (e.g. 0-1 or 0-255).
    # If your input is [0,1], use data_range=1.0
    # If your input is [0,255] (after scaling), use data_range=255.0
    # Since as_img_array scales to 0-255, let's use that.
    
    pred_for_ssim = as_img_array(pred_float32) # Now [0,255] float32
    targ_for_ssim = as_img_array(targ_float32) # Now [0,255] float32

    # The ssim function from pytorch_msssim should handle dtype consistency for its window
    # if the input dtypes are consistent.
    try:
        ssim_val = ssim(pred_for_ssim, targ_for_ssim, data_range=255.0, size_average=False, nonnegative_ssim=True)
        # nonnegative_ssim=True can help avoid negative SSIM values that sometimes occur.
    except RuntimeError as e:
        print(f"RuntimeError during SSIM calculation: {e}")
        print(f"Pred Dtype: {pred_for_ssim.dtype}, Pred Device: {pred_for_ssim.device}")
        print(f"Targ Dtype: {targ_for_ssim.dtype}, Targ Device: {targ_for_ssim.device}")
        # As a fallback, try casting the window inside if possible, or just return 0
        # This usually means the internal window is not matching.
        # Try to ensure inputs are float32 for ssim library.
        ssim_val = torch.zeros(predictions.shape[0], device=predictions.device)


    return ssim_val.tolist()

# calc_msssim would need similar treatment if you use it.
def calc_msssim(predictions: torch.Tensor, targets: torch.Tensor) -> list:
    if predictions.ndim == 3: predictions = predictions.unsqueeze(0)
    if targets.ndim == 3: targets = targets.unsqueeze(0)
    pred_float32 = predictions.float()
    targ_float_32 = targets.float()
    pred_for_msssim = as_img_array(pred_float32)
    targ_for_msssim = as_img_array(targ_float_32)
    try:
        msssim_val = ms_ssim(pred_for_msssim, targ_for_msssim, data_range=255.0, size_average=False)
    except RuntimeError as e:
        print(f"RuntimeError during MS-SSIM calculation: {e}")
        msssim_val = torch.zeros(predictions.shape[0], device=predictions.device)
    return msssim_val.tolist()

import nltk
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokens2sentence(outputs):
    sentences = []
    #print(outputs)
    for tokens in outputs:
        sentence = []
        for token in tokens:
            
            word = tokenizer.decode([int(token)])
 
            if word == '[PAD]':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences  
 
def computebleu(sentences, targets):
  score = 0 
  assert (len(sentences) == len(targets))
  def cut_token(sentence):
    tmp = []
    for token in sentence:
      if token == '[UNK]':
        tmp.append(token)
      else:
        tmp += [word for word in token]
    return tmp 

  for sentence, target in zip(sentences, targets):
    sentence = cut_token(sentence)
   
    target = cut_token(target)

    score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))                                                                                          
  return score



def calc_metrics(y_true, y_pred, mode=None, to_print=True):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """
    def multiclass_acc(preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    
    test_preds = y_pred
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    
    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    # pos - neg
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    if to_print:
        # print("mae: ", mae)
        # print("corr: ", corr)
        # print("mult_acc: ", mult_a7)
        print("Classification Report (pos/neg) :")
        # print(classification_report(binary_truth, binary_preds, digits=5))
        print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
        
        # non-neg - neg
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        if to_print:
            print("Classification Report (non-neg/neg) :")
            # print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
        
        return accuracy_score(binary_truth, binary_preds)
    
    
class DiffPruningLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, dynamic=True, ratio_weight=2.0, main_weight=1.):
        super().__init__()
        self.base_criterion = base_criterion
        self.main_weight = 1.
        self.surp_weight = 0.022
        self.rho_weight = 0.01    
        self.vq_weight = 2.0    
        self.print_mode = True
        
        self.count = 0
        self.main_loss_record = 0.
        self.surp_loss_record = 0.
        self.vq_loss_record = 0.
        self.keep_ratio_record = 0.
        
        self.dynamic = dynamic
        if self.dynamic:
            print('using dynamic loss')

    def forward(self, outputs, labels):
        pred, mask_m, rho, vq_loss = outputs
        surp_loss = 0.0
        score = mask_m
        keep_ratio = score.mean(1)
   
        surp_loss = surp_loss + ((keep_ratio - rho) ** 2).mean()    ### The supervised loss. 
        main_loss = self.base_criterion(pred, labels)              ### Reconstruction loss.

        loss = self.main_weight * main_loss + \
               self.surp_weight * surp_loss + \
               self.rho_weight * rho + self.vq_weight * vq_loss
        # loss = self.clf_weight * cls_loss + vq_loss
        if self.print_mode:
            self.main_loss_record += main_loss.item()
            self.surp_loss_record += surp_loss.item()
            self.vq_loss_record += vq_loss.item()
            self.keep_ratio_record += keep_ratio.mean().item()
            self.count += 1
            if self.count == 100:
                print('loss info: main_loss=%.4f, surp_loss=%.4f, vq_loss=%.4f, keep ratio=%.4f' 
                        % (self.main_loss_record / self.count, 
                           self.surp_loss_record / self.count, 
                           self.vq_loss_record / self.count,
                           self.keep_ratio_record / self.count))
                self.main_loss_record = 0
                self.surp_loss_record = 0
                self.vq_loss_record = 0
                self.keep_ratio_record = 0
                self.count = 0
        return loss