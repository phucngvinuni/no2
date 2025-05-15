# datasets.py
import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import glob
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing import Tuple, Any, List, Dict

# Assuming CIFAR_M and ImageFolder might be used for other experiments, you can keep their imports if needed.
# from CIFAR_M import CIFAR_M
# from dataset_folder import ImageFolder # This was for classification-style ImageFolder

class RandomMaskingGenerator:
    """
    Generates a random mask for MAE-style processing.
    Output: A boolean numpy array where True means "masked".
    """
    def __init__(self, input_size_patches: Tuple[int, int], mask_ratio: float):
        # input_size_patches: (num_patches_height, num_patches_width) for the ViT encoder
        self.height_patches, self.width_patches = input_size_patches
        self.num_patches = self.height_patches * self.width_patches
        self.num_mask = int(mask_ratio * self.num_patches)
        if self.num_mask < 0: self.num_mask = 0
        if self.num_mask > self.num_patches: self.num_mask = self.num_patches


    def __repr__(self):
        return f"RandomMasker(total_patches={self.num_patches}, num_mask={self.num_mask})"

    def __call__(self) -> np.ndarray: # Returns boolean mask
        if self.num_mask == 0: # No masking, all visible
            return np.zeros(self.num_patches, dtype=bool)
        if self.num_mask == self.num_patches: # All masked (unlikely useful for recon training)
            return np.ones(self.num_patches, dtype=bool)
            
        # True means MASKED, False means VISIBLE
        mask = np.hstack([
            np.ones(self.num_mask, dtype=bool),         # Masked tokens
            np.zeros(self.num_patches - self.num_mask, dtype=bool) # Visible tokens
        ])
        np.random.shuffle(mask)
        return mask


class SemComInputProcessor:
    """
    Applies image transformations (resize, ToTensor) and generates the
    SemCom patch mask for the ViT encoder.
    """
    def __init__(self,
                 image_pixel_size: int, # Target H, W for the image tensor
                 semcom_patch_grid_size: Tuple[int, int], # (num_patches_h, num_patches_w)
                 mask_ratio: float,
                 is_train: bool): # is_train is not used here but kept for interface consistency
        self.image_pixel_size = image_pixel_size

        # Image transform to get a CHW tensor in [0,1] range
        # This is the target for SemCom reconstruction and input to SemCom.
        self.image_transform = transforms.Compose([
            transforms.Resize((image_pixel_size, image_pixel_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # Scales PIL image [0,255] to PyTorch tensor [0,1]
            # No further normalization if SemCom reconstructs to [0,1] (due to final sigmoid)
        ])

        self.mask_generator = RandomMaskingGenerator(semcom_patch_grid_size, mask_ratio)

    def __call__(self, image_pil: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # image_pil: Input PIL Image
        transformed_image_tensor = self.image_transform(image_pil) # CHW, [0,1]
        
        # Generate boolean mask for SemCom encoder's patches (True means masked)
        semcom_patch_mask_np = self.mask_generator()
        semcom_patch_mask_tensor = torch.from_numpy(semcom_patch_mask_np) # Boolean Tensor
        
        return transformed_image_tensor, semcom_patch_mask_tensor


class YOLODataset(data.Dataset):
    """
    Dataset loader for YOLO formatted annotations.
    Returns:
        - semcom_input_tuple: (image_tensor_for_semcom_input [CHW, 0-1], semcom_encoder_patch_mask [boolean])
        - targets_tuple: (image_tensor_for_reconstruction_target [CHW, 0-1], yolo_gt_for_metrics_dict)
            where yolo_gt_for_metrics_dict = {'boxes': abs_pixel_xyxy_tensor, 'labels': class_id_tensor}
    """
    def __init__(self,
                 img_root_dir_for_split: str, # e.g., path/to/dataset/train or path/to/dataset/valid
                 img_pixel_size: int, # Target H,W for image tensor (e.g., 640 for YOLOv11)
                 semcom_vit_patch_grid: Tuple[int, int], # (num_h_patches, num_w_patches) for SemCom ViT encoder
                 semcom_encoder_mask_ratio: float,
                 is_train_split: bool,
                 # Add albumentations_transform=None here if you want to use it for robust obj detection aug
                 ):
        self.img_dir = os.path.join(img_root_dir_for_split, 'images')
        self.label_dir = os.path.join(img_root_dir_for_split, 'labels')
        self.img_pixel_size = img_pixel_size
        self.is_train_split = is_train_split

        # Find all image files
        self.img_files = sorted(
            glob.glob(os.path.join(self.img_dir, '*.jpg')) +
            glob.glob(os.path.join(self.img_dir, '*.png')) +
            glob.glob(os.path.join(self.img_dir, '*.jpeg'))
        )
        if not self.img_files:
            raise FileNotFoundError(f"No images found in {self.img_dir}. Check path and extensions.")

        # Create corresponding label file paths
        self.label_files = [
            os.path.join(self.label_dir, os.path.splitext(os.path.basename(f))[0] + '.txt')
            for f in self.img_files
        ]
        
        # Filter out images that don't have corresponding label files (especially for train/val)
        # For test set, sometimes labels are not available, allow it.
        initial_img_count = len(self.img_files)
        if self.is_train_split or os.path.exists(self.label_files[0] if self.label_files else ""): # Check if labels are expected
            valid_indices = [i for i, lf in enumerate(self.label_files) if os.path.exists(lf)]
            self.img_files = [self.img_files[i] for i in valid_indices]
            self.label_files = [self.label_files[i] for i in valid_indices]
            if len(self.img_files) < initial_img_count:
                print(f"Warning: {initial_img_count - len(self.img_files)} images were removed "
                      f"from {self.img_dir} due to missing label files in {self.label_dir}.")
        if not self.img_files:
            raise FileNotFoundError(f"No image/label pairs found after filtering for {img_root_dir_for_split}.")


        self.semcom_processor = SemComInputProcessor(
            image_pixel_size=img_pixel_size,
            semcom_patch_grid_size=semcom_vit_patch_grid,
            mask_ratio=semcom_encoder_mask_ratio,
            is_train=is_train_split # Pass training status
        )
        # TODO: If is_train_split, you might want to add more image augmentations here
        # using torchvision.transforms or preferably Albumentations (which also handles bounding boxes).
        # self.augmentations = ... if is_train_split else None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning next sample.")
            # This is a simple way to skip; more robust is to filter out bad images beforehand
            # or have collate_fn handle None.
            return self.__getitem__((index + 1) % len(self))

        # TODO: Apply augmentations HERE if using Albumentations (img_pil, boxes_normalized) -> (aug_img_pil, aug_boxes_normalized)
        # if self.is_train_split and self.augmentations:
        #    ... apply augmentations ...

        # Process image for SemCom (resize to pixel_size, convert to tensor [0,1], generate patch mask)
        img_tensor_for_semcom, semcom_encoder_patch_mask = self.semcom_processor(img_pil)

        # Load YOLO labels (class_id, cx_norm, cy_norm, w_norm, h_norm)
        boxes_normalized_xyxy = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            
                            x_min = np.clip(cx - w / 2, 0.0, 1.0)
                            y_min = np.clip(cy - h / 2, 0.0, 1.0)
                            x_max = np.clip(cx + w / 2, 0.0, 1.0)
                            y_max = np.clip(cy + h / 2, 0.0, 1.0)

                            if x_max > x_min and y_max > y_min: # Check for valid box
                                boxes_normalized_xyxy.append([x_min, y_min, x_max, y_max])
                                class_labels.append(class_id)
                        except ValueError:
                            print(f"Warning: Malformed line in label file {label_path}: '{line.strip()}'")
        
        boxes_normalized_tensor = torch.as_tensor(boxes_normalized_xyxy, dtype=torch.float32)
        labels_tensor = torch.as_tensor(class_labels, dtype=torch.int64)

        # Convert normalized XYXY boxes to absolute pixel XYXY for torchmetrics
        abs_pixel_boxes_tensor = boxes_normalized_tensor.clone()
        if abs_pixel_boxes_tensor.numel() > 0: # Only scale if boxes exist
            abs_pixel_boxes_tensor[:, [0, 2]] *= self.img_pixel_size # Scale X coordinates
            abs_pixel_boxes_tensor[:, [1, 3]] *= self.img_pixel_size # Scale Y coordinates
        
        yolo_gt_for_metrics_dict = {
            "boxes": abs_pixel_boxes_tensor,
            "labels": labels_tensor
        }

        # Tuple for SemCom model input
        semcom_input_tuple = (img_tensor_for_semcom, semcom_encoder_patch_mask)
        # Tuple for targets (SemCom reconstruction target, YOLO GT for metrics)
        targets_tuple = (img_tensor_for_semcom.clone(), yolo_gt_for_metrics_dict)

        return semcom_input_tuple, targets_tuple


def build_dataset(is_train: bool, args: Any) -> data.Dataset:
    """Builds the appropriate dataset based on args."""
    if args.data_set == 'fish':
        # Determine if it's train, validation, or test split
        if is_train: # Training phase
            data_split_subdir = 'train'
        elif not args.eval: # Validation phase (during training loop)
            data_split_subdir = 'valid'
        else: # Evaluation phase (args.eval is True)
            data_split_subdir = 'test'
        
        # args.data_path should be the root directory containing train/, valid/, test/
        current_split_root_dir = os.path.join(args.data_path, data_split_subdir)
        if not os.path.isdir(current_split_root_dir):
            raise FileNotFoundError(
                f"Dataset directory for split '{data_split_subdir}' not found at: {current_split_root_dir}. "
                f"Ensure args.data_path ('{args.data_path}') is the parent of '{data_split_subdir}' directories."
            )
        
        print(f"Building 'fish' dataset for split '{data_split_subdir}' from: {current_split_root_dir}")
        dataset = YOLODataset(
            img_root_dir_for_split=current_split_root_dir,
            img_pixel_size=args.input_size, # Pixel H,W of image (e.g., 640)
            semcom_vit_patch_grid=args.window_size, # (num_h_patches, num_w_patches) for SemCom ViT
            semcom_encoder_mask_ratio=args.mask_ratio,
            is_train_split=is_train # For enabling/disabling training-specific augmentations (if any)
        )
    # elif args.data_set.startswith('cifar') or args.data_set.startswith('imagenet'):
        # You would need to adapt these for the new (SemCom_input_tuple, Targets_tuple) structure
        # or create separate dataset classes for them if they are for classification.
        # For Approach 1, we are focusing on 'fish' (YOLO) dataset.
        # print(f"Warning: Dataset '{args.data_set}' is not fully adapted for reconstruction + YOLO eval yet.")
        # transform_for_other = SemComInputProcessor(args.input_size, args.window_size, args.mask_ratio, is_train)
        # if args.data_set.startswith('cifar_S32'):
        #     from CIFAR_M import CIFAR_M # Assuming this exists
        #     dataset = CIFAR_M(args.data_path, train=is_train, transform=transform_for_other, download=True)
        # else: # imagenet / cifar_S224
        #     from dataset_folder import ImageFolder # Assuming this exists
        #     root = os.path.join(args.data_path, 'Train' if is_train else 'Test') # Adjust path
        #     dataset = ImageFolder(root, transform=transform_for_other) # This ImageFolder is for classification
    else:
        raise NotImplementedError(f"Dataset '{args.data_set}' is not implemented for reconstruction + YOLO evaluation.")
    return dataset

def yolo_collate_fn(batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, Dict[str, torch.Tensor]]]]
                   ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                              Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
    """
    Custom collate_fn for YOLODataset.
    - Stacks SemCom image tensors and mask tensors.
    - Stacks SemCom reconstruction target image tensors.
    - Keeps YOLO ground truth target dictionaries as a list.
    """
    semcom_input_images = []
    semcom_encoder_masks = []
    semcom_reconstruction_targets = []
    yolo_gt_target_list_of_dicts = []

    for item in batch:
        semcom_input_tuple, targets_tuple = item
        
        semcom_input_images.append(semcom_input_tuple[0])
        semcom_encoder_masks.append(semcom_input_tuple[1])
        
        semcom_reconstruction_targets.append(targets_tuple[0])
        yolo_gt_target_list_of_dicts.append(targets_tuple[1])

    collated_semcom_input_images = torch.stack(semcom_input_images, 0)
    collated_semcom_encoder_masks = torch.stack(semcom_encoder_masks, 0)
    collated_semcom_reconstruction_targets = torch.stack(semcom_reconstruction_targets, 0)

    collated_semcom_input_tuple = (collated_semcom_input_images, collated_semcom_encoder_masks)
    # yolo_gt_target_list_of_dicts is already a list of dicts, which is what torchmetrics expects for targets.
    collated_targets_tuple = (collated_semcom_reconstruction_targets, yolo_gt_target_list_of_dicts)

    return collated_semcom_input_tuple, collated_targets_tuple