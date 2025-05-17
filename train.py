from random import seed
import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from models.Recon_subnetwork import UNetModel # Removed update_ema_params if not used
from models.Seg_subnetwork import SegmentationSubNetwork
from tqdm import tqdm
import torch.nn as nn
from data.dataset_beta_thresh import (
    MVTecTrainDataset,MVTecTestDataset,
    CustomTestDataset,CustomTrainDataset
)
from math import exp # Unused if BinaryFocalLoss or Gaussian function not here
import torch.nn.functional as F
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
# from scipy.ndimage import gaussian_filter # Moved to eval if only used there
# from skimage.measure import label, regionprops # Moved to eval
from sklearn.metrics import roc_auc_score # Removed auc, average_precision_score if only for full eval
import pandas as pd
from collections import defaultdict

def weights_init(m): # Keep if used for initializing models not covered by their own init
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)    

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True): 
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce # reduce means 'reduction'='mean' if True

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def train(training_dataset_loader, testing_dataset_loader, args, data_len, sub_class, class_type, device):
   
    in_channels = args["channels"]
    # Initialize UNetModel
    unet_model = UNetModel(
        img_size=args['img_size'], 
        base_channels=args['base_channels'], 
        channel_mults=args.get('channel_mults', ""), # Use .get for optional args with default
        dropout=args["dropout"], 
        n_heads=args["num_heads"], 
        n_head_channels=args["num_head_channels"],
        attention_resolutions=args["attention_resolutions"],
        in_channels=in_channels,
        # biggan_updown is True by default in UNetModel if not specified
    ).to(device)


    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, 
            loss_weight=args.get('loss_weight', 'none'), # .get for optional args
            loss_type=args['loss-type'], 
            noise=args["noise_fn"], 
            img_channels=in_channels
            )

    seg_model=SegmentationSubNetwork(
        in_channels=in_channels * 2, # Assuming concatenated (original_img, reconstructed_img)
        out_channels=1
    ).to(device)


    optimizer_ddpm = optim.Adam(unet_model.parameters(), lr=args['diffusion_lr'], weight_decay=args['weight_decay'])
    optimizer_seg = optim.Adam(seg_model.parameters(), lr=args['seg_lr'], weight_decay=args['weight_decay'])
    
    loss_focal = BinaryFocalLoss().to(device)
    loss_smL1= nn.SmoothL1Loss().to(device)
    
    tqdm_epoch = range(0, args['EPOCHS'])
    # Cosine Annealing for segmentation model's optimizer
    scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=args.get('scheduler_T_max', args['EPOCHS']), eta_min=args.get('scheduler_eta_min', 0))
    # Optional: Scheduler for DDPM optimizer
    # scheduler_ddpm = optim.lr_scheduler.CosineAnnealingLR(optimizer_ddpm, T_max=args.get('scheduler_T_max', args['EPOCHS']), eta_min=args.get('scheduler_eta_min', 0))

    train_loss_list=[]
    # train_noise_loss_list=[] # If you want to track components separately
    # train_focal_loss_list=[]
    # train_smL1_loss_list=[]
    # loss_x_list=[] # For plotting epochs
    
    best_combined_auroc = 0.0 # Tracks sum of image and pixel AUROC
    best_image_auroc = 0.0
    best_pixel_auroc = 0.0
    best_epoch = 0
    
    # image_auroc_list=[] # For plotting eval performance over epochs
    # pixel_auroc_list=[]
    # performance_x_list=[] # Epoch numbers for plotting eval performance

    for epoch in tqdm_epoch:
        unet_model.train()
        seg_model.train()
        
        epoch_train_loss = 0.0
        epoch_noise_loss = 0.0 # For tracking noise loss component
        epoch_focal_loss = 0.0 # For tracking focal loss component
        epoch_sml1_loss = 0.0  # For tracking smooth L1 loss component

        tbar = tqdm(training_dataset_loader, desc=f"Epoch {epoch+1}/{args['EPOCHS']}", leave=False)
        for i, sample in enumerate(tbar):
            
            aug_image = sample['augmented_image'].to(device) # This is x_0 for DDPM
            anomaly_mask = sample["anomaly_mask"].to(device) # Ground truth for segmentation
            anomaly_label = sample["has_anomaly"].to(device).squeeze() # 0 for normal, 1 for anomaly

            # --- Current Features ---
            # This is where you need to get your (N, 3) current_features tensor.
            # It should correspond to the `aug_image` batch.
            # If it's part of your dataset, it would be like:
            # current_features = sample['current_features'].to(device)
            # As a placeholder, we'll create a dummy tensor:
            if 'current_features' in sample and sample['current_features'] is not None:
                 current_features = sample['current_features'].to(device)
            else:
                 # This is a DUMMY PLACEHOLDER. Replace with actual data loading.
                 current_features = torch.randn(aug_image.shape[0], 3, device=device) 
            # --- End Current Features ---

            # DDPM part: Predicts reconstructed x0 (pred_x0) and calculates noise loss
            noise_loss_val, pred_x0, normal_t, x_normal_t, x_noiser_t = ddpm_sample.norm_guided_one_step_denoising(
                unet_model, aug_image, anomaly_label, args, current_features # Pass current_features
            )
            
            # Segmentation part: Predicts mask from original image and DDPM's reconstruction
            # Input to seg_model is concatenation of original image and reconstructed x0
            seg_input = torch.cat((aug_image, pred_x0), dim=1)
            pred_mask = seg_model(seg_input) 

            # Calculate segmentation losses
            focal_loss_val = loss_focal(pred_mask, anomaly_mask)
            smL1_loss_val = loss_smL1(pred_mask, anomaly_mask)
            
            # Total loss: weighted sum of DDPM noise loss and segmentation losses
            # Weights (e.g., 5 for focal_loss) can be tuned or moved to args
            total_loss = noise_loss_val + args.get("focal_loss_weight", 5) * focal_loss_val + args.get("sml1_loss_weight", 1) * smL1_loss_val
            
            optimizer_ddpm.zero_grad()
            optimizer_seg.zero_grad()
            total_loss.backward()
            optimizer_ddpm.step()
            optimizer_seg.step()
            
            epoch_train_loss += total_loss.item()
            epoch_noise_loss += noise_loss_val.item()
            epoch_focal_loss += (args.get("focal_loss_weight", 5) * focal_loss_val.item())
            epoch_sml1_loss += (args.get("sml1_loss_weight", 1) * smL1_loss_val.item())

            tbar.set_postfix(loss=total_loss.item(), noise_loss=noise_loss_val.item(), focal=focal_loss_val.item(), sml1=smL1_loss_val.item())

        # End of epoch
        avg_epoch_train_loss = epoch_train_loss / len(training_dataset_loader)
        avg_epoch_noise_loss = epoch_noise_loss / len(training_dataset_loader)
        avg_epoch_focal_loss = epoch_focal_loss / len(training_dataset_loader)
        avg_epoch_sml1_loss = epoch_sml1_loss / len(training_dataset_loader)
        
        print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_epoch_train_loss:.4f}, Avg Noise Loss: {avg_epoch_noise_loss:.4f}, Avg Focal: {avg_epoch_focal_loss:.4f}, Avg SmL1: {avg_epoch_sml1_loss:.4f}")

        scheduler_seg.step()
        # if scheduler_ddpm: scheduler_ddpm.step()

        if (epoch + 1) % args.get("eval_every_epochs", 10) == 0: # Evaluate every N epochs
            temp_image_auroc, temp_pixel_auroc = eval(testing_dataset_loader, args, unet_model, seg_model, data_len, sub_class, device) # Pass data_len if used by eval
            # image_auroc_list.append(temp_image_auroc) # For plotting
            # pixel_auroc_list.append(temp_pixel_auroc)
            # performance_x_list.append(epoch + 1)

            current_combined_auroc = temp_image_auroc + temp_pixel_auroc
            if current_combined_auroc >= best_combined_auroc: # Prioritize combined, then image AUROC
                if temp_image_auroc >= best_image_auroc : # Secondary check on image_auroc
                    best_combined_auroc = current_combined_auroc
                    best_image_auroc = temp_image_auroc
                    best_pixel_auroc = temp_pixel_auroc
                    best_epoch = epoch + 1
                    save(unet_model, seg_model, args=args, final='best', epoch=epoch + 1, sub_class=sub_class)
                    print(f"*** New best model saved at epoch {best_epoch} with Image AUROC: {best_image_auroc:.2f}, Pixel AUROC: {best_pixel_auroc:.2f} ***")

        # Log training losses periodically
        if (epoch + 1) % args.get("log_loss_every_epochs", 5) == 0:
            train_loss_list.append(round(avg_epoch_train_loss, 4)) # Log average epoch loss
            # You can log components too if needed:
            # train_noise_loss_list.append(round(avg_epoch_noise_loss,4))
            # train_focal_loss_list.append(round(avg_epoch_focal_loss,4))
            # train_smL1_loss_list.append(round(avg_epoch_sml1_loss,4))
            # loss_x_list.append(epoch + 1)

    # Save the last model
    save(unet_model, seg_model, args=args, final='last', epoch=args['EPOCHS'], sub_class=sub_class)
    print(f"Last model saved at epoch {args['EPOCHS']}")

    # Record best performance to CSV
    # Construct path using args["output_path"] and args["arg_num"]
    metrics_dir = os.path.join(args["output_path"], "metrics", f"ARGS={args['arg_num']}")
    os.makedirs(metrics_dir, exist_ok=True)
    csv_filename = os.path.join(metrics_dir, f"{args['eval_normal_t']}_{args['eval_noisier_t']}t_{args['condition_w']}_{class_type}_image_pixel_auroc_train.csv")
    
    performance_summary = {"classname": [sub_class], "Image-AUROC": [best_image_auroc], "Pixel-AUROC": [best_pixel_auroc], "epoch": [best_epoch]}
    df_class_summary = pd.DataFrame(performance_summary)
    
    file_exists = os.path.isfile(csv_filename)
    df_class_summary.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
    print(f"Training summary saved to {csv_filename}")


def eval(testing_dataset_loader, args, unet_model, seg_model, data_len, sub_class, device): # data_len might be unused here
    unet_model.eval()
    seg_model.eval()
    
    # Create output directories if they don't exist (though eval usually doesn't save visualizations like main eval.py)
    # viz_dir = os.path.join(args["output_path"], "metrics", f"ARGS={args['arg_num']}", sub_class, "eval_viz_during_train")
    # os.makedirs(viz_dir, exist_ok=True)

    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample = GaussianDiffusionModel(
            args['img_size'], betas, 
            loss_weight=args.get('loss_weight', 'none'),
            loss_type=args['loss-type'], 
            noise=args["noise_fn"], 
            img_channels=in_channels
            )
    
    total_image_pred = [] # Use lists for easier appending
    total_image_gt = []
    total_pixel_gt = []
    total_pixel_pred = []
    
    with torch.no_grad(): # Ensure no gradients are computed during evaluation
        tbar_eval = tqdm(testing_dataset_loader, desc=f"Evaluating {sub_class}", leave=False)
        for i, sample in enumerate(tbar_eval):
            image = sample["image"].to(device) # Ground truth image x_0
            target_has_anomaly = sample['has_anomaly'].to(device) # Image-level label
            gt_mask = sample["mask"].to(device) # Pixel-level ground truth mask

            # --- Current Features (for eval) ---
            if 'current_features' in sample and sample['current_features'] is not None:
                 current_features = sample['current_features'].to(device)
            else:
                 current_features = torch.randn(image.shape[0], 3, device=device) # DUMMY PLACEHOLDER
            # --- End Current Features ---

            # Timesteps for evaluation from args
            normal_t_tensor = torch.tensor([args["eval_normal_t"]], device=image.device).repeat(image.shape[0])
            noisier_t_tensor = torch.tensor([args["eval_noisier_t"]], device=image.device).repeat(image.shape[0])
            
            # Get model's reconstruction (pred_x_0_condition)
            # The 'loss' returned here is eval loss, not used for backprop
            _, pred_x_0_condition, _, _, _, _, _ = ddpm_sample.norm_guided_one_step_denoising_eval(
                unet_model, image, normal_t_tensor, noisier_t_tensor, args, current_features # Pass current_features
            )
            
            # Get segmentation mask
            seg_input_eval = torch.cat((image, pred_x_0_condition), dim=1)
            pred_mask_seg = seg_model(seg_input_eval)
            out_mask = pred_mask_seg # This is the anomaly map

            # Image-level score calculation (example: mean of top-k pixel scores in predicted mask)
            # Flatten the predicted mask (B, 1, H, W) -> (B, H*W)
            flat_out_mask = out_mask.view(out_mask.shape[0], -1) 
            # Take mean of all pixel scores as image score (can be refined, e.g., top-k)
            # image_score = torch.mean(flat_out_mask, dim=1) # Original took top-50, which might be too specific
            
            # Alternative from original eval: mean of top 50 pixels
            if flat_out_mask.shape[1] >= 50:
                 topk_values, _ = torch.topk(flat_out_mask, 50, dim=1, largest=True)
                 image_score = torch.mean(topk_values, dim=1)
            else: # Handle cases with fewer than 50 pixels (e.g. very small images)
                 image_score = torch.mean(flat_out_mask, dim=1)


            total_image_pred.extend(image_score.cpu().numpy())
            total_image_gt.extend(target_has_anomaly.cpu().numpy())

            # Pixel-level scores
            total_pixel_pred.extend(out_mask.flatten().cpu().numpy())
            total_pixel_gt.extend(gt_mask.flatten().cpu().numpy().astype(int))
            
    # Convert lists to numpy arrays for metric calculation
    total_image_pred_np = np.array(total_image_pred)
    total_image_gt_np = np.array(total_image_gt)
    total_pixel_pred_np = np.array(total_pixel_pred)
    total_pixel_gt_np = np.array(total_pixel_gt)

    print(f"\nEvaluation for {sub_class}:")
    # Image AUROC
    if len(np.unique(total_image_gt_np)) > 1: # Check if there's more than one class for AUROC
        auroc_image = round(roc_auc_score(total_image_gt_np, total_image_pred_np) * 100, 2)
        print(f"  Image AUROC: {auroc_image:.2f}%")
    else:
        auroc_image = 0.0 # Or handle as undefined / print warning
        print(f"  Image AUROC: Not defined (only one class in ground truth)")
    
    # Pixel AUROC
    if len(np.unique(total_pixel_gt_np)) > 1:
        auroc_pixel = round(roc_auc_score(total_pixel_gt_np, total_pixel_pred_np) * 100, 2)
        print(f"  Pixel AUROC: {auroc_pixel:.2f}%")
    else:
        auroc_pixel = 0.0
        print(f"  Pixel AUROC: Not defined (only one class in pixel ground truth)")
   
    return auroc_image, auroc_pixel


def save(unet_model, seg_model, args, final, epoch, sub_class):
    # Construct save path based on args
    model_dir = os.path.join(args["output_path"], "model", f"diff-params-ARGS={args['arg_num']}", sub_class)
    os.makedirs(model_dir, exist_ok=True)
    
    save_path = os.path.join(model_dir, f"params-{final}.pt")
    
    torch.save(
        {
            'n_epoch': epoch, # Save epoch number (1-based)
            'unet_model_state_dict': unet_model.state_dict(),
            'seg_model_state_dict': seg_model.state_dict(),
            # "args": args # Saving full args dict can make file large, consider saving only key ones
            "args_summary": { # Save a summary or reference to args file
                "arg_num": args["arg_num"],
                "img_size": args["img_size"],
                "base_channels": args["base_channels"],
                "class_type": args.get("class_type", "Unknown") # Store class type if available
            }
        }, 
        save_path
    )
    print(f"Saved model checkpoint to {save_path} (Epoch: {epoch})")
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Argument parsing (assuming args1.json is the structure)
    args_file_name = "args1.json" # Example, can be made a script argument
    args_file_path = os.path.join('./args', args_file_name) # Ensure this path is correct
    if not os.path.exists(args_file_path):
        # Try a common alternative if script is run from project root
        args_file_path = os.path.join(os.path.dirname(__file__), 'args', args_file_name)
        if not os.path.exists(args_file_path):
             # If still not found, try relative to current working directory if it's the project root
            args_file_path_alt = os.path.join(os.getcwd(), 'args', args_file_name)
            if os.path.exists(args_file_path_alt):
                args_file_path = args_file_path_alt
            else:
                print(f"Error: Args file not found at {args_file_path} or {args_file_path_alt}")
                return

    with open(args_file_path, 'r') as f:
        args_loaded = json.load(f)
    
    args = defaultdict_from_json(args_loaded) # Use defaultdict for easier access
    args['arg_num'] = args_file_name.split('.')[0].replace('args', '') # Get number from filename, e.g. '1' from 'args1.json'
    
    # Seed for reproducibility
    manual_seed = args.get("seed", 42) # Get seed from args or default to 42
    seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
    print(f"Set seed to {manual_seed}")


    # Define dataset classes (adjust paths in args1.json as needed)
    mvtec_classes = args.get('mvtec_classes', ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'])
    visa_classes = args.get('visa_classes', ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'])
    mpdd_classes = args.get('mpdd_classes', ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'])
    dagm_classes = args.get('dagm_classes', [f'Class{i}' for i in range(1, 11)])
    custom_dataset_classes = args.get('custom_dataset_classes', ['chamber']) # Example from args

    # Select which dataset to run on (e.g., from args or hardcode for testing)
    # current_run_dataset_type = args.get("dataset_type_to_run", "Custom") # e.g. "MVTec", "VisA", "Custom"
    # if current_run_dataset_type == "MVTec":
    #     current_classes_to_run = mvtec_classes
    # elif current_run_dataset_type == "VisA":
    #     current_classes_to_run = visa_classes
    # elif current_run_dataset_type == "Custom":
    current_classes_to_run = custom_dataset_classes # Default to custom as per original
    # else:
    #     print(f"Unknown dataset_type_to_run: {current_run_dataset_type}")
    #     return

    for sub_class_name in current_classes_to_run:    
        print(f"\n--- Training for Class: {sub_class_name} ---")
        args["current_class"] = sub_class_name # Store current class in args for potential use

        # Determine dataset paths and types
        class_type_str = "Unknown" # For logging/saving paths
        
        if sub_class_name in visa_classes and "visa_root_path" in args:
            dataset_root_path = os.path.join(args["visa_root_path"], sub_class_name)
            TrainDS = VisATrainDataset
            TestDS = VisATestDataset
            class_type_str = 'VisA'
        elif sub_class_name in mpdd_classes and "mpdd_root_path" in args:
            dataset_root_path = os.path.join(args["mpdd_root_path"], sub_class_name)
            TrainDS = MPDDTrainDataset
            TestDS = MPDDTestDataset
            class_type_str = 'MPDD'
        elif sub_class_name in mvtec_classes and "mvtec_root_path" in args:
            dataset_root_path = os.path.join(args["mvtec_root_path"], sub_class_name)
            TrainDS = MVTecTrainDataset
            TestDS = MVTecTestDataset
            class_type_str = 'MVTec'
        elif sub_class_name in dagm_classes and "dagm_root_path" in args:
            dataset_root_path = os.path.join(args["dagm_root_path"], sub_class_name)
            TrainDS = DAGMTrainDataset
            TestDS = DAGMTestDataset
            class_type_str = 'DAGM'
        elif sub_class_name in custom_dataset_classes and "custom_dataset_root_path" in args:
            dataset_root_path = os.path.join(args["custom_dataset_root_path"], sub_class_name)
            TrainDS = CustomTrainDataset
            TestDS = CustomTestDataset
            class_type_str = 'Custom'
        else:
            print(f"Warning: Dataset path or type not defined for class {sub_class_name}. Skipping.")
            continue
        
        args["class_type"] = class_type_str # Store for save function

        print(f"Using dataset path: {dataset_root_path} for class type: {class_type_str}")
        print(f"Training with args: {args['arg_num']}, Image Size: {args['img_size']}")     

        # Initialize datasets and dataloaders
        # Note: `args` is passed to dataset constructors if they use it (e.g., for anomaly_source_path)
        training_dataset = TrainDS(dataset_root_path, sub_class_name, img_size=args["img_size"], args=args)
        testing_dataset = TestDS(dataset_root_path, sub_class_name, img_size=args["img_size"]) # Args might not be needed for test DS
        
        if len(training_dataset) == 0:
            print(f"Skipping {sub_class_name} due to empty training dataset.")
            continue

        training_loader = DataLoader(training_dataset, batch_size=args['Batch_Size'], shuffle=True, 
                                     num_workers=args.get("num_workers_train", 4), pin_memory=True, drop_last=True)
        testing_loader = DataLoader(testing_dataset, batch_size=args.get("test_batch_size", 1), shuffle=False, 
                                    num_workers=args.get("num_workers_test", 4))

        # Create directories for outputs (models, images, metrics)
        # These are structured based on args["output_path"] and args["arg_num"]
        output_dirs_to_create = [
            os.path.join(args["output_path"], "model", f"diff-params-ARGS={args['arg_num']}", sub_class_name),
            # os.path.join(args["output_path"], "diffusion-training-images", f"ARGS={args['arg_num']}", sub_class_name), # If saving training images
            os.path.join(args["output_path"], "metrics", f"ARGS={args['arg_num']}", sub_class_name)
        ]
        for dir_path in output_dirs_to_create:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {dir_path}: {e}")
                # Decide if to stop or continue if a directory can't be made
        
        # data_len for testing_dataset, used in original eval, might not be strictly needed if eval iterates loader
        test_data_len = len(testing_dataset) 
        
        train(training_loader, testing_loader, args, test_data_len, sub_class_name, class_type_str, device)

    print("\n--- All training sessions complete. ---")

if __name__ == '__main__':
    main()