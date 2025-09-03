from __future__ import absolute_import, division, print_function
import os
import numpy as np
import torch
import cv2

from options import MonodepthOptions
from networks import ResnetEncoder, DepthDecoder  # Monodepth2 network components
# Note: Ensure that the Monodepth2 repository (with 'networks' and 'datasets' modules) is in PYTHONPATH

def readlines(filename):
    """Read all lines from a text file and return as a list"""
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def compute_depth_errors(gt, pred):
    """Compute error metrics between predicted and ground truth depths:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}"""
    # Mask out zero depth values (invalid ground truth)
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]
    # Calculate metrics
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel  = np.mean(((gt - pred) ** 2) / gt)
    rmse    = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    # Accuracy metrics
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = np.mean((thresh < 1.25    ).astype(np.float32))
    a2 = np.mean((thresh < 1.25**2).astype(np.float32))
    a3 = np.mean((thresh < 1.25**3).astype(np.float32))
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def disp_to_depth(disp, min_depth, max_depth):
    """
    Convert network sigmoid disparity output into depth map:contentReference[oaicite:2]{index=2}.
    Disparity is scaled to depth between min_depth and max_depth.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    # Scale disparity to [min_disp, max_disp]
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()  # Parse command-line arguments

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")

    # Load predicted disparities from file or compute using the model
    pred_disps = None
    if opt.ext_disp_to_eval:
        # Load existing disparities (numpy .npy file)
        pred_disps = np.load(opt.ext_disp_to_eval, allow_pickle=True)
        if pred_disps.ndim == 4:  # if disparities have shape (N,H,W,1)
            pred_disps = pred_disps.squeeze(-1)
        print(f"Loaded disparities from {opt.ext_disp_to_eval}, shape = {pred_disps.shape}")
    else:
        # If no disparity file is provided, run the model on the dataset to get disparities
        if opt.load_weights_folder is None:
            raise Exception("You must specify --ext_disp_to_eval or --load_weights_folder to evaluate depth.")
        # Load model weights
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), f"Cannot find folder {opt.load_weights_folder}"
        print("-> Loading model from ", opt.load_weights_folder)
        # Load encoder
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        depth_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path, map_location=device)
        # Extract settings from encoder dict
        feed_height = encoder_dict.get('height', opt.height)
        feed_width  = encoder_dict.get('width', opt.width)
        use_stereo = encoder_dict.get('use_stereo', False)
        # Initialize model
        encoder = ResnetEncoder(opt.num_layers, opt.weights_init == "pretrained")
        depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=opt.scales)
        # Load weights
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        depth_dict = torch.load(depth_path, map_location=device)
        depth_decoder.load_state_dict(depth_dict)
        encoder.to(device)
        depth_decoder.to(device)
        encoder.eval()
        depth_decoder.eval()
        # Load dataset for evaluation
        split_dir = os.path.join(os.path.dirname(__file__), "splits", opt.eval_split)
        file_list = "test_files.txt" if os.path.isfile(os.path.join(split_dir, "test_files.txt")) else "val_files.txt"
        eval_filenames = readlines(os.path.join(split_dir, file_list))
        print(f"-> Evaluating {len(eval_filenames)} images from split '{opt.eval_split}'")
        # Use the SCARED dataset class (with ground truth) for loading images
        from datasets import SCAREDRAWDataset
        eval_dataset = SCAREDRAWDataset(opt.data_path, eval_filenames, feed_height, feed_width, 
                                        [0], 4, is_train=False)
        pred_disps = []
        with torch.no_grad():
            for idx in range(len(eval_dataset)):
                # Load one sample (image) at a time to avoid memory issues
                sample = eval_dataset[idx]
                input_color = sample[("color", 0, 0)].unsqueeze(0).to(device)  # shape [1,3,H,W]
                output = depth_decoder(encoder(input_color))
                disp = output[("disp", 0)]  # disparity at scale 0
                # Resize disparity to original resolution if needed
                disp_resized = torch.nn.functional.interpolate(
                                   disp, (feed_height, feed_width), mode="bilinear", align_corners=False)
                pred_disp = disp_resized.squeeze().cpu().numpy()
                pred_disps.append(pred_disp)
        pred_disps = np.stack(pred_disps)
        print(f"-> Computed disparities for {pred_disps.shape[0]} images")

    # Load ground truth depths and evaluate
    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.eval_split)
    file_list = "test_files.txt" if os.path.isfile(os.path.join(split_folder, "test_files.txt")) else "val_files.txt"
    eval_filenames = readlines(os.path.join(split_folder, file_list))
    # Initialize dataset for ground truth depth loading (without image loading)
    from datasets import SCAREDRAWDataset
    gt_dataset = SCAREDRAWDataset(opt.data_path, eval_filenames, opt.height, opt.width, [0], 4, is_train=False)
    assert pred_disps.shape[0] == len(gt_dataset), "Mismatch between disparities and ground truth files"

    # Metrics accumulators
    num_samples = pred_disps.shape[0]
    accum_metrics = np.zeros(7, dtype=np.float64)  # [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]

    for i in range(num_samples):
        # Get predicted depth from disparity
        disp = pred_disps[i]
        disp_tensor = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        _, depth_pred_tensor = disp_to_depth(disp_tensor, opt.min_depth, opt.max_depth)
        depth_pred = depth_pred_tensor.squeeze().cpu().numpy()

        # Load corresponding ground truth depth
        folder, frame_idx, side = gt_dataset.index_to_folder_and_frame_idx(i)
        depth_gt = gt_dataset.get_depth(folder, frame_idx, side, do_flip=False)
        depth_gt = depth_gt.astype(np.float32)
        # Masking and scaling
        valid_mask = depth_gt > 0
        # Optionally apply median scaling for monocular models
        if not opt.disable_median_scaling and not opt.eval_stereo:
            med_pred = np.median(depth_pred[valid_mask])
            med_gt   = np.median(depth_gt[valid_mask])
            depth_pred *= (med_gt / (med_pred + 1e-6))
        # Clamp depth predictions to within [min_depth, max_depth] for metrics
        depth_pred = np.clip(depth_pred, opt.min_depth, opt.max_depth)
        # Compute metrics for this sample
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(depth_gt, depth_pred)
        accum_metrics += np.array([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

    # Average over all samples
    mean_metrics = accum_metrics / num_samples

    # Print evaluation results
    print("\n-> Evaluation results for '{}' split ({} samples):".format(opt.eval_split, num_samples))
    print(f"   abs_rel: {mean_metrics[0]:.4f} | sq_rel: {mean_metrics[1]:.4f} | "
          f"rmse: {mean_metrics[2]:.3f} | rmse_log: {mean_metrics[3]:.3f} | "
          f"a1: {mean_metrics[4]:.3f} | a2: {mean_metrics[5]:.3f} | a3: {mean_metrics[6]:.3f}")
