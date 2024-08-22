import os
import sys
import argparse
import cv2 as cv
import numpy as np
import time

import torch
from main_test_swinir import define_model, get_image_pair, setup, test

# test SwinIR by partioning the image into patches
test_patch_wise = False
fileroot = r"C:/Users/takao/Desktop/denoising_raw_data"
filenames = [f for f in os.listdir(fileroot) if f.endswith(".png")]

processed_path = r"C:/Users/takao/Desktop/denoising_processed_data/swir"

if os.path.exists(processed_path):
    print(f'save path {processed_path} exists.')

else:
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='gray_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()
    window_size = setup(args)[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        print("path does not exist. check path argument.")
        sys.exit(0)

    model = define_model(args)
    model.eval()
    model = model.to(device)
    
    for filename in filenames:
        dst_path = os.path.join(fileroot, filename)
        start = time.time()
        # print(imgname)
        imgname, _, img_gt = get_image_pair(args, dst_path)

        img_gt = np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_gt = torch.from_numpy(img_gt).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_gt.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_gt = torch.cat([img_gt, torch.flip(img_gt, [2])], 2)[:, :, :h_old + h_pad, :]
            img_gt = torch.cat([img_gt, torch.flip(img_gt, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_gt, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        print(f"total process time: {time.time()-start} seconds")

        cv.imwrite(os.path.join(processed_path, "processed_"+imgname+".png"), output)
        
        # cv.namedWindow(imgname)
        # cv.imshow(imgname, output)
        # key = cv.waitKey(0)
        # if key == ord("q"):
        #     pass


if __name__ == '__main__':
    main()
    sys.exit(0)
