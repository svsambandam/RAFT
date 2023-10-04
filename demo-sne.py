import sys

sys.path.append('core')

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import imageio
import ipdb
import numpy as np
import torch
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from PIL import Image
from tqdm import tqdm

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)) 

    image = torch.from_numpy(img.astype("float32"))
    # print(img.shape,'--------------------------------------------')
    if image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image /= 255.0
        img = image[:, :, :3] * image[:, :, -1:] + torch.tensor([1.,1.,1.]) * (1.0 - image[:, :, -1:])
        img *= 255.0
    else:
        img = image[:, :, :3]
    # print(img.shape,'--------------------------------------------')
    return img.permute(2, 0, 1)[None].to(DEVICE)


def viz(img1, img2, flo):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img1, img2, flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.show()


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)
    
def index(args):
    ### LOAD 
    datapath = args.path
    scale_factor=1.0
    split= args.split
    num = args.k_nearest ## num of positions to choose from


    meta = load_from_json(Path(datapath + f"/transforms_{split}.json"))
    image_filenames = []
    poses = []
    times = []
    for frame in meta["frames"]:
        fname = Path(datapath) / Path(frame["file_path"].replace("./", "") + ".png")
        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))
        if "time" in frame:
            times.append(frame["time"])
        else:
            times.append((len(poses)-1)/len(meta))
    poses = np.array(poses).astype(np.float32)
    times = torch.tensor(times, dtype=torch.float32)

    camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
    # in x,y,z order
    camera_to_world[..., 3] *= scale_factor
    positions = camera_to_world[:,:,-1]

    dist = torch.sum((positions[:,None,:] - positions[None,:,:])**2, axis=-1)**0.5
    idx = torch.argsort(dist)[:,1:num+1]
    _ , tidx = torch.min(times[idx], dim=1)
    tidx =  tidx.squeeze()

    return idx, times, poses


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    idx, times, poses = index(args)
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, args.split, '*.png')) + \
                 glob.glob(os.path.join(args.path, args.split, '*.jpg'))

        images = sorted(images)
        assert (len(images) == len(idx))

        for i in tqdm(range(len(images))):
            for j in range(args.k_nearest):
                image1 = load_image(images[i])
                image2 = load_image(images[idx[i,j]])

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2) 

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                # viz(image1, image2, flow_up)

                if j == 0:
                    flows_ = flow_up
                else:
                    flows_ = torch.cat((flows_, flow_up))
                # print( '-----', i, idx[i,j])
            if i == 0:
                flows = flows_[None]
            else:
                flows = torch.cat((flows, flows_[None]))

        np.savez(args.path + f"/flows_{args.split}", flows=flows.cpu().numpy(), idx=idx.cpu().numpy())
        print('saved file to ', args.path + f"/flows_{args.split}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--split', default="train", help="train, test, or val")
    parser.add_argument('--k_nearest', default=5, help="number of nearest images to RAFT")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)

# python demo-sne.py --model=models/raft-kitti.pth --path=/ubc/cs/research/kmyi/svsamban/research/sdfstudio/data/dnerf/hook --mixed_precision --split val