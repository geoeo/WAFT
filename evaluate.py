import os
import sys
import csv
import time
import cv2
import math
import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

from tqdm import tqdm
from model import fetch_model
from utils.utils import resize_data, load_ckpt
from utils.flow_viz import flow_to_image
from custom import get_heatmap, vis_heatmap

from dataloader.flow.chairs import FlyingChairs
from dataloader.flow.things import FlyingThings3D
from dataloader.flow.sintel import MpiSintel
from dataloader.flow.kitti import KITTI
from dataloader.flow.spring import Spring
from dataloader.flow.hd1k import HD1K
from dataloader.stereo.tartanair import TartanAir
from dataloader.flow.realm import Realm

from inference_tools import InferenceWrapper, AverageMeter

val_loss = AverageMeter()
val_epe = AverageMeter()
val_fl = AverageMeter()
val_px1 = AverageMeter()

def reset_all_metrics():
    val_loss.reset()
    val_epe.reset()
    val_fl.reset()
    val_px1.reset()

def update_metrics(args, output, flow_gt, valid):
    flow = output['flow'][-1]
    batch_size = flow.shape[0]
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    val = valid >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    px1 = (epe < 1.0).float()
    nf = []
    for i in range(len(output['flow'])):                 
        raw_b = output['info'][i][:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = output['info'][i][:, :2]
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
        term2 = ((flow_gt - output['flow'][i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        term1 = weight - math.log(2) - log_b
        nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        nf.append(nf_loss.mean(dim=1))

    loss = torch.zeros_like(nf[-1])
    for i in range(len(nf)):
        loss += (args.gamma ** (len(nf) - i - 1)) * nf[i]
    for i in range(batch_size):
        val_epe.update(epe[i][val[i]].mean().item(), 1)
        val_px1.update(px1[i][val[i]].mean().item(), 1)
        val_fl.update(100 * out[i][val[i]].sum().item(), val[i].sum().item())
        val_loss.update(loss[i][val[i]].mean().item(), 1)

@torch.no_grad()
def validate_sintel(args, model):
    """ Peform validation using the Sintel (train) split """
    for dstype in ['clean', 'final']:
        reset_all_metrics()
        val_dataset = MpiSintel(split='training', dstype=dstype)
        val_loader = data.DataLoader(val_dataset, batch_size=1, 
            pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
        pbar = tqdm(total=len(val_loader))
        print(f"load data success {len(val_loader)}")
        for i_batch, data_blob in enumerate(val_loader):
            image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
            output = model.calc_flow(image1, image2)
            update_metrics(args, output, flow_gt, valid)
            pbar.update(1)
        pbar.close()
        print(f"Validation {dstype} EPE: {val_epe.avg}, 1px: {100 * (1 - val_px1.avg)}")

@torch.no_grad()
def validate_kitti(args, model):
    """ Peform validation using the KITTI-2015 (train) split """
    val_dataset = KITTI(split='training')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    print(f"load data success {len(val_loader)}")
    reset_all_metrics()
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        output = model.calc_flow(image1, image2)
        update_metrics(args, output, flow_gt, valid_gt)
    
    print("Validation KITTI: %f, %f" % (val_epe.avg, val_fl.avg))

@torch.no_grad()
def validate_spring(args, model):
    """ Peform validation using the Spring (val) split """
    val_dataset = Spring(split='val') #+ Spring(split='train')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    
    reset_all_metrics()
    print(f"load data success {len(val_loader)}")
    pbar = tqdm(total=len(val_loader))
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        output = model.calc_flow(image1, image2)
        update_metrics(args, output, flow_gt, valid)
        pbar.update(1)

    pbar.close()
    print(f"Validation Spring EPE: {val_epe.avg}, 1px: {100 * (1 - val_px1.avg)}, loss: {val_loss.avg}")

@torch.no_grad()
def process_realm(args, model, output_path=None):
    """ Run inference on consecutive keyframes of an OpenREALM sequence.
        The sequence has no optical-flow ground truth, so no error metrics are computed.
        Per keyframe pair the raw flow (.npy), a flow visualisation and the uncertainty
        heatmap are written to `output_path` (default outputs/<sequence name>), plus a stats.csv with per-pair
        statistics including the inference time (one untimed warm-up pass precedes the loop). """
    dataset = Realm(root=args.data_root)
    if output_path is None:
        output_path = osp.join('outputs', osp.basename(osp.normpath(args.data_root)))
    os.makedirs(output_path, exist_ok=True)
    print(f"load data success {len(dataset)}")

    # warm-up so the first timed pair does not include CUDA / cuDNN initialisation
    image1, image2, _ = dataset[0]
    model.calc_flow(image1[None].cuda(), image2[None].cuda())
    torch.cuda.synchronize()

    rows = []
    pbar = tqdm(total=len(dataset))
    for index in range(len(dataset)):
        image1, image2, (ts1, ts2) = dataset[index]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.calc_flow(image1, image2)
        torch.cuda.synchronize()
        inference_time_ms = (time.perf_counter() - start) * 1000.0
        flow = output['flow'][-1][0].permute(1, 2, 0).cpu().numpy()
        name = f"{ts1}_{ts2}"
        np.save(osp.join(output_path, f"flow_{name}.npy"), flow)
        cv2.imwrite(osp.join(output_path, f"flow_{name}.png"), flow_to_image(flow, convert_to_bgr=True))

        magnitude = np.linalg.norm(flow, axis=-1)
        row = {'frame1': ts1, 'frame2': ts2, 'inference_time_ms': inference_time_ms,
               'flow_mean_px': magnitude.mean(), 'flow_max_px': magnitude.max()}
        if 'info' in output:
            heatmap = get_heatmap(output['info'][-1], args)
            vis_heatmap(osp.join(output_path, f"heatmap_{name}.png"),
                        image1[0].permute(1, 2, 0).cpu().numpy(),
                        heatmap[0].permute(1, 2, 0).cpu().numpy())
            row['uncertainty_mean'] = heatmap.mean().item()
        rows.append(row)
        pbar.update(1)
    pbar.close()

    with open(osp.join(output_path, 'stats.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    flow_mean = np.mean([r['flow_mean_px'] for r in rows])
    time_mean = np.mean([r['inference_time_ms'] for r in rows])
    summary = f"Processed {len(rows)} keyframe pairs from {args.data_root}: mean inference time {time_mean:.1f} ms, mean flow magnitude {flow_mean:.2f} px"
    if 'uncertainty_mean' in rows[0]:
        summary += f", mean uncertainty {np.mean([r['uncertainty_mean'] for r in rows]):.3f}"
    print(summary + f". Outputs written to {output_path}")

def eval(args):
    args.gpus = [0]
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.cuda()
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)

    with torch.no_grad():
        if args.dataset == 'spring':
            validate_spring(args, wrapped_model)
        elif args.dataset == 'sintel':
            validate_sintel(args, wrapped_model)
        elif args.dataset == 'kitti':
            validate_kitti(args, wrapped_model)
        elif args.dataset == 'realm':
            process_realm(args, wrapped_model, output_path=getattr(args, 'output_path', None))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--ckpt', help='checkpoint path', required=True, type=str)
    parser.add_argument('--dataset', help='dataset to evaluate on', choices=['sintel', 'kitti', 'spring', 'realm'], required=True, type=str)
    parser.add_argument('--scale', help='scale factor for input images', default=0.0, type=float)
    parser.add_argument('--data_root', help='root of an OpenREALM sequence (realm dataset only)', default='custom/realm_1_100m_w_tandem', type=str)
    parser.add_argument('--output_path', help='where to write results (realm dataset only, default outputs/<sequence name>)', default=None, type=str)
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()
