import argparse
import json
from os.path import join

import tqdm
from mmcv import mkdir_or_exist

from visual_nuscenes import NuScenes

"""
use_gt = False
# out_dir = './result_vis/'
out_dir = './result_vis/PETRv2-vov-p4-800x320-nus/'
# result_json = "work_dirs/pp-nus/results_eval/pts_bbox/results_nusc"
result_json = "work_dirs/PETRv2-vov-p4-800x320-nus/results_eval/pts_bbox/results_nusc"
dataroot = '/mnt/datasets/nuScenes'
version = 'v1.0-mini'
# version = 'v1.0-trainval'
mkdir_or_exist(out_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True, pred=False,
                    annotations="sample_annotation")
else:
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True, pred=True, annotations=result_json,
                    score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

for token in tqdm.tqdm(tokens[:100]):
    if use_gt:
        nusc.render_sample(token, out_path="./result_vis/" + token + "_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path="./result_vis/" + token + "_pred.png", verbose=False)

"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet Result visualize')
    parser.add_argument('--dataroot', type=str,
                        default='/mnt/datasets/nuScenes/',
                        help='The nuScenes dataset dir')
    parser.add_argument('--out-dir', '-out', required=True,
                        help='output visual result')
    parser.add_argument('--result-json', type=str, required=True
                        , help='The result json file')
    parser.add_argument('--version', type=str,
                        help='The version of datasets nuScenes',
                        default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval'])
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--use-gt',
        action='store_true',
        help='whether to use ground truth.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    result_json = args.result_json
    if result_json.endswith('.json'):
        result_json = result_json[:-len('.json')]
    if args.use_gt:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True, pred=False,
                        annotations='sample_annotation')
    else:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True, pred=True,
                        annotations=result_json, score_thr=.25)

    with open('{}.json'.format(result_json)) as f:
        table = json.load(f)
    tokens = list(table['results'].keys())
    mkdir_or_exist(args.out_dir)
    for token in tqdm.tqdm(tokens[:100]):
        if args.use_gt:
            nusc.render_sample(token, out_path=join(args.out_dir, token + '-gt.png'), verbose=False)
        else:
            nusc.render_sample(token, out_path=join(args.out_dir, token + '-pred.png'), verbose=False)


if __name__ == '__main__':
    main()
