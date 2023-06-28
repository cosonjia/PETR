import os.path as osp

import mmcv
import numpy as np
import onnxruntime as ort
import torch
from mmcv.image import tensor2imgs
from mmdet3d.core import (bbox3d2result)
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
# from .misc import position_embeding_constant


class ModelRunner:
    def __init__(self, model, use_gpu=True, depth_num=64):
        if use_gpu:
            self.session = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])
        else:
            self.session = ort.InferenceSession(model, providers=['CPUExecutionProvider'])

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.input_shapes = [i.shape for i in self.session.get_inputs()]
        self.embed_dims = 256
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        from torch import nn
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        ).cuda(device=0)
        self.load_checkpoint(mm=self.position_encoder, filename='ckpts/PETR-vov-p4-800x320.pth',
                             keys='position_encoder')

    def load_checkpoint(self, mm, filename, keys: list, revise_keys=None):
        from collections import OrderedDict
        from typing import List
        def match(target):
            for k in keys:
                if k in target:
                    return True
            return False

        # use _load_from_state_dict to enable checkpoint version control
        def load(module, prefix=''):
            # recursively check parallel module in case that the model has a
            # complicated structure, e.g., nn.Module(nn.Module(DDP))

            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                         all_missing_keys, unexpected_keys,
                                         err_msg)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        unexpected_keys: List[str] = []
        all_missing_keys: List[str] = []
        err_msg: List[str] = []
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        metadata = getattr(state_dict, '_metadata', OrderedDict())
        state_dict = OrderedDict({k: v for k, v in state_dict.items() if match(k)})
        load(mm, prefix='pts_bbox_head.position_encoder.')
        print(mm)

    def __call__(self, img, img_metas, **kwargs):
        input_feed = self.feed(img, img_metas)
        # np.save('input_feed',input_feed)
        outs = self.session.run(output_names=None, input_feed=input_feed)
        return outs

    def feed(self, img, img_metas, export=False):

        img_metas = img_metas[0].data[0][0]
        img_feats_shape = self.input_shapes[1]
        if len(img_feats_shape) == 4:
            img_feats_shape = img_feats_shape.insert(0, 1)
        img2lidars = []
        # dynamic lidar2img

        img2lidar = []
        for i in range(len(img_metas['lidar2img'])):
            img2lidar.append(np.linalg.inv(img_metas['lidar2img'][i]))
        img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        coords_position_embeding, _ = self.position_embeding_constant(img_feats_shape, img2lidars)
        if export:
            return img[0].data[0], coords_position_embeding
        pos_embed = coords_position_embeding.cpu().numpy()
        return {self.input_names[0]: img[0].data[0].cpu().numpy(),
                self.input_names[1]: pos_embed}

    def model_export(self, model, img, img_metas, model_file, **kwargs):
        import torch
        model.eval()
        img, pos_embed = self.feed(img, img_metas, export=True)
        # img = torch.from_numpy(inputs[0])
        # pos_embed = torch.from_numpy(inputs[1])
        pos_embed = pos_embed.cpu()
        img = img.cpu()
        torch.onnx.export(model,
                          # (False, {"img": img, "pos_embed": pos_embed}),
                          # ({"img": img, "pos_embed": pos_embed},{}),
                          (False, img, pos_embed),
                          f=model_file,
                          verbose=False,
                          # input_names=['img', 'pos_embed'],
                          output_names=['all_cls_scores', 'all_bbox_preds'],
                          do_constant_folding=False,
                          keep_initializers_as_inputs=False,
                          dynamic_axes=None,
                          )
        print(f"Export model to '{model_file}' successfully")
        return model_file

    def position_embeding_constant(self, img_feats_shape, img2lidars, depth_num=64, depth_start=1):
        import torch
        from mmdet.models.utils.transformer import inverse_sigmoid
        eps = 1e-5
        pad_h, pad_w = self.input_shapes[0][-2:]  # 480, 800
        # pad_h, pad_w = 320, 800

        position_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

        B, N, C, H, W = img_feats_shape  # (1,6,256,20,50)
        coords_h = torch.arange(H, device=torch.device('cuda:0')).float() * pad_h / H  # [20]
        coords_w = torch.arange(W, device=torch.device('cuda:0')).float() * pad_w / W  # [50]

        # if self.LID:
        index = torch.arange(start=0, end=depth_num, step=1, device=torch.device('cuda:0')).float()  # depth_num=64
        index_1 = index + 1
        # position_range[3]=61.2, self.depth_start=1
        bin_size = (position_range[3] - depth_start) / (depth_num * (1 + depth_num))
        coords_d = depth_start + bin_size * index * index_1

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)

        # img2lidars = []
        # dynamic lidar2img
        # for img_meta in img_metas:
        #     img2lidar = []
        #     for i in range(len(img_meta['lidar2img'])):
        #         img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
        #     img2lidars.append(np.asarray(img2lidar))
        # img2lidars = np.asarray(img2lidars)

        # load the constant img2lidar
        # img2lidars = np.fromfile('img2lidar_constant_1_6_4_4_fp64.bin', dtype=np.float64).reshape(1,6,4,4)

        # img2lidars.tofile('img2lidar_constant_1_6_4_4_fp64.bin')

        img2lidars = coords.new_tensor(img2lidars).cuda()  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - position_range[0]) / (position_range[3] - position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - position_range[1]) / (position_range[4] - position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - position_range[2]) / (position_range[5] - position_range[2])

        # coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        # coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        # coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        # coords_mask = coords_mask.permute(0, 1, 3, 2)
        coords_mask = None
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def postprocess(self, data):
        pass


def single_inference(model,
                     runner,
                     data_loader,
                     show=False,
                     out_dir=None,
                     show_score_thr=0.3,
                     model_file=None,
                     ):
    """Inference on model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    import torch
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    rescale = True
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if model_file:
                return runner.model_export(model, model_file=model_file, **data)
            else:
                outs = runner(return_loss=False, rescale=True, **data)
        img_metas = data['img_metas'][0].data[0]
        all_cls_scores, all_bbox_preds = outs
        preds_dicts = {'all_cls_scores': torch.from_numpy(all_cls_scores),
                       'all_bbox_preds': torch.from_numpy(all_bbox_preds)}
        bbox_list = model.pts_bbox_head.get_bboxes(
            preds_dicts, img_metas, rescale=rescale)
        result = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
