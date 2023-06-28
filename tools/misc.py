def position_embeding_constant(self, img_feats_shape, img2lidars):
    import torch
    from mmdet.models.utils.transformer import inverse_sigmoid
    eps = 1e-5
    # pad_h, pad_w, _ = img_metas[0]['pad_shape'][0] # 480, 800
    pad_h, pad_w = 480, 800
    depth_num = 64
    depth_start = 1
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

    coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
    coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
    # coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
    coords_mask = coords_mask.permute(0, 1, 3, 2)
    coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
    coords3d = inverse_sigmoid(coords3d)
    coords_position_embeding = self.position_encoder(coords3d)

    return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask
