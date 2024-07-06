import torch
import torch.nn as nn
import torch.nn.functional as F


class Matcher(nn.Module):
    """Attention Matcher: random, sort, hungary
    """
    def __init__(self, attention, attention_threshold, neg, geometry, geo_threshold=0.7, nn_match=False):
        super(Matcher, self).__init__()
        self.att = attention        # true
        self.att_threshold = attention_threshold       # 0.7
        self.k = neg        # 0.2
        self.geometry = geometry        # true
        self.geo_threshold = geo_threshold      # 0.7
        self.nn_match = nn_match        # true

    '''
       input: f [n, c, hw]
       计算输入特征向量f的自注意力矩阵
    '''
    @torch.no_grad()
    def _attention_match(self, f):
        f_at = f.pow(2).mean(1, keepdim=True) # b, 1, hw  按通道求平方，然后在通道维度上取平均值；
        f_at = F.normalize(f_at, p=2, dim=-1) # b, 1, hw
        # 最大-最小归一化 确保范围在[0, 1]之间
        f_at = (f_at - f_at.min(dim=-1, keepdim=True)[0]) / (f_at.max(dim=-1, keepdim=True)[0] - f_at.min(dim=-1, keepdim=True)[0] + 1e-10) # b, 1, hw
        # 根据阈值将f_at中大于阈值的设置为1，小于的设置为0
        f_at[f_at >= self.att_threshold] = 1
        f_at[f_at < self.att_threshold] = 0

        return f_at

    '''
        input: f1[0], f2[0]  b, c, hw
        计算两组特征之间的余弦相似度，并基于相似度生成正负匹配矩阵
    '''
    @torch.no_grad()
    def _cosine_pos_neg_match(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=1) # b, c, hw
        x2 = F.normalize(x2, p=2, dim=1) # b, c, hw
        cos = torch.matmul(x1.transpose(-2, -1), x2) # b, hw, hw  计算两组向量之间的余弦相似度
        neg = cos.clone()
        neg[neg < self.k] = 0
        neg[neg >= self.k] = 1
        if not self.nn_match:
            return [neg]
        pos_inds = cos.max(dim=-1, keepdim=True)[1]     # pos矩阵中的每一行，最大相似度对应的位置被设置为1，其余为0
        pos = torch.zeros_like(cos).scatter(dim=-1, index=pos_inds, src=torch.ones_like(cos))
        return [neg, pos]

    @torch.no_grad()
    def _geometry_match(self, coord_1, coord_2, size, pos_ratio=0.7):
        """
        coord_1, coord_2: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
        size (7, 7)
        返回生成的正匹配掩码pos_mask，这个掩码表示了两组坐标中，在几何上匹配的位置
        """
        H, W = size

        # generate center_coord, width, height
        # [1, H, W]
        x_array = torch.arange(0., float(W), dtype=coord_1.dtype, device=coord_1.device).reshape(1, 1, -1).repeat(1, H, 1)
        y_array = torch.arange(0., float(H), dtype=coord_1.dtype, device=coord_1.device).reshape(1, -1, 1).repeat(1, 1, W)
        # [N, 1, 1]  将网格坐标映射到一个固定尺寸的网格上，用于后续的几何匹配
        q_bin_width = ((coord_1[:, 2] - coord_1[:, 0]) / W).reshape(-1, 1, 1)
        q_bin_height = ((coord_1[:, 3] - coord_1[:, 1]) / H).reshape(-1, 1, 1)
        k_bin_width = ((coord_2[:, 2] - coord_2[:, 0]) / W).reshape(-1, 1, 1)
        k_bin_height = ((coord_2[:, 3] - coord_2[:, 1]) / H).reshape(-1, 1, 1)
        # [N, 1, 1]  左上角坐标
        q_start_x = coord_1[:, 0].reshape(-1, 1, 1)
        q_start_y = coord_1[:, 1].reshape(-1, 1, 1)
        k_start_x = coord_2[:, 0].reshape(-1, 1, 1)
        k_start_y = coord_2[:, 1].reshape(-1, 1, 1)
        # [N, 1, 1]  计算两组坐标中每个样本中bounding box在对角线上的长度，并取最大值
        q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
        k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
        max_bin_diag = torch.max(q_bin_diag, k_bin_diag)
        # [N, H, W]  计算每个样本bounding box的中心坐标
        center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
        center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
        center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
        center_k_y = (y_array + 0.5) * k_bin_height + k_start_y
        # [N, HW, HW]  计算中心坐标之间的欧氏距离，并归一化为最大对角线长度
        dist_center = torch.sqrt((center_q_x.reshape(-1, H * W, 1) - center_k_x.reshape(-1, 1, H * W)) ** 2
                                + (center_q_y.reshape(-1, H * W, 1) - center_k_y.reshape(-1, 1, H * W)) ** 2) / max_bin_diag
        # 通过判断距离是否小于pos_ratio生成二值化的正匹配掩码
        pos_mask = (dist_center < pos_ratio).float().detach()

        return pos_mask

    '''
       b, hw, hw
       计算不稳定匹配的数量  返回平均不稳定匹配的数量。
       不稳定匹配的概念可能与自注意力机制中的一些异常或不合理的匹配有关。
       通过统计不稳定匹配的数量，可以帮助评估自注意力机制的性能或稳定性。
    '''
    @torch.no_grad()
    def _count_unstable(self, mask_att, mask_final):
        count_att = mask_att.sum(dim=(-2, -1))
        count_final = mask_final.sum(dim=(-2, -1))

        return (count_att - count_final).mean()

    @torch.no_grad()
    def _count_geometry(self, mask_att, mask_final):
        count_att = mask_att.sum(dim=(-2, -1))
        count_final = mask_final.sum(dim=(-2, -1))

        return (count_final - count_att).mean()

    @torch.no_grad()
    def _count_neighbour(self, mask_final, mask_nn):
        count_final = mask_final.sum(dim=(-2, -1))
        mask_final = (mask_final + mask_nn).clamp_max(1)
        count_new = mask_final.sum(dim=(-2, -1))

        return mask_final, (count_new - count_final).mean()

    @torch.no_grad()
    # [q_b, coord_q], [k_b, coord_k]      q_b: NxCxHxW  k_b: NxCxHxW
    def forward(self, f1, f2):
        if f1[0].dim() == f2[0].dim() == 4:
            b, _, h, w = f1[0].size()
            f1[0] = f1[0].reshape(b, f1[0].size(1), h*w)    # q_b: NxCxH*W
            f2[0] = f2[0].reshape(b, f2[0].size(1), h*w)
        # similarity positive & negative match mask  b个[hw,hw]相似度矩阵
        mask_pair = self._cosine_pos_neg_match(f1[0], f2[0]) # b, hw, hw; 0 - 1
        # attention match mask
        if self.att and not self.geometry:
            att1 = self._attention_match(f1[0]) # b, 1,   hw
            att2 = self._attention_match(f2[0]) # b, 1, hw
            mask_att = torch.matmul(att1.transpose(-2, -1), att2) # b, hw, hw; 0 - 1
            mask_final = mask_att * mask_pair[0] # b, hw, hw; 0 - 1
            unstale = self._count_unstable(mask_att, mask_final)
            geometry = torch.zeros(1)
        # geometry match mask
        elif not self.att and self.geometry:
            mask_geo = self._geometry_match(f1[1], f2[1], (7, 7), self.geo_threshold)
            mask_final = mask_geo.clone()
            unstale = torch.zeros(1)
            geometry = mask_final.sum(dim=(-2, -1)).mean()
        # attention & geometry match mask
        elif self.att and self.geometry:
            # 得到特征的注意力矩阵
            att1 = self._attention_match(f1[0]) # b, 1, hw
            att2 = self._attention_match(f2[0]) # b, 1, hw
            # 计算两个自注意力矩阵的匹配矩阵
            mask_att = torch.matmul(att1.transpose(-2, -1), att2) # b, hw, hw; 0 - 1
            mask_att *= mask_pair[0] # b, hw, hw; 0 - 1     # mask_pair为二值化矩阵，相乘用于过滤不需要的匹配
            mask_final = mask_att.clone()
            unstale = self._count_unstable(mask_att, mask_final)

            mask_geo = self._geometry_match(f1[1], f2[1], (7, 7), self.geo_threshold)
            mask_final[mask_geo == 1] = 1       # 根据几何匹配结果，将匹配矩阵中对应位置设置为1，表示具有几何匹配
            geometry = self._count_geometry(mask_att, mask_final)       # 计算几何匹配的数量
        else:
            raise NotImplementedError('Wrong match modes!')
        # Nearest Neighbour sup
        if self.nn_match:
            valid = mask_final.sum(-1, keepdim=True).bool() # b, hw, 1
            mask_nn = mask_pair[1] * valid
            mask_final, neighbour = self._count_neighbour(mask_final, mask_nn)
        else:
            neighbour = torch.zeros(1)

        return mask_final, unstale, geometry, neighbour