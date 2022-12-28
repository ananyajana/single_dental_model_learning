import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import farthest_point_sample, index_points, square_distance


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PointTransformerCls(nn.Module):
    def __init__(self, num_classes=8, num_channels=24):
        super().__init__()
        #output_channels = cfg.num_class
        #d_points = cfg.input_dim

        output_channels = num_classes
        d_points = num_channels
        
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x[:, 9:12, :]
        B, D, N = xyz.size()
        print('B: {}, D: {}, N: {}'.format(B, D, N))
        xyz = xyz.contiguous().view(B, N, D)
        #x = x.permute(0, 2, 1)
        #xyz = x[..., :3]
        #x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        print('x size after conv1: ', x.size())
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        print('x size after conv2: ', x.size())
        x = x.permute(0, 2, 1)
        print('x size after permute: ', x.size())
        #new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        #new_xyz, new_feature = sample_and_group(npoint=2048, nsample=32, xyz=xyz, points=x)         
        new_xyz, new_feature = sample_and_group(npoint=N, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        print('x after linear: ', x.size())

        #x = F.log_softmax(x, dim=-1)
        return x

class Point_Transformer_partseg(nn.Module):
    def __init__(self, num_classes=8, num_channels=3):
        super(Point_Transformer_partseg, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   #nn.LeakyReLU(scale=0.2))
                                   nn.LeakyReLU(0.2))

        #self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                           nn.BatchNorm1d(64),
        #                           nn.LeakyReLU(scale=0.2))

        #self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.num_classes, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    #def execute(self, x, cls_label):
    def forward(self, x):
        batch_size, _, N = x.size()
        x = x[:, 9:12, :].contiguous() # take only the xyz channels and ignore the rest
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        #cls_label_one_hot = cls_label.view(batch_size,16,1)
        #cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        #x_global_feature = concat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1) # 1024
        x = torch.cat((x, x_global_feature), 1) # 1024 * 3
        x = self.relu(self.bns1(self.convs1(x)))
        #print('x size after convs1: ', x.size())
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        #print('x size after convs2: ', x.size())
        x = self.convs3(x)
        #print('x size after convs3: ', x.size())
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        #print('x size after softmax: ', x.size())
        return x

if __name__ == '__main__':
    #data = torch.rand(2, 24, 2048).cuda()
    data = torch.rand(2, 3, 2048).cuda()
    #norm = torch.rand(2, 3, 2048)
    #cls_label = torch.rand([2, 8])
    print("===> testing modelD ...")
    #model = pointMLP(50)
    model = Point_Transformer_partseg(num_classes=8, num_channels=3).cuda()
    #model = PointTransformerCls(num_classes=8, num_channels=24).cuda()
    #model = pointMLP_seg()
    #out = model(data, cls_label)  # [2,2048,50]
    #out = model(data, norm, cls_label)  # [2,2048,50]
    out = model(data)  # [2,2048,50]
    print(out.shape)
