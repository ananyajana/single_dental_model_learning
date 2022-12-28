import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_knn_idx(x, k=24):
    batch_size = x.size(0)
    #num_points = x.size(2)
    num_points = x.size(1)
    x = x.contiguous().view(batch_size, -1, num_points)
    idx = knn(x, k=k)   # (batch_size, num_points, k)
    idx = idx.view(batch_size, -1)
    return idx


def get_knn(x, k=24, dim9=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if dim9 == True:
        #idx = knn(x[:, 9:12], k=k)
        idx = knn(x[:, :9], k=k)
    else:
        idx = knn(x[:, :9], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    #feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    #feature = torch.cat((feature-x, x, feature), dim=3).permute(0, 3, 1, 2).contiguous()
    feature = torch.cat((feature, x, feature-x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        #self.in4 = nn.InstanceNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        #print('1 x size: ', x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        #print('2 x size: ', x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        #print('3 x size: ', x.size())
        x = torch.max(x, 2, keepdim=True)[0]
        #print('4 x size: ', x.size())
        x = x.view(-1, 512)
        #print('5 x size: ', x.size())

        #x = F.relu(self.fc1(x))
        x = F.relu(self.bn4(self.fc1(x)))
        #print('6 x size: ', x.size())
        '''
        #x = F.relu(self.bn4(self.fc1(x)))
        #print('6 x size: ', x.size())
        x = self.fc1(x)
        print('61 x size: ', x.size())
        #x = self.bn4(x)
        _,c = x.size()
        #x = x.view(batchsize, c, -1)
        x = x.view(batchsize, -1, c)
        print('611 x size: ', x.size())
        x = self.in4(x)
        print('62 x size: ', x.size())
        x = F.relu(x)
        print('63 x size: ', x.size())
        x = F.relu(self.bn5(self.fc2(x)))
        '''
        x = F.relu(self.bn5(self.fc2(x)))
        #x = F.relu(self.fc2(x))
        #print('7 x size: ', x.size())
        x = self.fc3(x)
        #print('8 x size: ', x.size())

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class MBESegNet(nn.Module):
    #def __init__(self, num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5):
    def __init__(self, num_classes=15, num_channels=15):
        super(MBESegNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        #self.with_dropout = with_dropout
        #self.dropout_p = dropout_p
        self.sem_feas = 6
        #self.k = k
        self.k = 4
        self.k2 = 16

        # feature extractor
        self.fea_conv1 = torch.nn.Conv1d(self.sem_feas, 16, 1)

        # BEM-S  mlps for generating p_tilde and f_tilde
        self.bems_mlp1 = nn.Conv1d(9+9, 9+9, 1) 
        self.bems_mlp2 = nn.Conv1d(16+16, 16+16, 1) 
        
        # BEM-S mlps for generating pi_cap and fi_cap
        self.bems_mlp3 = nn.Conv1d(16+16, 9, 1)
        self.bems_mlp4 = nn.Conv1d(9+9, 16, 1)

        # BEM-S
        self.bems_mlp5 = nn.Conv1d(9+9+9, 32, 1)
        self.bems_mlp6 = nn.Conv1d(16+16+16, 32, 1)

        # BEM-S attention conv
        self.bems_mlp7 = nn.Conv1d(32, 32, 1) # attention mlp

        # BEM-L  mlps for generating p_tilde and f_tilde
        self.beml_mlp1 = nn.Conv1d(9+9, 9+9, 1) 
        self.beml_mlp2 = nn.Conv1d(64+64, 64+64, 1) 
        
        # BEM-L mlps for generating pi_cap and fi_cap
        self.beml_mlp3 = nn.Conv1d(64+64, 9, 1)
        self.beml_mlp4 = nn.Conv1d(9+9, 64, 1)

        # BEM-L
        self.beml_mlp5 = nn.Conv1d(9+9+9, 128, 1)
        self.beml_mlp6 = nn.Conv1d(64+64+64, 128, 1)

        # BEM-L attention conv
        self.beml_mlp7 = nn.Conv1d(128, 128, 1) # attention mlp


        # BEM-L big mlps for generating p_tilde and f_tilde
        self.beml2_mlp1 = nn.Conv1d(9+9, 9+9, 1) 
        self.beml2_mlp2 = nn.Conv1d(64+64, 64+64, 1) 
        
        # BEM-L big mlps for generating pi_cap and fi_cap
        self.beml2_mlp3 = nn.Conv1d(64+64, 9, 1)
        self.beml2_mlp4 = nn.Conv1d(9+9, 64, 1)

        # BEM-L big
        self.beml2_mlp5 = nn.Conv1d(9+9+9, 128, 1)
        self.beml2_mlp6 = nn.Conv1d(64+64+64, 128, 1)

        # BEM-L big attention conv
        self.beml2_mlp7 = nn.Conv1d(128, 128, 1) # attention mlp

        # MLP last
        self.mlp_conv1 = torch.nn.Conv1d(64+512+512, 512, 1)
        self.mlp_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.mlp_conv3 = torch.nn.Conv1d(256, 128, 1)

        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)

        self.mlp_bn1 = nn.BatchNorm1d(512)
        self.mlp_bn2 = nn.BatchNorm1d(256)
        self.mlp_bn3 = nn.BatchNorm1d(128)


    #def forward(self, x, a_s, a_l):
    def forward(self, x):
        pi_list = []
        new_pi_list = []
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        coords = x[:, :9, :].clone()
        #print('coords ', coords)
        #print('coords requires_grad', coords.requires_grad)
        #print('x size: ', x.size())

        # convert the 6D feature vector to a 16D feature vector
        feas = self.fea_conv1(x[:, 9:15, :])
        #print('feas size: ', feas.size())

        #concat the feas to the original coords so that knn can be done for all at
        # the same time
        #coords = x[:, :9, :].copy()
        x_clone = x.clone()
        x = torch.cat((x[:, :9, :], feas), dim=1).contiguous()
        #print('x size after fea extractor: ', x.size())
        #raise ValueError("Exit!")

        # we get the knn for all the 
        x = get_knn(x, k=self.k)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        #print('x size after knn: ', x.size())
        #print('xc01 size before view: ', xc01.size())
        #_, c, p, _ = xc01.size()
        #xc01 = xc01.view(-1, c, p)
        a, b, c, d = x.size()
        x = x.view(a, b, -1) # we want the n_pts and k_neighbors to be multiplied before the conv happens

        # BEM-S coordinate operations
        pi = x[:, 25:34, :]
        pij = x[:, :9, :]
        inp_p = torch.cat((pi, pij), dim=1).contiguous()
        pij_tilde = self.bems_mlp1(inp_p)
        #print('pij_tilde size: ', pij_tilde.size())
        # cloning it for returning
        pi_0 = pi.clone()
        pi_list.append(pi_0)

        # BEM-S semantic features operations
        fi = x[:, 34:50, :]
        fij = x[:, 9:25, :]
        inp_f = torch.cat((fi, fij), dim=1).contiguous()
        fij_tilde = self.bems_mlp2(inp_f)
        #print('fij_tilde size: ', fij_tilde.size())

        # BEM-S get shifted pij
        fij_delta = x[:, 59:75 , :] 
        pij_cap = self.bems_mlp3(torch.cat((fij, fij_delta), dim=1)) + pij
        #print('pij_cap size: ', pij_cap.size())
        # cloning it for returning
        new_pi_0 = pij_cap.clone()
        new_pi_list.append(new_pi_0)

        # BEM-S get shifted fij
        pij_delta = x[:, 50:59 , :] 
        fij_cap = self.bems_mlp4(torch.cat((pij, pij_delta), dim=1)) + fij
        #print('fij_cap size: ', fij_cap.size())
        
        # calculating lpi and lfi
        lpi = self.bems_mlp5(torch.cat((pij_cap, pij_tilde), dim=1))
        lfi = self.bems_mlp6(torch.cat((fij_cap, fij_tilde), dim=1))
        #print('lpi size: ', lpi.size())
        #print('lfi size: ', lfi.size())

        # different local aggregation in BEM-S
        # attention weights for fi are used to aggregate pi
        #attn_wts = F.leaky_relu(self.bn7(self.conv7(attn_wts_ip)))
        attn_wts = self.bems_mlp7(lfi)
        #print('attn_wts size: ', attn_wts.size())
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(batchsize*self.k, -1, n_pts))
        #print('attn_wts size after softmax: ', attn_wts.size())
        a,b,c = lpi.size()
        lpi = lpi.view(a*self.k, b, -1)
        #print('lpi size after view: ', lpi.size())
        res = torch.mul(attn_wts, lpi) # res should have the size BxNx64x1
        #print('res size: ', res.size())
        res = res.view(batchsize, self.k, -1, n_pts)
        #print('res size after view: ', res.size())
        api = torch.sum(res, dim=1)
        #print('api size: ', api.size())

        # yield proper view of lfi
        a,b,c = lfi.size()
        lfi = lfi.view(a, self.k, b, -1)

        afi = torch.max(lfi, dim=1)[0] # res should have the size BxNx64x1, the max pooling is along the channel dim
        #print('afi size: ', afi.size())


        si = torch.cat((api, afi), dim=1) # this is 64D
        #print('si size: ', si.size())
        si_1 = si.clone()
        #raise ValueError("Exit!")

        #print("==========\nBEM-L starts\n==========")

        # BEM-L small knn

        #concat the feas to the original coords so that knn can be done for all at
        # the same time
        #coords = x[:, :9, :].copy()
        x = torch.cat((x_clone[:, :9, :], si), dim=1).contiguous()
        #print('x size after fea extractor: ', x.size())
        #raise ValueError("Exit!")

        # we get the knn for all the 
        x = get_knn(x, k=self.k, dim9=False)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        #print('x size after knn: ', x.size())
        #print('xc01 size before view: ', xc01.size())
        #_, c, p, _ = xc01.size()
        #xc01 = xc01.view(-1, c, p)
        a, b, c, d = x.size()
        x = x.view(a, b, -1) # we want the n_pts and k_neighbors to be multiplied before the conv happens

        # BEM-S coordinate operations
        # i = 9 + 64
        # j = 9 + 64 + 9
        pi = x[:, 73:82, :]
        # cloning it for returning
        pi_1 = pi.clone()
        pi_list.append(pi_1)
        pij = x[:, :9, :]
        inp_p = torch.cat((pi, pij), dim=1).contiguous()
        pij_tilde = self.beml_mlp1(inp_p)
        #print('pij_tilde size: ', pij_tilde.size())

        # BEM-S semantic features operations
        fi = x[:, 82:+146, :]
        fij = x[:, 9:73, :]
        inp_f = torch.cat((fi, fij), dim=1).contiguous()
        fij_tilde = self.beml_mlp2(inp_f)
        #print('fij_tilde size: ', fij_tilde.size())

        # BEM-S get shifted pij
        # i = 146 + 9
        # j = 146 + 9 + 64
        fij_delta = x[:, 155:219 , :] 
        pij_cap = self.beml_mlp3(torch.cat((fij, fij_delta), dim=1)) + pij
        #print('pij_cap size: ', pij_cap.size())
        # cloning it for returning
        new_pi_1 = pij_cap.clone()
        new_pi_list.append(new_pi_1)

        # BEM-S get shifted fij
        # i = 146
        # j = 155
        pij_delta = x[:, 146:155 , :] 
        fij_cap = self.beml_mlp4(torch.cat((pij, pij_delta), dim=1)) + fij
        #print('fij_cap size: ', fij_cap.size())
        
        # calculating lpi and lfi
        lpi = self.beml_mlp5(torch.cat((pij_cap, pij_tilde), dim=1))
        lfi = self.beml_mlp6(torch.cat((fij_cap, fij_tilde), dim=1))
        #print('lpi size: ', lpi.size())
        #print('lfi size: ', lfi.size())

        # different local aggregation in BEM-S
        # attention weights for fi are used to aggregate pi
        #attn_wts = F.leaky_relu(self.bn7(self.conv7(attn_wts_ip)))
        attn_wts = self.beml_mlp7(lfi)
        #print('attn_wts size: ', attn_wts.size())
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(batchsize*self.k, -1, n_pts))
        #print('attn_wts size after softmax: ', attn_wts.size())
        a,b,c = lpi.size()
        lpi = lpi.view(a*self.k, b, -1)
        #print('lpi size after view: ', lpi.size())
        res = torch.mul(attn_wts, lpi) # res should have the size BxNx64x1
        #print('res size: ', res.size())
        res = res.view(batchsize, self.k, -1, n_pts)
        #print('res size after view: ', res.size())
        api = torch.sum(res, dim=1)
        #print('api size: ', api.size())

        # yield proper view of lfi
        a,b,c = lfi.size()
        lfi = lfi.view(a, self.k, b, -1)

        afi = torch.max(lfi, dim=1)[0] # res should have the size BxNx64x1, the max pooling is along the channel dim
        #print('afi size: ', afi.size())


        si = torch.cat((api, afi), dim=1) # this is 64D
        #print('si size: ', si.size())
        si_2 = si.clone()
        #raise ValueError("Exit!")
    
        # BEM-L large knn context
        #print("==========\nBEM-L  2 starts\n==========")

        # BEM-L small knn

        #concat the feas to the original coords so that knn can be done for all at
        # the same time
        #coords = x[:, :9, :].copy()
        x = torch.cat((x_clone[:, :9, :], si), dim=1).contiguous()
        #print('x size after fea extractor: ', x.size())
        #raise ValueError("Exit!")

        # we get the knn for all the 
        x = get_knn(x, k=self.k2, dim9=False)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        #print('x size after knn: ', x.size())
        #print('xc01 size before view: ', xc01.size())
        #_, c, p, _ = xc01.size()
        #xc01 = xc01.view(-1, c, p)
        a, b, c, d = x.size()
        x = x.view(a, b, -1) # we want the n_pts and k_neighbors to be multiplied before the conv happens

        # BEM-S coordinate operations
        # i = 9 + 64
        # j = 9 + 64 + 9
        pi = x[:, 73:82, :]
        pij = x[:, :9, :]
        inp_p = torch.cat((pi, pij), dim=1).contiguous()
        pij_tilde = self.beml2_mlp1(inp_p)
        #print('pij_tilde size: ', pij_tilde.size())
        # cloning it for returning
        pi_2 = pi.clone()
        pi_list.append(pi_2)

        # BEM-S semantic features operations
        fi = x[:, 82:+146, :]
        fij = x[:, 9:73, :]
        inp_f = torch.cat((fi, fij), dim=1).contiguous()
        fij_tilde = self.beml2_mlp2(inp_f)
        #print('fij_tilde size: ', fij_tilde.size())

        # BEM-S get shifted pij
        # i = 146 + 9
        # j = 146 + 9 + 64
        fij_delta = x[:, 155:219 , :] 
        pij_cap = self.beml2_mlp3(torch.cat((fij, fij_delta), dim=1)) + pij
        #print('pij_cap size: ', pij_cap.size())
        # cloning it for returning
        new_pi_2 = pij_cap.clone()
        new_pi_list.append(new_pi_2)

        # BEM-S get shifted fij
        # i = 146
        # j = 155
        pij_delta = x[:, 146:155 , :] 
        fij_cap = self.beml2_mlp4(torch.cat((pij, pij_delta), dim=1)) + fij
        #print('fij_cap size: ', fij_cap.size())
        
        # calculating lpi and lfi
        lpi = self.beml2_mlp5(torch.cat((pij_cap, pij_tilde), dim=1))
        lfi = self.beml2_mlp6(torch.cat((fij_cap, fij_tilde), dim=1))
        #print('lpi size: ', lpi.size())
        #print('lfi size: ', lfi.size())

        # different local aggregation in BEM-S
        # attention weights for fi are used to aggregate pi
        #attn_wts = F.leaky_relu(self.bn7(self.conv7(attn_wts_ip)))
        attn_wts = self.beml2_mlp7(lfi)
        #print('attn_wts size: ', attn_wts.size())
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(batchsize*self.k2, -1, n_pts))
        #print('attn_wts size after softmax: ', attn_wts.size())
        a,b,c = lpi.size()
        lpi = lpi.view(a*self.k2, b, -1)
        #print('lpi size after view: ', lpi.size())
        res = torch.mul(attn_wts, lpi) # res should have the size BxNx64x1
        #print('res size: ', res.size())
        res = res.view(batchsize, self.k2, -1, n_pts)
        #print('res size after view: ', res.size())
        api = torch.sum(res, dim=1)
        #print('api size: ', api.size())

        # yield proper view of lfi
        a,b,c = lfi.size()
        lfi = lfi.view(a, self.k2, b, -1)

        afi = torch.max(lfi, dim=1)[0] # res should have the size BxNx64x1, the max pooling is along the channel dim
        #print('afi size: ', afi.size())


        si = torch.cat((api, afi), dim=1) # this is 64D
        #print('si size: ', si.size())
        si_3 = si.clone()
    
        # BEM-L large knn context   
        # concatenate the BEM-L features
        si_comb = torch.cat((si_2, si_3), dim=1)
        #print('si_comb size: ', si_comb.size())

        # GMP
        si_max = torch.max(si_comb, dim=2, keepdim=True)[0] # res should have the size BxNx512x1, max pooling is along the channel dim
        #print('si_max size: ', si_max.size())
        #x = torch.max(x_glm2, 2, keepdim=True)[0]

        # Upsample
        x = torch.nn.Upsample(n_pts)(si_max)
        #print('x size after upsample: ', x.size())

        # Dense fusion
        x = torch.cat([si_1, si_comb, x], dim=1)
        #print('x size after cat: ', x.size())


        x = F.relu(self.mlp_bn1(self.mlp_conv1(x)))
        #print('x size after conv1: ', x.size())
        x = F.relu(self.mlp_bn2(self.mlp_conv2(x)))
        #print('x size after conv2: ', x.size())
        x = F.relu(self.mlp_bn3(self.mlp_conv3(x)))
        #print('x size after conv3: ', x.size())
        #if self.with_dropout:
        #    x = self.dropout(x)
        # output
        x = self.output_conv(x)
        #print('x size after output_conv: ', x.size())
        x = x.transpose(2,1).contiguous()
        #print('x size after transpose: ', x.size())
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        #print('x size after softmax: ', x.size())
        x = x.view(batchsize, n_pts, self.num_classes)
        #print('x size after view: ', x.size())

        # we will write pij_cap as new pi
        #print('new_pi_list[0] size: ', new_pi_list[0].size())
        #print('pi_list[0] size: ', pi_list[0].size())
        return x, new_pi_list, pi_list

class bem_block(nn.Module):
    def __init__(self, in_c=[9,16], out_c=3, k=24):
        super(bem_block, self).__init__()

        self.channel = in_c[0]
        self.channel2 = in_c[1]
        self.channel3 = out_c
        self.k_n = k


        # BEM-S  mlps for generating p_tilde and f_tilde
        self.bems_mlp1 = nn.Conv1d(self.channel*2, self.channel*2, 1) 
        self.bems_mlp2 = nn.Conv1d(self.channel2*2, self.channel2*2, 1) 
        #self.bems_mlp1 = nn.Conv1d(9+9, 9+9, 1) 
        #self.bems_mlp2 = nn.Conv1d(16+16, 16+16, 1) 
        
        # BEM-S mlps for generating pi_cap and fi_cap
        self.bems_mlp3 = nn.Conv1d(self.channel2*2, self.channel, 1) 
        self.bems_mlp4 = nn.Conv1d(self.channel*2, self.channel2, 1) 
        #self.bems_mlp3 = nn.Conv1d(16+16, 9, 1)
        #self.bems_mlp4 = nn.Conv1d(9+9, 16, 1)

        # BEM-S
        self.bems_mlp5 = nn.Conv1d(self.channel*3, self.channel3, 1) 
        self.bems_mlp6 = nn.Conv1d(self.channel2*3, self.channel3, 1) 
        #self.bems_mlp5 = nn.Conv1d(9+9+9, 32, 1)
        #self.bems_mlp6 = nn.Conv1d(16+16+16, 32, 1)

        # BEM-S attention conv
        self.bems_mlp7 = nn.Conv1d(self.channel3, self.channel3, 1) # attention mlp

    def forward(self, input_xyz, feature, input_neigh_idx):
        B, N, D = input_xyz.size()
        input_neigh_idx = input_neigh_idx.unsqueeze(2)

        num_points = input_xyz.size()[1]

        a, b, _ = input_neigh_idx.size()
        #B = a
        #print('input_nei_idx size: ', input_neigh_idx.size())
        _, _, c = input_xyz.size()
        #print('input_xyz size: ', input_xyz.size())
        input_neigh_idx1 = input_neigh_idx.expand((a, b, c))
        #print('input_nei_idx1 size: ', input_neigh_idx1.size())
        neigh_xyz = torch.gather(input_xyz, 1, input_neigh_idx1) 
        #print('neigh_xyz[0] size: ', neigh_xyz[0].size())
        #print('neigh_xyz size: ', neigh_xyz.size())

        feature = feature.permute(0, 2, 1)
        #print('feature size: ', feature.size())
        _, _, c= feature.size()
        input_neigh_idx2 = input_neigh_idx.expand((a, b, c))
        #input_neigh_idx2 = input_neigh_idx2.unsqueeze(3)
        neigh_feat = torch.gather(feature, 1, input_neigh_idx2) 
        #print('4 neigh_feat size: ', neigh_feat.size())

        #tile_feat = feature.tile(1, 1, self.k_n, 1) # B * N * k * d_out/2
        tile_feat = torch.unsqueeze(feature, 2).tile(1, 1, self.k_n, 1) # B * N * k * d_out/2
        #tile_xyz = input_xyz.tile(1, 1, self.k_n, 1) # B * N * k * 3
        #print('xyz size: ', input_xyz.size())
        tile_xyz = torch.unsqueeze(input_xyz, 2).tile(1, 1, self.k_n, 1) # B * N * k * 3
        # we are making every dim in pytorch convention
        tile_feat = tile_feat.permute(0, 3, 2, 1) # B, 3, k, N
        tile_xyz = tile_xyz.permute(0, 3, 2, 1) # B, 3, k, N
        # shaping the neigh_xyz as tile_xyz
        a, b, _, _ = tile_xyz.size()
        tile_xyz = tile_xyz.reshape(a, b, -1)

        # shaping the neigh_feat as tile_feat
        a, b, _, _ = tile_feat.size()
        tile_feat = tile_feat.reshape(a, b, -1)
        #print('tile_xyz size: ', tile_xyz.size())
        #print('tile_feat size: ', tile_feat.size())

        neigh_feat = neigh_feat.permute(0, 2, 1)
        neigh_xyz = neigh_xyz.permute(0, 2, 1)

        # BEM-S coordinate operations
        inp_p = torch.cat((tile_xyz, neigh_xyz), dim=1).contiguous()
        pij_tilde = self.bems_mlp1(inp_p)
        #print('pij_tilde size: ', pij_tilde.size())

        # BEM-S semantic features operations
        inp_f = torch.cat((tile_feat, neigh_feat), dim=1).contiguous()
        fij_tilde = self.bems_mlp2(inp_f)
        #print('fij_tilde size: ', fij_tilde.size())

        # BEM-S get shifted pij or pij_cap
        shifted_neigh_xyz = self.bems_mlp3(torch.cat((neigh_feat, tile_feat - neigh_feat), dim=1)) + neigh_xyz
        #print('shifted_neigh_xyz size: ', shifted_neigh_xyz.size())

        # BEM-S get shifted fij or fij_cap
        shifted_neigh_feat = self.bems_mlp4(torch.cat((neigh_xyz, tile_xyz - neigh_xyz), dim=1)) + neigh_feat
        #print('shifted_neigh_feat size: ', shifted_neigh_feat.size())

        # calculating lpi and lfi
        lpi = self.bems_mlp5(torch.cat((shifted_neigh_xyz, pij_tilde), dim=1))
        lfi = self.bems_mlp6(torch.cat((shifted_neigh_feat, fij_tilde), dim=1))
        #print('lpi size: ', lpi.size())
        #print('lfi size: ', lfi.size())

        # different local aggregation in BEM-S
        # attention weights for fi are used to aggregate pi
        #attn_wts = F.leaky_relu(self.bn7(self.conv7(attn_wts_ip)))
        attn_wts = self.bems_mlp7(lfi)
        #print('attn_wts size: ', attn_wts.size())
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*self.k_n, -1, N))
        #print('attn_wts size after softmax: ', attn_wts.size())
        a,b,c = lpi.size()
        lpi = lpi.view(a*self.k_n, b, -1)
        #print('lpi size after view: ', lpi.size())
        res = torch.mul(attn_wts, lpi) # res should have the size BxNx64x1
        #print('res size: ', res.size())
        res = res.view(B, self.k_n, -1, N)
        #print('res size after view: ', res.size())
        api = torch.sum(res, dim=1)
        #print('api size: ', api.size())

        # yield proper view of lfi
        a,b,c = lfi.size()
        lfi = lfi.view(a, self.k_n, b, -1)

        afi = torch.max(lfi, dim=1)[0] # res should have the size BxNx64x1, the max pooling is along the channel dim
        #print('afi size: ', afi.size())


        si = torch.cat((api, afi), dim=1) # this is 64D
        #print('si size: ', si.size())
        #raise ValueError("Exit!")
        return si, shifted_neigh_xyz 


class MBESegNet2(nn.Module):
    #def __init__(self, num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5):
    def __init__(self, num_classes=15, num_channels=15):
        super(MBESegNet2, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        #self.with_dropout = with_dropout
        #self.dropout_p = dropout_p
        #self.sem_feas = 6
        self.sem_feas = 15
        #self.k = k
        self.k = 4
        self.k2 = 16

        self.channels = [9, 16, 64]
        self.in_channels = [[9, 16],[9, 64], [9, 64]]
        self.out_channels = [32, 128, 128]
        self.mbesegnet_blocks = nn.ModuleList()

        # TBD: convert this to a loop
        #self.bem1 = bem_block(in_c=self.channels[0], out_c=self.channels[1],  k=self.k)
        #self.bem1 = bem_block(in_c=self.in_channels[0], out_c=self.out_channels[0],  k=self.k)
        self.bem1 = bem_block(in_c=self.in_channels[0], out_c=self.out_channels[0], k=self.k)
        self.mbesegnet_blocks.append(self.bem1)
        # BEM-L small knn
        #self.bem1 = bem_block(in_c=self.channels[0], out_c=self.channels[2],  k=self.k)
        #self.bem1 = bem_block(in_c=self.in_channels[1], out_c=self.out_channels[1],  k=self.k)
        self.bem1 = bem_block(in_c=self.in_channels[1], out_c=self.out_channels[1], k=self.k)
        self.mbesegnet_blocks.append(self.bem1)
        # BEM-L big knn
        #self.bem1 = bem_block(in_c=self.channels[0], out_c=self.channels[2],  k=self.k2)
        #self.bem1 = bem_block(in_c=self.in_channels[2], out_c=self.out_channels[2],  k=self.k2)
        self.bem1 = bem_block(in_c=self.in_channels[2], out_c=self.out_channels[2], k=self.k2)
        self.mbesegnet_blocks.append(self.bem1)

        # feature extractor
        self.fea_conv1 = torch.nn.Conv1d(self.sem_feas, 16, 1)


        # MLP last
        self.mlp_conv1 = torch.nn.Conv1d(64+512+512, 512, 1)
        self.mlp_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.mlp_conv3 = torch.nn.Conv1d(256, 128, 1)

        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)

        self.mlp_bn1 = nn.BatchNorm1d(512)
        self.mlp_bn2 = nn.BatchNorm1d(256)
        self.mlp_bn3 = nn.BatchNorm1d(128)


    def forward(self, x):
        B, D, N = x.size()

        # Feature Extraction
        feas = self.fea_conv1(x)
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        coords = x[:, :9, :].clone()
        centers = x[:, 9:12, :].clone().permute(0, 2, 1)
        #x = x.permute(0, 2, 1)
        #print('x size before knn: ', x.size())

        # for BEM-S knn and BEM-L small knn
        input_neigh_idx = get_knn_idx(centers, k=self.k)
        #input_neigh_idx = self.batch_knn(xc, xc, self.k_n)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        #input_neigh_idx = input_neigh_idx.unsqueeze(2)

        # for BEM-L large knn
        input_neigh_idx2 = get_knn_idx(centers, k=self.k2)
        #print('x size: ', x.size())
        #input_neigh_idx2 = input_neigh_idx2.unsqueeze(2)

        new_pij_list = []
        # convert the 15D feature vector to a 16D feature vector
        #print('feas size: ', feas.size())

        #concat the feas to the original coords so that knn can be done for all at
        # the same time
        #coords = x[:, :9, :].copy()
        x_clone = x.clone()
        x = torch.cat((x[:, :9, :], feas), dim=1).contiguous()
        #print('x size after fea extractor: ', x.size())
        #raise ValueError("Exit!")

        input_xyz = coords.reshape(B, N, -1)
        #print("==========\nBEM-S starts\n==========")
        i=0
        si_1, shifted_neigh_xyz1 = self.mbesegnet_blocks[i](input_xyz, feas, input_neigh_idx)
        new_pij_list.append(shifted_neigh_xyz1)
        i += 1

        #print("==========\nBEM-L starts\n==========")
        # both the next modules get the same inputs, but calculates different outputs based on the knn
        si_2, shifted_neigh_xyz2 = self.mbesegnet_blocks[i](input_xyz, si_1, input_neigh_idx)
        new_pij_list.append(shifted_neigh_xyz2)
        i += 1

        si_3, shifted_neigh_xyz3 = self.mbesegnet_blocks[i](input_xyz, si_1, input_neigh_idx2)
        new_pij_list.append(shifted_neigh_xyz3)
        i += 1


        # concatenate the BEM-L features
        si_comb = torch.cat((si_2, si_3), dim=1)
        #print('si_comb size: ', si_comb.size())

        # GMP
        si_max = torch.max(si_comb, dim=2, keepdim=True)[0] # res should have the size BxNx512x1, max pooling is along the channel dim
        #print('si_max size: ', si_max.size())

        # Upsample
        x = torch.nn.Upsample(N)(si_max)
        #print('x size after upsample: ', x.size())

        # Dense fusion
        x = torch.cat([si_1, si_comb, x], dim=1)
        #print('x size after cat: ', x.size())


        x = F.relu(self.mlp_bn1(self.mlp_conv1(x)))
        #print('x size after conv1: ', x.size())
        x = F.relu(self.mlp_bn2(self.mlp_conv2(x)))
        #print('x size after conv2: ', x.size())
        x = F.relu(self.mlp_bn3(self.mlp_conv3(x)))
        #print('x size after conv3: ', x.size())
        #if self.with_dropout:
        #    x = self.dropout(x)
        # output
        x = self.output_conv(x)
        #print('x size after output_conv: ', x.size())
        x = x.transpose(2,1).contiguous()
        #print('x size after transpose: ', x.size())
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        #print('x size after softmax: ', x.size())
        x = x.view(B, N, self.num_classes)
        #print('x size after view: ', x.size())

        # we will write pij_cap as new pi
        #print('new_pij_list[0] size: ', new_pij_list[0].size())
        #raise ValueError("Exit!")
        #print('pi_list[0] size: ', pi_list[0].size())
        return x, new_pij_list, coords

    def batch_knn(self, x, y, k):
        batch_sz, _, _ = x.size()
        all_tensors = []
        for i in range(batch_sz):
            cur_tensor = x[i]
            cur_tensor2 = y[i]
            #print('cur_tensor size: ', cur_tensor.size())
            cur_neighbor = torch_cluster_knn(cur_tensor, cur_tensor2, k)[1]
            #print('cur_neighbor size: ', cur_neighbor.size())
            # append all neighbors for a particular sample
            all_tensors.append(cur_neighbor)

        # re arrange the neighbor indices in tensor format
        input_neigh_idx = torch.stack(all_tensors, dim=0)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        return input_neigh_idx

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet().to(device)
    summary(model, [(15, 6000), (6000, 6000), (6000, 6000)])
