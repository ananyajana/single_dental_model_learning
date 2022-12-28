import os
import numpy as np
import torch
import torch.nn as nn
#from meshsegnet import *
#from pointnet import *
#from pointnet2_sem import *
#from tsgcn import *
from pointnet import PointNet_Seg
from pointnet2_sem import Pointnet2_seg
from tsgcn import TSGCN_seg
from dgcnn import DGCNN_semseg
from meshsegnet import MeshSegNet
from gac import GAC_seg
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
#from Mesh_dataset2 import get_keys
import h5py
from collections import OrderedDict
from vedo import *
import utils
import time
import logging
import torch.nn.functional as F

#if __name__ == '__main__':

np.random.seed(13)

#network='pointnet'
#network='pointnet2'
#network='dgcnn'
network='meshsegnet'
#network='gac'
#network='tsgcn'

# set up logger
save_dir='./test_logs_{}'.format(network)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
logger, logger_results = utils.setup_logger(save_dir, checkpoint=False)

save_mesh = True
#num_classes = 15
num_classes = 8
num_channels = 24
if network == 'meshsegnet':
    num_channels = 15
print(num_classes)

# DGCNN specific
#k = 20
k = 32
# k = 30 in TSGCNet
emb_dims = 1024
dropout = 0.5

#gpu_id = utils.get_avail_gpu()
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#gpu_id = 0
#torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

#file_path='../data/file_lists/'
#file_path='../data/file_lists_16k_8_class/'
file_path='../data/file_lists_16k_8_class_shuffled/'
file_list = '{}_list_{}.csv' # use 1-fold as example
#data_path='./data/tooth_data_folds_revised'
#data_path='../pred_4/data/tooth_data_folds_revised/'
#data_path = '../pred_43/data/tooth_data_folds_revised_16k_8_class/'
#data_path = '../pred_58/data/tooth_data_folds_revised_16k_8_class_shuffled/'
data_path = '../pred_58/data/tooth_data_folds_revised_10k_29march'

#test_src = '../data/all_vtps_combined'
#test_src = '../data/all_vtps_combined_flipped_16k_mesh_labeler_labeled'
#test_src = '../data/all_vtps_combined_flipped_16k_8_class'
#test_src = '../data/all_vtps_combined_flipped_10k'
test_src = '../data/all_vtps_combined_flipped_10k_8_class'


#mesh_path = './'  # need to define
#sample_filenames = ['Example.stl'] # need to define
output_path = './outputs_meshsegnet_10k'
if not os.path.exists(output_path):
    os.mkdir(output_path)


seed = 7
torch.manual_seed(seed)
################################### configs above

model_path = './models_{}'.format(network)
#model_path = './models_ptnet_test'
model_name = 'Mesh_Segementation_MeshSegNet_15_classes_60samples_best.tar' # need to define


# set model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if network == 'pointnet':
    model = PointNet_Seg(num_classes=num_classes, channel=num_channels).to(device, dtype=torch.float)
elif network == 'pointnet2':
    model = Pointnet2_seg(num_classes=num_classes, num_channels=num_channels)
elif network =='tsgcn':
    model = TSGCN_seg(num_classes=num_classes, num_channels=num_channels)
elif network =='dgcnn':
    model = DGCNN_semseg(num_classes=num_classes, num_channels=num_channels, k=k, emb_dims=emb_dims, dropout=dropout)
elif network == 'meshsegnet':
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5)
elif network =='gac':
    model = GAC_seg(num_classes=num_classes, num_channels=num_channels)
#model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

# load trained model
#checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
#model.load_state_dict(checkpoint['model_state_dict'])

print('loading checkpoint model')
chkpt = torch.load(os.path.join(model_path, model_name), map_location='cpu')['model_state_dict']
new_chkpt=OrderedDict()
for k, v in chkpt.items():
    name = k[7:] # remove module
    new_chkpt[name] = v
model.load_state_dict(new_chkpt)
#del checkpoint

model = model.to(device, dtype=torch.float)

#cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


num_keys = 1
class_weights = torch.ones(num_classes).to(device, dtype=torch.float)

# Predicting
model.eval()
with torch.no_grad():
    # for fold in range(1, 6): # currently we are not doing 5 fold cross validation
    #for fold in range(1, 2):
    #for fold in range(5, 6):
    for fold in range(1, 2):    
        # we will calculate th eloss for the entire fold
        losses, mdsc, msen, mppv = [], [], [], []
        
        # this is also applicable to each fold
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0

        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()

        end = time.time()
        # this array is to hold the classwise DSC scores
        all_dsc = torch.zeros(num_classes, 1).to(device, dtype=torch.float).squeeze()
        #all_dsc = np.zeros([num_classes, 1], dtype='float32')
        # total accuracy score
        total_acc = 0

        for phase_str in ['test']:
            h5_file = phase_str + '{}.h5'
            h5_filename = os.path.join(data_path, h5_file.format(fold))

            data_list = pd.read_csv(file_path + file_list.format(phase_str, fold), header=None)
            # the first entry is the column name, let's get rid of that
            data_list = data_list[1:]
            print('{} fold {} data_list len {}'.format(phase_str, fold, len(data_list)))

            for idx in range(len(data_list)):

                idx_mesh = data_list.iloc[idx][0] #vtk file name
                print('idx_mesh num', idx_mesh)

                h5_file = h5py.File(h5_filename, "r")
                keys = list(h5_file.keys())
                print('idx is: ', idx)
                print('keys are: ', keys)

                num_keys = len(keys)
                print('number of keys: ', num_keys)

                slide_data = h5_file[keys[idx]]

                X = None
                X = slide_data['data'][()]  # we can use all the 24 features
                Y = slide_data['label'][()]  # we can use all the 24 features
                #print('Y: ', Y)
                #raise ValueError("Exit!")
                cell_cnt = len(Y)
                #Y = Y.reshape([cell_cnt, 1])
                Y = Y.reshape([1, cell_cnt])
                #print('Y shape: ', Y.shape)

                labels = torch.from_numpy(Y).to(device, dtype=torch.long)
                outputs = None
                #if network == 'pointnet' or network == 'pointnet2' or network == 'meshsegnet':
                if network == 'meshsegnet':
                    #X = slide_data['data'][()][:, :15]  # we can use 15 channels for MeshSegNet
                    X = X[:, :15]
                    # computing A_S and A_L
                    A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
                    A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
                    D = distance_matrix(X[:, 9:12], X[:, 9:12])
                    A_S[D<0.1] = 1.0
                    A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

                    A_L[D<0.2] = 1.0
                    A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

                    # numpy -> torch.tensor
                    X = X.transpose(1, 0)
                    X = X.reshape([1, X.shape[0], X.shape[1]])
                    X = torch.from_numpy(X).to(device, dtype=torch.float)
                    A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
                    A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
                    A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
                    A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

                    outputs = model(X, A_S, A_L).to(device, dtype=torch.float)
                else:
                    #X = slide_data['data'][()]  # we can use all the 24 features
                    X = X.transpose(1, 0)
                    X = X.reshape([1, X.shape[0], X.shape[1]])
                    inputs = torch.from_numpy(X).to(device, dtype=torch.float)
                    outputs = model(inputs)

                op = outputs.view(-1, num_classes)
                lbl = labels.view(-1)


                #labels = Y.reshape([cell_cnt, 1], dtype='int32')
                #print('labels shape: ', labels.shape)
                labels = labels.unsqueeze(0)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                loss = 0.0
                if network == 'meshsegnet':
                    loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                else:
                    loss = F.nll_loss(op, lbl, weight=class_weights)

                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
                cur_dsc = batch_DSC(outputs, one_hot_labels, ignore_background=False)

                # do the classwise DICE score
                #print('all_dsc: ', all_dsc)
                #print('all_dsc size: ', all_dsc.size())
                #print('cur_dsc: ', torch.stack(cur_dsc.tolist()))
                #print('cur_dsc size: ', torch.stack(cur_dsc.tolist()).size())
                all_dsc = torch.add(all_dsc, torch.stack(cur_dsc.tolist()))
                #print('all_dsc is: ', all_dsc)

                #raise ValueError("Exit")


                print('outputs size:', outputs.size())
                #raise ValueError("Exit")
                # Do the overall accuracy
                total_acc += accuracy_check2(outputs, one_hot_labels).item()

                # print statistics
                running_loss += loss.item()
                running_mdsc += dsc.item()
                running_msen += sen.item()
                running_mppv += ppv.item()


                # calculate the labels
                patch_prob_output = outputs.cpu().numpy()

                #predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)
                predicted_labels_d = np.zeros([cell_cnt, 1], dtype=np.int32)

                for i_label in range(num_classes):
                    predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

                np.save(os.path.join(output_path, 'labels_{}_{}.npy'.format(idx_mesh, network)), predicted_labels_d)

                if save_mesh is True:
                    i_mesh = os.path.join(test_src, 'Sample_0{}_d.vtp'.format(idx_mesh, network))
                    mesh = load(i_mesh)

                    mesh.celldata['labels'] = predicted_labels_d
                    vedo.write(mesh, os.path.join(output_path, 'Sample_0{}_d_predicted_{}.vtp'.format(idx_mesh, network)))

                    print('Sample filename: {}  prediction completed'.format(i_mesh))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        epoch = 1
        num_epochs = 1
        i_batch = 0
        logger.info('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, 1, running_loss/num_keys, running_mdsc/num_keys, running_msen/num_keys, running_mppv/num_keys, batch_time.avg))
        logger.info('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, 1, running_loss, running_mdsc, running_msen, running_mppv, batch_time.avg))
        #logger.info('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print, batch_time.avg))
        #logger.info(' all dsc loss : classwise:(indices in the same order'.format(all_dsc/num_keys))
        res = torch.div(all_dsc, num_keys)
        print(' all dsc loss : classwise:(indices in the same order: {}'.format(torch.div(all_dsc, num_keys)))
        print(' overalll accuracy : {}'.format(total_acc/num_keys))

        total_dsc_np = torch.div(all_dsc, num_keys).detach().cpu().numpy().reshape(-1, 8)
        #numpy_data = np.array([[1, 2], [3, 4]])
        #df = pd.DataFrame(data=numpy_data, index=["row1", "row2"], columns=["column1", "column2"])
        #df.to_excel("output_{}.xlsx".format(network))
        
        df = pd.DataFrame(data=total_dsc_np, index=["row1"], columns=["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"])
        df.to_excel("dsc_{}.xlsx".format(network))
        df2 = pd.DataFrame(data=(total_acc/num_keys), index=["row1"], columns=["overall_accuracy"])
        df2.to_excel("overall_accuracy_{}.xlsx".format(network))
