import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_dataset2 import *
from meshsegnet import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd
import time
import logging

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

add_log = True

# set up logger
save_dir='./logs'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

logger, logger_results = utils.setup_logger(save_dir, checkpoint=False)


use_visdom = False # if you don't use visdom, please set to False


train_list='../pred_58/data/tooth_data_folds_revised_10k_8_class_single_tooth/train1.h5'
val_list='../pred_58/data/tooth_data_folds_revised_10k_8_class_single_tooth/val1.h5'

previous_check_point_path = './models'
previous_check_point_name = 'latest_checkpoint.tar'

model_path = './models/'
model_name = 'Mesh_Segementation_MeshSegNet_15_classes_60samples' # need to define
checkpoint_name = 'latest_checkpoint.tar'

#num_classes = 15
num_classes = 8
num_channels = 15 #number of features
num_epochs = 200
num_workers = 0
#train_batch_size = 10
#val_batch_size = 10
#train_batch_size = 20
#val_batch_size = 20
#train_batch_size = 80
#val_batch_size = 80

# 4gpu setting with 24gb memory each
#train_batch_size = 36
#val_batch_size = 36
#train_batch_size = 54
#val_batch_size = 54

# 6 gpu with 48 gb each
#train_batch_size = 72
#val_batch_size = 72
# did not work on santorini
#train_batch_size = 104
# 8 gpu with 48 gb each
train_batch_size = 40
val_batch_size = 40
# did not work on santorini
#train_batch_size = 104
#val_batch_size = 104

#train_batch_size = 32
#val_batch_size = 32
#train_batch_size = 14
#val_batch_size = 14
num_batches_to_print = 20
lr = 0.001
seed = 1

if use_visdom:
    # set plotter
    global plotter
    plotter = utils.VisdomLinePlotter(env_name=model_name)

if not os.path.exists(model_path):
    os.mkdir(model_path)

torch.manual_seed(seed)

# set dataset
# we will set the patch size to be bigger because our meshes are ~30k cells
#training_dataset = Mesh_Dataset(data_list_path=train_list,
training_dataset = Mesh_Dataset2(data_list_path=train_list,
                                num_classes=num_classes,
                                #patch_size=12000)
                                #patch_size=6000)
                                patch_size=8000)
                                #patch_size=16000)
                                #patch_size=22000)
#val_dataset = Mesh_Dataset(data_list_path=val_list,
val_dataset = Mesh_Dataset2(data_list_path=val_list,
                           num_classes=num_classes,
                           #patch_size=12000)
                           #patch_size=6000)
                           patch_size=8000)
                           #patch_size=16000)
                           #patch_size=22000)

train_loader = DataLoader(dataset=training_dataset,
                          batch_size=train_batch_size,
                          shuffle=True,
                          num_workers=num_workers)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        shuffle=False,
                        num_workers=num_workers)

# set model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
#model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5)
#model = model.cuda()
model = nn.DataParallel(model)
model = model.to(device, dtype=torch.float)
opt = optim.Adam(model.parameters(), amsgrad=True)
#opt = optim.Adam(model.parameters(), lr=lr)

losses, mdsc, msen, mppv = [], [], [], []
val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []

best_val_dsc = 0.0


'''
# re-load
checkpoint = torch.load(os.path.join(previous_check_point_path, previous_check_point_name), map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_init = checkpoint['epoch']
losses = checkpoint['losses']
mdsc = checkpoint['mdsc']
msen = checkpoint['msen']
mppv = checkpoint['mppv']
val_losses = checkpoint['val_losses']
val_mdsc = checkpoint['val_mdsc']
val_msen = checkpoint['val_msen']
val_mppv = checkpoint['val_mppv']
del checkpoint

best_val_dsc = max(val_mdsc)
'''

#cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

#print('Training model...')
logger.info('Training model...')
#class_weights = torch.ones(15).to(device, dtype=torch.float)
class_weights = torch.ones(num_classes).to(device, dtype=torch.float)

# batch accumulation parameter
#accum_iter = 8

for epoch in range(num_epochs):

    # training
    model.train()
    running_loss = 0.0
    running_mdsc = 0.0
    running_msen = 0.0
    running_mppv = 0.0
    loss_epoch = 0.0
    mdsc_epoch = 0.0
    msen_epoch = 0.0
    mppv_epoch = 0.0
    
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()

    end = time.time()

    
    for i_batch, batched_sample in enumerate(train_loader):

        # send mini-batch to device
        inputs = batched_sample['cells'].to(device, dtype=torch.float)
        labels = batched_sample['labels'].to(device, dtype=torch.long)
        A_S = batched_sample['A_S'].to(device, dtype=torch.float)
        A_L = batched_sample['A_L'].to(device, dtype=torch.float)
        one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, A_S, A_L)
        op = outputs.view(-1, num_classes)
        lbl = labels.view(-1)

        #loss = F.nll_loss(op, lbl, weight=class_weights)
        loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)

        # normalize loss to account for batch accumulation
        #loss = loss / accum_iter 
        
        dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
        sen = weighting_SEN(outputs, one_hot_labels, class_weights)
        ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
        loss.backward()
        opt.step()
        #opt.zero_grad()
        '''
        # weights update
        if ((i_batch + 1) % accum_iter == 0) or (i_batch + 1 == len(train_loader)):
            opt.step()
            opt.zero_grad()
        '''
        # print statistics
        running_loss += loss.item()
        running_mdsc += dsc.item()
        running_msen += sen.item()
        running_mppv += ppv.item()
        loss_epoch += loss.item()
        mdsc_epoch += dsc.item()
        msen_epoch += sen.item()
        mppv_epoch += ppv.item()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print, batch_time.avg))
            
        if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
            #print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print, batch_time.avg))
            logger.info('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print, batch_time.avg))

            running_loss = 0.0
            running_mdsc = 0.0
            running_msen = 0.0
            running_mppv = 0.0

    # record losses and metrics
    losses.append(loss_epoch/len(train_loader))
    mdsc.append(mdsc_epoch/len(train_loader))
    msen.append(msen_epoch/len(train_loader))
    mppv.append(mppv_epoch/len(train_loader))

    if add_log is True:
        writer.add_scalar("Loss/train", loss_epoch/len(train_loader), epoch)
        writer.add_scalar("mdsc/train", mdsc_epoch/len(train_loader), epoch)
        writer.add_scalar("msen/train", msen_epoch/len(train_loader), epoch)
        writer.add_scalar("mppv/train", mppv_epoch/len(train_loader), epoch)

    #reset
    loss_epoch = 0.0
    mdsc_epoch = 0.0
    msen_epoch = 0.0
    mppv_epoch = 0.0
    running_mppv = 0.0
    # validation
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        running_val_mdsc = 0.0
        running_val_msen = 0.0
        running_val_mppv = 0.0
        val_loss_epoch = 0.0
        val_mdsc_epoch = 0.0
        val_msen_epoch = 0.0
        val_mppv_epoch = 0.0
        
        batch_time2 = utils.AverageMeter()
        end = time.time()
        
        for i_batch, batched_val_sample in enumerate(val_loader):

            # send mini-batch to device
            inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
            labels = batched_val_sample['labels'].to(device, dtype=torch.long)
            A_S = batched_val_sample['A_S'].to(device, dtype=torch.float)
            A_L = batched_val_sample['A_L'].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            outputs = model(inputs, A_S, A_L)
            op = outputs.view(-1, num_classes)
            lbl = labels.view(-1)

            #loss = F.nll_loss(op, lbl, weight=class_weights)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

            running_val_loss += loss.item()
            running_val_mdsc += dsc.item()
            running_val_msen += sen.item()
            running_val_mppv += ppv.item()
            val_loss_epoch += loss.item()
            val_mdsc_epoch += dsc.item()
            val_msen_epoch += sen.item()
            val_mppv_epoch += ppv.item()
            
            # measure elapsed time
            batch_time2.update(time.time() - end)
            end = time.time()


            if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                #print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_mdsc/num_batches_to_print, running_val_msen/num_batches_to_print, running_val_mppv/num_batches_to_print, batch_time2.avg))
                logger.info('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}, Batch Time:{8}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_mdsc/num_batches_to_print, running_val_msen/num_batches_to_print, running_val_mppv/num_batches_to_print, batch_time2.avg))
                running_val_loss = 0.0
                running_val_mdsc = 0.0
                running_val_msen = 0.0
                running_val_mppv = 0.0

        # record losses and metrics
        val_losses.append(val_loss_epoch/len(val_loader))
        val_mdsc.append(val_mdsc_epoch/len(val_loader))
        val_msen.append(val_msen_epoch/len(val_loader))
        val_mppv.append(val_mppv_epoch/len(val_loader))

        if add_log is True:
            writer.add_scalar("Loss/val", loss_epoch/len(val_loader), epoch)
            writer.add_scalar("mdsc/val", mdsc_epoch/len(val_loader), epoch)
            writer.add_scalar("msen/val", msen_epoch/len(val_loader), epoch)
            writer.add_scalar("mppv/val", mppv_epoch/len(val_loader), epoch)

        # reset
        val_loss_epoch = 0.0
        val_mdsc_epoch = 0.0
        val_msen_epoch = 0.0
        val_mppv_epoch = 0.0

        # output current status
        #print('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
        logger.info('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
        if use_visdom:
            plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
            plotter.plot('DSC', 'train', 'DSC', epoch+1, mdsc[-1])
            plotter.plot('SEN', 'train', 'SEN', epoch+1, msen[-1])
            plotter.plot('PPV', 'train', 'PPV', epoch+1, mppv[-1])
            plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])
            plotter.plot('DSC', 'val', 'DSC', epoch+1, val_mdsc[-1])
            plotter.plot('SEN', 'val', 'SEN', epoch+1, val_msen[-1])
            plotter.plot('PPV', 'val', 'PPV', epoch+1, val_mppv[-1])

    # save the checkpoint
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'losses': losses,
                'mdsc': mdsc,
                'msen': msen,
                'mppv': mppv,
                'val_losses': val_losses,
                'val_mdsc': val_mdsc,
                'val_msen': val_msen,
                'val_mppv': val_mppv},
                model_path+checkpoint_name)

    # save the best model
    if best_val_dsc < val_mdsc[-1]:
        best_val_dsc = val_mdsc[-1]
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    model_path+'{}_best.tar'.format(model_name))

    # save all losses and metrics data
    pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'val_loss': val_losses, 'val_DSC': val_mdsc, 'val_SEN': val_msen, 'val_PPV': val_mppv}
    stat = pd.DataFrame(pd_dict)
    stat.to_csv('losses_metrics_vs_epoch.csv')


