#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:14:36 2022

@author: aj611
"""

from vedo import *
import os

dataurl = '../../outputs/'
#file1 = os.path.join(dataurl+'Sample_010_d.vtp')
#s = load(file1)
#s = Mesh(os.path.join(dataurl+'Sample_010_d.vtp')) 
#s = Mesh(os.path.join(dataurl+'Sample_01_d.vtp')) # good example
#s = Mesh(os.path.join(dataurl+'Sample_015_d.vtp')) # gum recession
#s = Mesh(os.path.join(dataurl+'Sample_037_d.vtp')) 
#s = Mesh(os.path.join(dataurl+'Sample_044_d.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_047_d.vtp')) # missing teeth
#s = Mesh(os.path.join(dataurl+'Sample_048_d.vtp')) # missing teeth

# pointnet test sample
#s = Mesh(os.path.join(dataurl+'Sample_02_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_03_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_04_d_predicted_pointnet.vtp')) # full wrong prediction in the last two teeth
#s = Mesh(os.path.join(dataurl+'Sample_05_d_predicted_pointnet.vtp')) # bad prediction
#s = Mesh(os.path.join(dataurl+'Sample_06_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_07_d_predicted_pointnet.vtp')) # full wrong pred
#s = Mesh(os.path.join(dataurl+'Sample_08_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_09_d_predicted_pointnet.vtp')) 
#s = Mesh(os.path.join(dataurl+'Sample_010_d_predicted_pointnet.vtp'))   

# pointnet2 test sample
#s = Mesh(os.path.join(dataurl+'Sample_02_d_predicted_pointnet2.vtp')) # this sample has some tooth alignment issues
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_pointnet2.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_03_d_predicted_pointnet2.vtp')) # this sample has empty space between teeth
#s = Mesh(os.path.join(dataurl+'Sample_04_d_predicted_pointnet2.vtp')) # full wrong prediction
#s = Mesh(os.path.join(dataurl+'Sample_05_d_predicted_pointnet2.vtp'))  # full wrong prediction, this sample has alignment issues
#s = Mesh(os.path.join(dataurl+'Sample_06_d_predicted_pointnet2.vtp')) # wrong prediciton
#s = Mesh(os.path.join(dataurl+'Sample_07_d_predicted_pointnet2.vtp'))  # better than pointnet version probably
#s = Mesh(os.path.join(dataurl+'Sample_08_d_predicted_pointnet2.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_09_d_predicted_pointnet2.vtp'))   # sample has gap between teeth
#s = Mesh(os.path.join(dataurl+'Sample_010_d_predicted_pointnet2.vtp'))

# tsgcn
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_02_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_03_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_04_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_05_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_06_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_07_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_08_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_09_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_010_d_predicted_tsgcn.vtp'))

# meshsegnet
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_02_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_03_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_04_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_05_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_06_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_07_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_08_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_09_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_010_d_predicted_meshsegnet.vtp'))

#dgcnn
#s = Mesh(os.path.join(dataurl+'Sample_044_d_predicted_dgcnn.vtp'))



#tsgcn
#s = Mesh(os.path.join(dataurl+'Sample_047_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_03_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_049_d_predicted_tsgcn.vtp'))


# meshsegnet
#s = Mesh(os.path.join(dataurl+'Sample_047_d_predicted_meshsegnet.vtp'))


# pointnet test sample, 12 tooth and problem
#s = Mesh(os.path.join(dataurl+'Sample_044_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_047_d_predicted_pointnet.vtp')) # bad example
#s = Mesh(os.path.join(dataurl+'Sample_015_d_predicted_pointnet.vtp'))

# test 1 fold sub jects list

#tsgcn
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_02_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_03_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_04_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_05_d_predicted_tsgcn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_06_d_predicted_tsgcn.vtp'))


########################################## Test fold 1

#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_pointnet2.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_dgcnn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_meshsegnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_015_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_015_d_predicted_pointnet2.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_015_d_predicted_dgcnn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_015_d_predicted_meshsegnet.vtp'))


#s = Mesh(os.path.join(dataurl+'Sample_037_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_037_d_predicted_pointnet2.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_037_d_predicted_dgcnn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_037_d_predicted_meshsegnet.vtp'))

#s = Mesh(os.path.join(dataurl+'Sample_047_d_predicted_pointnet.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_047_d_predicted_pointnet2.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_047_d_predicted_dgcnn.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_047_d_predicted_meshsegnet.vtp'))

#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_tsgcn.vtp'))
#gac
#s = Mesh(os.path.join(dataurl+'Sample_01_d_predicted_gac.vtp'))
#s = Mesh(os.path.join(dataurl+'Sample_024_d.vtp'))

#mmnet
s = Mesh(os.path.join(dataurl+'Sample_021_d_predicted_mmnet.vtp'))

#s = Mesh(os.path.join(dataurl+'Sample_021_d_predicted_gac.vtp'))
print(s.NCells())
#print('labels len :', len(s.celldata['labels']))
print('labels len :', len(s.celldata['labels']))

#labels_arr = s.celldata['labels']
labels_arr = s.celldata['labels']
print(labels_arr)
#label_color_map = {'0':[254, 254, 254], '1':[252, 78, 3], '2': [252, 128, 3], '3': [252, 186, 3], '4': [252, 235, 3], '5': [231, 252, 3],\
#            '6': [198, 252, 3], '7': [161, 252, 3], '8': [57, 252, 3], '9': [3, 252, 173], '10': [3, 252, 235], '11': [3, 215, 252],\
#            '12': [3, 173, 252], '13': [3, 119, 252], '14': [177, 3, 252] }
    
#label_color_map = {'0':[254, 254, 254], '1': [252, 128, 3], '2': [252, 235, 3], '3': [198, 252, 3]  , '4': [57, 252, 3] , '5': [3, 252, 235] ,\
#            '6': [3, 173, 252] , '7': [177, 3, 252]}
    
# get a nicer color map for 8 class problem
#label_color_map = {'0':[254, 254, 254], '1': [252, 128, 3], '2': [198, 252, 3], '3': [3, 252, 235]  , '4': [177, 3, 252] , '5': [3, 173, 252] ,\
#            '6': [245, 66, 179] , '7': [245, 66, 81]}

#label_color_map = {'0':[254, 254, 254], '1': [252, 128, 3], '2': [252, 235, 3], '3': [198, 252, 3], '4': [3, 252, 235]  , '5': [177, 3, 252] , '6': [3, 173, 252] ,\
#            '7': [245, 66, 179]}

#label_color_map = {'0':[254, 254, 254], '1': [252, 128, 3], '2': [252, 235, 3], '3': [198, 252, 3], '4': [3, 252, 235]  , '5': [3, 173, 252] , '6': [177, 3, 252] ,\
#                   '7': [245, 66, 179]}
    
#--------------------- current one
label_color_map = {'0':[254, 254, 254], '1': [252, 128, 3], '2': [252, 235, 3], '3': [198, 252, 3], '4': [3, 252, 235]  , '5': [3, 173, 252] , '6': [177, 3, 252] ,\
                   '7': [191, 8, 8]}
    
#label_color_map = {'0':[254, 254, 254], '1': [252, 78, 3], '2': [252, 128, 3], '3': [252, 235, 3], '4': [3, 252, 235]  , '5': [3, 173, 252] , '6': [177, 3, 252] ,\
#               '7': [191, 8, 8]}
    
#label_color_map = {'0':[254, 254, 254], '1': [252, 78, 3], '2': [252, 235, 3], '3':  [198, 252, 3], '4': [3, 252, 235]  , '5': [3, 173, 252] , '6': [177, 3, 252] ,\
#               '7': [191, 8, 8]}

#### new color scheme from the paper:
#https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Exploring_Data-Efficient_3D_Scene_Understanding_With_Contrastive_Scene_Contexts_CVPR_2021_paper.pdf
'''
yellow bed - 204, 164, 53
light violet bed - 212, 187, 237
violet headstand - 121, 76, 166
brown bed - 92, 52, 52
dark green wall box - 44, 110, 14
light green floor - 157, 201, 137
hexagon table - 91, 105, 85
blue round table - 45, 111, 173
red round table - 227, 61, 61
pink rectangular table - 227, 61, 174
light pink chair  - 240, 192, 238
light blue chair - 103, 166, 191
indigo wall - 157, 198, 245
yellow rectangular table - 235, 227, 21

what about this ordering
------------------------------------
yellow bed - 204, 164, 53
light green floor - 157, 201, 137
light violet bed - 212, 187, 237
brown bed - 92, 52, 52
blue round table - 45, 111, 173
pink rectangular table - 227, 61, 174
hexagon table - 91, 105, 85

'''

#label_color_map = {'0':[254, 254, 254], '1': [204, 164, 53], '2': [157, 201, 137], '3': [212, 187, 237], '4': [92, 52, 52]  , '5': [45, 111, 173] , '6': [227, 61, 174] ,\
#                   '7': [227, 61, 61]}

  
'''
what about this ordering
------------------------------------
yellow bed - 204, 164, 53
light green floor - 157, 201, 137
light violet bed - 212, 187, 237
brown bed - 92, 52, 52
blue round table - 45, 111, 173
pink rectangular table - 227, 61, 174
dark green wall box - 44, 110, 14

'''
    
#label_color_map = {'0':[254, 254, 254], '1': [204, 164, 53], '2': [157, 201, 137], '3': [212, 187, 237], '4': [92, 52, 52]  , '5': [45, 111, 173] , '6': [227, 61, 174] ,\
#                   '7': [44, 110, 14]}
    
'''    
what about this ordering
------------------------------------
yellow bed - 204, 164, 53
light green floor - 157, 201, 137
light violet bed - 212, 187, 237
brown bed - 92, 52, 52
blue round table - 45, 111, 173
pink rectangular table - 227, 61, 174
dark green wall box - 44, 110, 14
red round table - 227, 61, 61

'''
    
#label_color_map = {'0':[254, 254, 254], '1': [204, 164, 53], '2': [157, 201, 137], '3': [227, 61, 61], '4': [92, 52, 52]  , '5': [45, 111, 173] , '6': [227, 61, 174] ,\
#                   '7': [44, 110, 14]}

    
colorlist = []    
for i in range(len(labels_arr)):
    cur_label = int(labels_arr[i])
    colorlist.append(label_color_map["{:d}".format(cur_label)])
    
print(colorlist)

#colorlist = [0]* s.NCells()
#colorlist = [[100,0,0]]*s.NCells()
#colorlist = [[0,0,0]]*s.NCells()
#colorlist = [[254,254,254]]*s.NCells()
#colorlist = [[154,154,154]]*s.NCells()
#colorlist[:10000] = [0, 0, 0]
#colorlist[:15000] = [100, 100, 100]
#colorlist = [i for i in range(s.NCells())]
#for i in range((s.NCells()//2)):
#    colorlist[i] += 100
    
#colorlist[:s.NCells()//2]
s.cellIndividualColors(colorlist)
#s.cellIndividualColors


s.show()
#
