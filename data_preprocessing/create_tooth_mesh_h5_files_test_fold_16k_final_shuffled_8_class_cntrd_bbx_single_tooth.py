# create a dataset
import os
import numpy as np
import pandas as pd
import time
import h5py
from vedo import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#file_path='../data/file_lists_16k/'
#file_path= '../data/file_lists_16k_8_class/'
# for the 15 class we will use this same path, but the contents
# of the actual files would need change in their pathnames
# which will be done inside the 'train', 'val' phase h5 folder
# creation. The test string anyway does not need to have the
# any changes as we will be picking up the unaugmented files for testing
file_path= '../data/file_lists_16k_8_class_shuffled/'
file_list = '{}_list_{}.csv' # use 1-fold as example
#data_path='./data/tooth_data_folds_revised'
#data_path='./data/tooth_data_folds_revised_16k_8_class/'
#data_path='./data/tooth_data_folds_revised_16k_8_class_shuffled/'
# we will change this folder path to make sure the right files are picked
# and the files will contain 8 class, but with centroid and bbox info
#src='all_vtps_combined_flipped_16k_mesh_labeler_labeled'
#data_path='./data/tooth_data_folds_revised_16k_8_class_shuffled_cntrd_bbx/'
data_path='./data/tooth_data_folds_revised_16k_8_class_shuffled_cntrd_bbx_single_tooth/'

#test_src = '../data/all_vtps_combined'
#test_src = '../data/all_vtps_combined_16k_meshes'
#test_src='../data/all_vtps_combined_16k_meshes_for_mesh_labeler_labeled/'
test_src='../data/all_vtps_combined_flipped_16k_8_class/' # although the folder contains flipped data
# our test h5 building code does not take the _flip files

NUM_LABELS = 15

TEETH_CNT = 14

THRESH = 8

chk_str = 'Sample_017_d' # we want to repeat this sample multiple times to create the train set

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        #if type(element) is list:
        #if type(element).__module__ is 'numpy':
        if 'numpy' in type(element).__module__:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    #print('flat_list is: ', flat_list)
    return flat_list


if os.path.exists(data_path) is False:
    os.makedirs(data_path, exist_ok=True)
'''
for fold in range(1, 6):
    for phase_str in ['test']:
        h5_file = phase_str + '{}.h5'
        h5_filename = os.path.join(data_path, h5_file.format(fold))
        if not os.path.exists(h5_filename):
            #print('11111111111111')
            #dataset = h5py.File(h5_filename, 'a')
            dataset = h5py.File(h5_filename, 'w')
        else:
            #print('2222222222222')
            dataset = h5py.File(h5_filename, 'w')

        data_list = pd.read_csv(file_path + file_list.format(phase_str, fold), header=None)
        # the first entry is the column name, let's get rid of that
        data_list = data_list[1:]
        print('{} fold {} data_list len {}'.format(phase_str, fold, len(data_list)))

        for idx in range(len(data_list)):
            idx_mesh = data_list.iloc[idx][0] #vtk file name
            print('idx_mesh num', idx_mesh)
            i_mesh = os.path.join(test_src, 'Sample_0{}_d.vtp'.format(idx_mesh))
            
            mesh = load(i_mesh)
            #labels = mesh.getCellArray('Label').astype('int32').reshape(-1, 1)
            labels = mesh.celldata['labels'].astype('int32').reshape(-1, 1)


            # move mesh to origin
            cells = np.zeros([mesh.NCells(), 9], dtype='float32')
            for i in range(len(cells)):

                cells[i][0], cells[i][1], cells[i][2] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(0)) # don't need to copy
                cells[i][3], cells[i][4], cells[i][5] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(1)) # don't need to copy
                cells[i][6], cells[i][7], cells[i][8] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(2)) # don't need to copy

            mean_cell_centers = mesh.centerOfMass()
            cells[:, 0:3] -= mean_cell_centers[0:3]
            cells[:, 3:6] -= mean_cell_centers[0:3]
            cells[:, 6:9] -= mean_cell_centers[0:3]

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            v1 = np.zeros([mesh.NCells(), 3], dtype='float32')
            v2 = np.zeros([mesh.NCells(), 3], dtype='float32')
            v1[:, 0] = cells[:, 0] - cells[:, 3]
            v1[:, 1] = cells[:, 1] - cells[:, 4]
            v1[:, 2] = cells[:, 2] - cells[:, 5]
            v2[:, 0] = cells[:, 3] - cells[:, 6]
            v2[:, 1] = cells[:, 4] - cells[:, 7]
            v2[:, 2] = cells[:, 5] - cells[:, 8]
            mesh_normals = np.cross(v1, v2) # calculating the normal for point P2
            mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
            mesh_normals[:, 0] /= mesh_normal_length[:]
            mesh_normals[:, 1] /= mesh_normal_length[:]
            mesh_normals[:, 2] /= mesh_normal_length[:]
            #mesh.addCellArray(mesh_normals, 'Normal')
            mesh.celldata['Normal'] = mesh_normals

            v3 = np.zeros([mesh.NCells(), 3], dtype='float32')
            v3[:, 0] = cells[:, 6] - cells[:, 0]
            v3[:, 1] = cells[:, 7] - cells[:, 1]
            v3[:, 2] = cells[:, 8] - cells[:, 2]

            mesh_normals2 = np.cross(v2, v3) # calculating the normal for point P3
            mesh_normal_length2 = np.linalg.norm(mesh_normals2, axis=1)
            mesh_normals2[:, 0] /= mesh_normal_length2[:]
            mesh_normals2[:, 1] /= mesh_normal_length2[:]
            mesh_normals2[:, 2] /= mesh_normal_length2[:]
            #mesh.addCellArray(mesh_normals2, 'Normal2')
            mesh.celldata['Normal2'] = mesh_normals2

            mesh_normals3 = np.cross(v3, v1) # calculating the normal for point P1
            mesh_normal_length3 = np.linalg.norm(mesh_normals3, axis=1)
            mesh_normals3[:, 0] /= mesh_normal_length3[:]
            mesh_normals3[:, 1] /= mesh_normal_length3[:]
            mesh_normals3[:, 2] /= mesh_normal_length3[:]
            #mesh.addCellArray(mesh_normals3, 'Normal3')
            mesh.celldata['Normal3'] = mesh_normals3





            # preprae input and make copies of original datadataset.close()
            # take out these calculations in a separate data preprocessing script
            # this will unnecessarily make the run longer and same is being done repeatedly for every epoch
            # need to calculate the normal at the cell center, calculating the normal using the vectors
            # from any two vertices should be fine as the surface is the same and hence all the three normals
            # that can be constructed would actually be the same
            points = mesh.points().copy()
            points[:, 0:3] -= mean_cell_centers[0:3]
            #normals = mesh.getCellArray('Normal').copy() # need to copy, they use the same memory address
            normals = mesh.celldata['Normal'].copy() # need to copy, they use the same memory address
            normals2 = mesh.celldata['Normal2'].copy() # need to copy, they use the same memory address
            normals3 = mesh.celldata['Normal3'].copy() # need to copy, they use the same memory address
            barycenters = mesh.cellCenters() # don't need to copy
            barycenters -= mean_cell_centers[0:3]

            # creating the normal for the barycenter using P1 and P2
            vp1 = np.zeros([mesh.NCells(), 3], dtype='float32')
            vp2 = np.zeros([mesh.NCells(), 3], dtype='float32')
            vp1[:, 0] = cells[:, 0] - barycenters[:, 0]
            vp1[:, 1] = cells[:, 1] - barycenters[:, 1]
            vp1[:, 2] = cells[:, 2] - barycenters[:, 2]
            vp2[:, 0] = barycenters[:, 0] - cells[:, 3]
            vp2[:, 1] = barycenters[:, 1] - cells[:, 4]
            vp2[:, 2] = barycenters[:, 2] - cells[:, 5]

            mesh_normals4 = np.cross(vp1, vp2) # calculating the normal for point barycenter
            mesh_normal_length4 = np.linalg.norm(mesh_normals4, axis=1)
            mesh_normals4[:, 0] /= mesh_normal_length4[:]
            mesh_normals4[:, 1] /= mesh_normal_length4[:]
            mesh_normals4[:, 2] /= mesh_normal_length4[:]
            #mesh.addCellArray(mesh_normals4, 'Normal4')
            mesh.celldata['Normal4'] = mesh_normals4

            normals4 = mesh.celldata['Normal4'].copy() # need to copy, they use the same memory address


            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)
            # include the other normals
            nmeans2 = normals2.mean(axis=0)
            nstds2 = normals2.std(axis=0)

            nmeans3 = normals3.mean(axis=0)
            nstds3 = normals3.std(axis=0)

            nmeans4 = normals4.mean(axis=0)
            nstds4 = normals4.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]
                # include the other normals
                normals2[:,i] = (normals2[:,i] - nmeans2[i]) / nstds2[i]
                normals3[:,i] = (normals3[:,i] - nmeans3[i]) / nstds3[i]
                normals4[:,i] = (normals4[:,i] - nmeans4[i]) / nstds4[i]

            # normals is at P2, normals3 is at P1 and normals2 is at P3, normals4 is at the barycenter
            X = np.column_stack((cells, barycenters, normals3, normals, normals2, normals4))
            Y = labels

            dataset.create_dataset('{:d}/data'.format(idx), data=X)
            dataset.create_dataset('{:d}/label'.format(idx), data=Y)
            print('{} fold {} writing data for mesh {}'.format(phase_str, fold, idx))
        dataset.close()
'''


# make changes to get the centroid and bbox of the 15 classes
# also calculate the offsets
for fold in range(1, 2):
    for phase_str in ['train', 'val']:
        h5_file = phase_str + '{}.h5'
        h5_filename = os.path.join(data_path, h5_file.format(fold))
        dataset = None
        if not os.path.exists(h5_filename):
            dataset = h5py.File(h5_filename, 'a')
        else:
            dataset = h5py.File(h5_filename, 'w')
        data_list = pd.read_csv(file_path + file_list.format(phase_str, fold), header=None)
        print('{} fold {} data_list len {}'.format(phase_str, fold, len(data_list)))
        #print('data_list: {}'.format(type(data_list)))

        for idx in range(len(data_list)):
            # calculate the filename
            i_mesh = data_list.iloc[idx][0] #vtk file name
            print('i_mesh: ', i_mesh)
            if 'train' in phase_str and chk_str not in i_mesh:
                break
            #raise ValueError("Exit!")
            # change the path to point to the 15 class folder
            i_mesh = i_mesh.replace('8_class', '15_class') #vtk file name
            #print('i_mesh: ', i_mesh)
            mesh = load(i_mesh)
            #labels = mesh.getCellArray('Label').astype('int32').reshape(-1, 1)
            labels = mesh.celldata['labels'].astype('int32').reshape(-1, 1)

            #create one-hot map
        #        label_map = np.zeros([mesh.cells.shape[0], self.num_classes], dtype='int32')
        #        label_map = np.eye(self.num_classes)[labels]
        #        label_map = label_map.reshape([len(labels), self.num_classes])

            # move mesh to origin
            cells = np.zeros([mesh.NCells(), 9], dtype='float32')
            for i in range(len(cells)):
                #cells[i][0], cells[i][1], cells[i][2] = mesh._polydata.GetPoint(mesh._polydata.GetCell(i).GetPointId(0)) # don't need to copy
                #cells[i][3], cells[i][4], cells[i][5] = mesh._polydata.GetPoint(mesh._polydata.GetCell(i).GetPointId(1)) # don't need to copy
                #cells[i][6], cells[i][7], cells[i][8] = mesh._polydata.GetPoint(mesh._polydata.GetCell(i).GetPointId(2)) # don't need to copy

                cells[i][0], cells[i][1], cells[i][2] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(0)) # don't need to copy
                cells[i][3], cells[i][4], cells[i][5] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(1)) # don't need to copy
                cells[i][6], cells[i][7], cells[i][8] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(2)) # don't need to copy

            mean_cell_centers = mesh.centerOfMass()
            cells[:, 0:3] -= mean_cell_centers[0:3]
            cells[:, 3:6] -= mean_cell_centers[0:3]
            cells[:, 6:9] -= mean_cell_centers[0:3]

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            v1 = np.zeros([mesh.NCells(), 3], dtype='float32')
            v2 = np.zeros([mesh.NCells(), 3], dtype='float32')
            v1[:, 0] = cells[:, 0] - cells[:, 3]
            v1[:, 1] = cells[:, 1] - cells[:, 4]
            v1[:, 2] = cells[:, 2] - cells[:, 5]
            v2[:, 0] = cells[:, 3] - cells[:, 6]
            v2[:, 1] = cells[:, 4] - cells[:, 7]
            v2[:, 2] = cells[:, 5] - cells[:, 8]
            mesh_normals = np.cross(v1, v2) # calculating the normal for point P2
            mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
            mesh_normals[:, 0] /= mesh_normal_length[:]
            mesh_normals[:, 1] /= mesh_normal_length[:]
            mesh_normals[:, 2] /= mesh_normal_length[:]
            #mesh.addCellArray(mesh_normals, 'Normal')
            mesh.celldata['Normal'] = mesh_normals

            v3 = np.zeros([mesh.NCells(), 3], dtype='float32')
            v3[:, 0] = cells[:, 6] - cells[:, 0]
            v3[:, 1] = cells[:, 7] - cells[:, 1]
            v3[:, 2] = cells[:, 8] - cells[:, 2]

            mesh_normals2 = np.cross(v2, v3) # calculating the normal for point P3
            mesh_normal_length2 = np.linalg.norm(mesh_normals2, axis=1)
            mesh_normals2[:, 0] /= mesh_normal_length2[:]
            mesh_normals2[:, 1] /= mesh_normal_length2[:]
            mesh_normals2[:, 2] /= mesh_normal_length2[:]
            #mesh.addCellArray(mesh_normals2, 'Normal2')
            mesh.celldata['Normal2'] = mesh_normals2

            mesh_normals3 = np.cross(v3, v1) # calculating the normal for point P1
            mesh_normal_length3 = np.linalg.norm(mesh_normals3, axis=1)
            mesh_normals3[:, 0] /= mesh_normal_length3[:]
            mesh_normals3[:, 1] /= mesh_normal_length3[:]
            mesh_normals3[:, 2] /= mesh_normal_length3[:]
            #mesh.addCellArray(mesh_normals3, 'Normal3')
            mesh.celldata['Normal3'] = mesh_normals3



            # preprae input and make copies of original datadataset.close()
            # take out these calculations in a separate data preprocessing script
            # this will unnecessarily make the run longer and same is being done repeatedly for every epoch
            # need to calculate the normal at the cell center, calculating the normal using the vectors
            # from any two vertices should be fine as the surface is the same and hence all the three normals
            # that can be constructed would actually be the same
            points = mesh.points().copy()
            points[:, 0:3] -= mean_cell_centers[0:3]
            #normals = mesh.getCellArray('Normal').copy() # need to copy, they use the same memory address
            normals = mesh.celldata['Normal'].copy() # need to copy, they use the same memory address
            normals2 = mesh.celldata['Normal2'].copy() # need to copy, they use the same memory address
            normals3 = mesh.celldata['Normal3'].copy() # need to copy, they use the same memory address
            barycenters = mesh.cellCenters() # don't need to copy
            barycenters -= mean_cell_centers[0:3]

            # creating the normal for the barycenter using P1 and P2
            vp1 = np.zeros([mesh.NCells(), 3], dtype='float32')
            vp2 = np.zeros([mesh.NCells(), 3], dtype='float32')
            vp1[:, 0] = cells[:, 0] - barycenters[:, 0]
            vp1[:, 1] = cells[:, 1] - barycenters[:, 1]
            vp1[:, 2] = cells[:, 2] - barycenters[:, 2]
            vp2[:, 0] = barycenters[:, 0] - cells[:, 3]
            vp2[:, 1] = barycenters[:, 1] - cells[:, 4]
            vp2[:, 2] = barycenters[:, 2] - cells[:, 5]

            mesh_normals4 = np.cross(vp1, vp2) # calculating the normal for point barycenter
            mesh_normal_length4 = np.linalg.norm(mesh_normals4, axis=1)
            mesh_normals4[:, 0] /= mesh_normal_length4[:]
            mesh_normals4[:, 1] /= mesh_normal_length4[:]
            mesh_normals4[:, 2] /= mesh_normal_length4[:]
            #mesh.addCellArray(mesh_normals4, 'Normal4')
            mesh.celldata['Normal4'] = mesh_normals4

            normals4 = mesh.celldata['Normal4'].copy() # need to copy, they use the same memory address


            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)
            # include the other normals
            nmeans2 = normals2.mean(axis=0)
            nstds2 = normals2.std(axis=0)

            nmeans3 = normals3.mean(axis=0)
            nstds3 = normals3.std(axis=0)

            nmeans4 = normals4.mean(axis=0)
            nstds4 = normals4.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]
                # include the other normals
                normals2[:,i] = (normals2[:,i] - nmeans2[i]) / nstds2[i]
                normals3[:,i] = (normals3[:,i] - nmeans3[i]) / nstds3[i]
                normals4[:,i] = (normals4[:,i] - nmeans4[i]) / nstds4[i]


            '''
            # calculate number of valid cells (tooth instead of gingiva)
            positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
            negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

            num_positive = len(positive_idx) # number of selected tooth cells

            if num_positive > self.patch_size: # all positive_idx in this patch
                positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
                selected_idx = positive_selected_idx
            else:   # patch contains all positive_idx and some negative_idx
                num_negative = self.patch_size - num_positive # number of selected gingiva cells
                positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
                negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
                selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))
            '''
            # get centroid for every label
            #X_lbl = np.zeros([self.patch_size, 15], dtype='float32')
            #Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
            offsets_list = []
            cur_idx_list = []
            cur_lbl_ctr_list = []
            cur_bbx_min_list = []
            cur_bbx_max_list = []
            for i in range(NUM_LABELS):
                cur_idx = np.argwhere(labels==i)[:, 0] # gingiva idx
                #print('len of cur_idx: ', len(cur_idx))
                #print('len of barycenters: ', len(barycenters))
                #print('len of barycenters[cur_idx]: ', len(barycenters[cur_idx, :]))
                cur_lbl_barycenters = barycenters[cur_idx, :]
                #print('len of cur_lbl_barycenters: ', len(cur_lbl_barycenters))
                #print('type of cur_lbl_barycenters: ', type(cur_lbl_barycenters))
                #print('barycenters[0]: ', barycenters[0])
                #print('cur_lbl_barycenters[0]: ', cur_lbl_barycenters[0])
                #cur_lbl_center = np.mean(cur_lbl_barycenters, axis=1)
                # find the center of the pointcloud of that particular label
                cur_lbl_center = np.mean(cur_lbl_barycenters, axis=0)
                cur_lbl_ctr_list.append(cur_lbl_center)
                #print('cur_lbl_ctr list: ', cur_lbl_ctr_list)
                #print('cur_lbl_center: ', cur_lbl_center)
                #print('len cur_lbl_center: ',len(cur_lbl_center))
                # find the offset
                offsets = np.sqrt(np.sum(np.square(cur_lbl_barycenters - cur_lbl_center), axis=1))
                #offsets_list.append(offsets)
                #cur_idx_list.append(cur_idx)
                offsets_list.append(np.array(offsets))
                cur_idx_list.append(np.array(cur_idx))
                #print('cur_idx_list :', cur_idx_list)
                #print('offsets: ', offsets)
                #print('len offsets: ', len(offsets))
                # x_cntrd would save the centroids
                #X_cntrd = np.zeros([len(cur_idx), 3], dtype='float32')

                # create the bbox
                cur_lbl_cells = cells[cur_idx, :]
                #print('cur_lbl_cells[0]: ', cur_lbl_cells[0])
                # save the x values, y values and z values for
                # all the cells separately
                xs = []
                xs.append(cur_lbl_cells[:, 0])
                xs.append(cur_lbl_cells[:, 3])
                xs.append(cur_lbl_cells[:, 6])
                #print(xs)
                #print('xs[0]: ', xs[0])
                #print('len of xs[0]: ', len(xs[0]))

                xs_arr = np.array(xs).flatten()
                #print('xs_arr: ', xs_arr)
                #print('len of xs_arr: ', len(xs_arr))
                x_min = y_min = z_min = x_max = y_max = z_max = 0.0
                if (len(xs_arr)) != 0:
                    #print('xs_arr is empty. Let us check the offsets and indices')
                    #print('cur_idx: ', cur_idx)
                    #print('len of cur_idx: ', len(cur_idx))
                    #raise ValueError("Exit!")
                    x_min = np.min(xs_arr)
                    x_max = np.max(xs_arr)

                    #print('x_min: {}, x_max: {}'.format(x_min, x_max))

                    ys = []
                    ys.append(cur_lbl_cells[1])
                    ys.append(cur_lbl_cells[4])
                    ys.append(cur_lbl_cells[7])
                    #print(ys)
                    #print('ys[0]: ', ys[0])
                    #print('len of ys[0]: ', len(ys[0]))

                    ys_arr = np.array(ys).flatten()
                    #print('ys_arr: ', ys_arr)
                    #print('len of ys_arr: ', len(ys_arr))
                    y_min = np.min(ys_arr)
                    y_max = np.max(ys_arr)
                    #print('y_min: {}, y_max: {}'.format(y_min, y_max))

                    zs = []
                    zs.append(cur_lbl_cells[2])
                    zs.append(cur_lbl_cells[5])
                    zs.append(cur_lbl_cells[8])
                    #print(xs)
                    #print('zs[0]: ', zs[0])
                    #print('len of zs[0]: ', len(zs[0]))

                    zs_arr = np.array(zs).flatten()
                    #print('zs_arr: ', zs_arr)
                    #print('len of zs_arr: ', len(zs_arr))
                    z_min = np.min(zs_arr)
                    z_max = np.max(zs_arr)
                    #print('z_min: {}, z_max: {}'.format(z_min, z_max))

                '''
                cur_bbx_min_list.append(x_min)
                cur_bbx_min_list.append(y_min)
                cur_bbx_min_list.append(z_min)
                cur_bbx_max_list.append(x_max)
                cur_bbx_max_list.append(y_max)
                cur_bbx_max_list.append(z_max)
                '''
                cur_bbx_min_list.append(np.array([x_min, y_min, z_min])) # does not work this way
                cur_bbx_max_list.append(np.array([x_max, y_max, z_max])) # does not work this way

                #print('cur_bbx_min_list: ', cur_bbx_min_list)
                #print('cur_bbx_max_list: ', cur_bbx_max_list)
                #print('len cur_bbx_min_list: ', len(cur_bbx_min_list))
                #print('cur_bbx_max_list: ', len(cur_bbx_max_list))

                #print('cells type: ', type(cells))
                #print('cells[0]: ', cells[0])
                #print('cells: ', cells)
                #print('len cells: ', len(cells))
                #print('type barycenters: ', type(barycenters))
                #print('len barycenters: ', len(barycenters))
                #print('barycenters[0]: ', barycenters[0])
                

            #offsets_list = []
            #cur_idx_list = []
            #cur_lbl_ctr_list = []
            #cur_bbx_min_list = []
            #cur_bbx_max_list = []
            '''
            print('len offsets_list: ', len(offsets_list))
            print('len cur_idx_list: ', len(cur_idx_list))
            print('len cur_lbl_ctr_list: ', len(cur_lbl_ctr_list))
            print('len cur_bbx_min_list: ', len(cur_bbx_min_list))
            print('len cur_bbx_max_list: ', len(cur_bbx_max_list))
            print('offsets_list[0]: ', offsets_list[0])
            print('cur_idx_list[0]: ', cur_idx_list[0])
            print('cur_lbl_ctr_list[0]: ', cur_lbl_ctr_list[0])
            print('cur_bbx_min_list[0]: ', cur_bbx_min_list[0])
            print('cur_bbx_max_list[0]: ', cur_bbx_max_list[0])
            print('type of offsets_list[0]: ', type(offsets_list[0]))
            print('type of cur_idx_list[0]: ', type(cur_idx_list[0]))
            print('type of cur_lbl_ctr_list[0]: ', type(cur_lbl_ctr_list[0]))
            print('type of cur_bbx_min_list[0]: ', type(cur_bbx_min_list[0]))
            print('type of cur_bbx_max_list[0]: ', type(cur_bbx_max_list[0]))
            #print('cur_idx_list[1]: ', cur_idx_list[1])
            print('\n--------- after change ----------\n')
            #offsets = np.array(offsets_list).flatten()
            #cur_idx = np.array(cur_idx_list).flatten()
            #offsets = flatten_list(offsets_list)
            #cur_idx = flatten_list(cur_idx_list)
            #cntrds = []
            #print('len of cells: ', len(cells))
            #print('type of cells: ', type(cells))
            #print('type of cells[0]: ', type(cells[0]))
            #print('len offsets: ', len(offsets))
            #print('len cur_idx: ', len(cur_idx))
            '''

            #offsets_list = []
            #cur_idx_list = []
            #cur_lbl_ctr_list = []
            #cur_bbx_min_list = []
            #cur_bbx_max_list = []
            # arrange the centroids, offsets, bbx_min and bbx_max`
            centroids = np.zeros([len(cells), 3], dtype='float64')
            offsets = np.zeros([len(cells), 1], dtype='float64')
            bbx_mins = np.zeros([len(cells), 3], dtype='float64')
            bbx_maxs = np.zeros([len(cells), 3], dtype='float64')

            #print('centroids :', centroids)
            #print('centroids shape:', centroids.shape)
            #for i in range(len(cur_idx)):
            for i in range(len(cur_idx_list)):
                cidx = cur_idx_list[i]
                #print('cidx is :', cidx)
                centroids[cidx, :] = cur_lbl_ctr_list[i]
                #offsets[cidx, :] = offsets_list[i, :]
                offsets[cidx] = offsets_list[i].reshape(-1, 1)
                bbx_mins[cidx, :] = cur_bbx_min_list[i]
                bbx_maxs[cidx, :] = cur_bbx_max_list[i]

            #print('centroids :', centroids)
            #print('offsets:', offsets)
            #print('bbx_mins :', bbx_mins)
            #print('bbx_maxs:', bbx_maxs)
            #print('centroids shape:', centroids.shape)


            '''
            selected_idx = np.sort(selected_idx, axis=None)

            X_train[:] = X[selected_idx, :15]
            Y_train[:] = Y[selected_idx, :]
            '''

            #print('before modifying max in labels, min in labels: ', np.max(labels), np.min(labels))
            #print('before modifying max in labels: {}, min in labels: {}, argmax: {}, argmin: {}'.format(np.max(labels), np.min(labels), np.argmax(labels), np.argmin(labels)))
            for i in range(len(labels)):
                if labels[i] >= THRESH:
                    labels[i] = int(TEETH_CNT + 1 - labels[i])
            #print('after modifying max in labels, min in labels: ', np.max(labels), np.min(labels))
            #print('after modifying max in labels: {}, min in labels: {}, argmax: {}, argmin: {}'.format(np.max(labels), np.min(labels), np.argmax(labels), np.argmin(labels)))
            #raise ValueError("Exit!")

            # normals is at P2, normals3 is at P1 and normals2 is at P3, normals4 is at the barycenter
            #X = np.column_stack((cells, barycenters, normals3, normals, normals2, normals4))
            # add the centroids, offsets, bbx_mins and bbx_maxs
            X = np.column_stack((cells, barycenters, normals3, normals, normals2, normals4, centroids, offsets, bbx_mins, bbx_maxs))
            Y = labels

            dataset.create_dataset('{:d}/data'.format(idx), data=X)
            dataset.create_dataset('{:d}/label'.format(idx), data=Y)
            print('{} fold {} writing data for mesh {}'.format(phase_str, fold, idx))
        dataset.close()
