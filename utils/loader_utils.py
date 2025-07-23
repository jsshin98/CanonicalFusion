import os
import glob
import random
import numpy as np
import cv2
import json

def split_list(data_root=None, filename=None, dataname=None, ratio=0.01, mode='TRAIN'):
    with open(os.path.join(data_root, '%s.txt' % filename), "r") as f:
        lines = f.readlines()

        if mode == 'TRAIN':
            train_set = open(os.path.join(data_root, filename.replace(filename, '%s_TRAIN' % dataname)) + '.txt', "w")
            val_set = open(os.path.join(data_root, filename.replace(filename, '%s_VAL' % dataname)) + '.txt', "w")
        else:
            train_set = open(os.path.join(data_root, filename.replace(filename, 'VAE_%s_TRAIN' % dataname)) + '.txt', "w")
            val_set = open(os.path.join(data_root, filename.replace(filename, 'VAE_%s_VAL' % dataname)) + '.txt', "w")
        for line in lines:
            prob = random.random()
            if prob > ratio:
                train_set.write(line)
            else:
                val_set.write(line)
        train_set.close()
        val_set.close()

def generate_list_diffusion(data_root='/home/somewhere/in/your/pc', filename=None, datalist=None, mode='TRAIN'):
    path2list = os.path.join(data_root, 'list')
    if os.path.isdir(path2list) is False and os.path.isfile(path2list) is False:
        os.mkdir(path2list)

    all_files = os.path.join(path2list, filename) + ".json"
    folder_list = []
    for dataname in datalist:
        folder_list += sorted(glob.glob(os.path.join(data_root, dataname, mode, 'COLOR/DIFFUSE/*')))

    img_list = []
    for folder in folder_list:
        img_list += sorted(glob.glob(os.path.join(folder, '*.png')))

    img_list = [img.replace(data_root, '.') for img in img_list]
    with open(all_files, "w") as f:
        json.dump(img_list, f)

def generate_list_zero123(data_root='/home/somewhere/in/your/pc', filename=None, datalist=None, mode='TRAIN'):
    path2list = os.path.join(data_root, 'list')
    if os.path.isdir(path2list) is False and os.path.isfile(path2list) is False:
        os.mkdir(path2list)

    all_files = os.path.join(path2list, filename) + ".json"
    folder_list = []
    for dataname in datalist:
        folder_list += sorted(glob.glob(os.path.join(data_root, dataname, mode, 'IMGS/*')))

    folder_list = ["/".join(folder.split('/')[-4:]) for folder in folder_list]
    with open(all_files, "w") as f:
        json.dump(folder_list, f)

def generate_list_files(data_root='/home/somewhere/in/your/pc', filename=None, dataname=None, num_test=20, mode='TRAIN'):
    path2list = os.path.join(data_root, 'list')
    if os.path.isdir(path2list) is False and os.path.isfile(path2list) is False:
        os.mkdir(path2list)

    all_files = os.path.join(path2list, filename) + ".txt"
    test_files = os.path.join(path2list, filename.replace('_ALL', '_TEST')) + ".txt"

    with open(all_files, "w") as f, open(test_files, "w") as f_test:
        # folder_list = sorted(glob.glob(os.path.join(data_root, dataname, mode, 'IMGS/*')))
        folder_list = sorted(glob.glob(os.path.join(data_root, dataname, mode, 'COLOR/DIFFUSE/*')))
        test_list = random.sample(folder_list, num_test)
        test_list = [x.split('/')[-1] for x in test_list]
        for folder in sorted(folder_list):
            img_list = glob.glob(folder + '/*.png')
            last_folder = folder.split('/')[-1]

            for k in range(len(img_list)):
                relative_path = img_list[k].replace(data_root, '')
                if last_folder in test_list:
                    f_test.write(relative_path)
                    f_test.write('\n')
                else:
                    f.write(relative_path)
                    f.write('\n')

def merge_train_files(data_root, dataset_list, prefix='MERGED', mode='TRAIN'):
    with open(os.path.join(data_root, prefix + '_TRAIN.txt'), "w") as f_train,\
         open(os.path.join(data_root, prefix + '_VAL.txt'), "w") as f_val:
        for dataset in dataset_list:
            if mode == 'TRAIN':
                cur_train_file = os.path.join(data_root, dataset + '_TRAIN.txt')
                cur_val_file = os.path.join(data_root, dataset + '_VAL.txt')
            else:
                cur_train_file = os.path.join(data_root, mode + '_' + dataset + '_TRAIN.txt')
                cur_val_file = os.path.join(data_root, mode + '_' + dataset + '_VAL.txt')
            with open(cur_train_file, "r") as f_cur:
                lines = f_cur.readlines()
                for line in lines:
                    f_train.write(line)
            with open(cur_val_file, "r") as f_cur:
                lines = f_cur.readlines()
                for line in lines:
                    f_val.write(line)
def generate_test_list_files(data_root='/home/somewhere/in/your/pc', filename=None, dataname=None):
    path2list = os.path.join(data_root, 'list')
    if os.path.isdir(path2list) is False and os.path.isfile(path2list) is False:
        os.mkdir(path2list)

    # gray = np.zeros((5120, 5120))
    # height, width = gray.shape
    # img_noise = np.zeros((height, width), dtype=np.float)
    # for i in range(height):
    #     for a in range(width):
    #         make_noise = np.random.normal()
    #         set_noise = make_noise
    #         img_noise[i][a] = gray[i][a] + set_noise
    # cv2.imwrite('./img_noise.png', img_noise.astype(np.int16))
    folder_list = []
    with open(os.path.join(path2list, filename) + ".txt", "w") as f:
        folder_list.append(glob.glob(os.path.join(data_root, 'IMG/*')))

        for folder in sorted(folder_list[0]):
            img_list = glob.glob(folder + '/*.png')

            for k in range(len(img_list)):
                relative_path = img_list[k].replace(data_root, '')
                # angle = int(relative_path.split('/')[5][:-9])
                # sh = int(relative_path.split('/')[5][-6:-4])
                # if sh == 0 and angle == 0:
                names = list()
                names.append(relative_path)

                save_flag = True
                for item in names:
                    if os.path.isfile(data_root + item) is False:
                        save_flag = False
                        break
                if save_flag:
                    for item in names:
                        f.write(item + ' ')
                    f.write('\n')
# def generate_list_files_all(data_root='/home/somewhere/in/your/pc', filename=None, dataname=None):
#     path2list = os.path.join(data_root, 'list')
#     if os.path.isdir(path2list) is False and os.path.isfile(path2list) is False:
#         os.mkdir(path2list)
#
#     folder_list = []
#     with open(os.path.join(path2list, filename) + ".txt", "w") as f:
#         folder_list.append(glob.glob(os.path.join(data_root, 'RP/COLOR/OPENGL/*' % dataname)))
#         folder_list.append(glob.glob(os.path.join(data_root, 'TH2.0/COLOR/OPENGL/*' % dataname)))
#         # folder_list.append(glob.glob(os.path.join(data_root, 'TH3.0/COLOR/OPENGL/*' % dataname)))
#         folder_list.append(glob.glob(os.path.join(data_root, 'IOYS_T/COLOR/OPENGL/*' % dataname)))
#         # folder_list.append(glob.glob(os.path.join(data_root, 'IOYS/COLOR/OPENGL/*' % dataname)))
#         # folder_list.append(glob.glob(os.path.join(data_root, 'BUFF/COLOR/OPENGL/*' % dataname)))
#
#         for i in len(folder_list):
#             for folder in sorted(folder_list[i]):
#                 img_list = glob.glob(folder + '/*.png')
#
#                 for k in range(len(img_list)):
#                     relative_path = img_list[k].replace(data_root, '')
#                     names = list()
#                     names.append(relative_path)
#
#                     save_flag = True
#                     for item in names:
#                         if os.path.isfile(data_root + item) is False:
#                             save_flag = False
#                             break
#                     if save_flag:
#                         for item in names:
#                             f.write(item + ' ')
#                         f.write('\n')

# generate training related files when this function is called as main.
if __name__ == '__main__':
    data_name = 'ITW' # ['TH2.0', 'TH3.0', 'IOYS_T', 'IOYS', 'RP', 'BUFF', 'FULL', 'ITW']
    data_path = '/media/keti/DATASET/INTHEWILD2'
    filename = '%s_ALL' % data_name
    # generate_list_files(data_root=data_path, filename=filename, dataname=data_name)
    generate_test_list_files(data_root=data_path, filename=filename, dataname=data_name)
    # split_list(data_root=os.path.join(data_path, 'list'), filename=filename, dataname=data_name)