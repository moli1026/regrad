import os
import shutil
import pdb
import json
import numpy as np 
import random
import xml.etree.ElementTree as ET
from PIL import Image



# dirpath = '/data1/cxg7/dataset/VMRD_3d/train'
# datapath = os.listdir(dirpath)
# for scene in datapath:
#     data = os.path.join(dirpath, scene)
#     if len(os.listdir(data)) < 10:
#         shutil.move(data, '/data1/cxg7/dataset/delfile')

dirpath = '/data1/cxg7/dataset/VMRD_3d/train'
# datapath = os.listdir(dirpath)


def VMRD_v3_2_v2():
    image_path = '/data1/cxg7/dataset/vmrd_v3/JPEGImages'
    anno_path = '/data1/cxg7/dataset/vmrd_v3/Annotations'
    Imageset_path = '/data1/cxg7/dataset/vmrd_v3/ImageSets'
    Grasp_path = '/data1/cxg7/dataset/vmrd_v3/Grasps'
    # label_path = '/data1/cxg7/dataset/vmrd_V3/labels'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)
    if not os.path.exists(Imageset_path):
        os.makedirs(Imageset_path)
    if not os.path.exists(Grasp_path):
        os.makedirs(Grasp_path)
    # if not os.path.exists(label_path):
    #     os.makedirs(label_path)
    for scene in sorted(os.listdir(dirpath)):
    # for scene in os.listdir(dirpath):
        data = os.path.join(dirpath, scene)
        for view in range(1,10):
            view = str(view)
            #image
            image = data + '/' + view + '/' + 'rgb.jpg'
            shutil.copy(image, image_path)
            os.rename(image_path + '/' + 'rgb.jpg', image_path +'/' + '{}_{}.jpg'.format(scene, view))
            # pdb.set_trace()

def f():
    image_path = '/data1/cxg7/dataset/VMRD_V3/JPEGImages'
    train_path = '/data1/cxg7/dataset/VMRD_V3/ImageSets/train.txt'
    test_path = '/data1/cxg7/dataset/VMRD_V3/ImageSets/test.txt'
    image = os.listdir(image_path)
    image_num = len(image)
    train_num = np.random.randint(0, image_num, int(0.8*image_num))
    with open(train_path, 'w') as f:
        for i in (train_num):
            f.write(os.path.join(image_path, image[i]))
            f.write('\n')
    test_num = np.array([i for i in range(image_num) if i not in train_num])
    with open(test_path, 'w') as f:
        for i in test_num:
            f.write(os.path.join(image_path, image[i]))
            f.write('\n')

def f2():
    image_path = '/data/cxg12/Code/VMRN_FRCNN/data/REGRAD/train'
    train_path = '/data/cxg12/Code/VMRN_FRCNN/data/REGRAD/train_v1.txt'
    test_path = '/data/cxg12/Code/VMRN_FRCNN/data/REGRAD/test.txt'
    # if not (os.path.exists(train_path) and os.path.exists(test_path)):
    #     os.makedirs(train_path)
    #     os.makedirs(train_path)
    image = os.listdir(image_path)
    image_num = 20000
    train_num = random.sample(range(0,image_num), int(image_num*0.8))
    with open(train_path, 'w') as f:
        for i in (train_num):
            f.write(image[i].split('.')[0])
            f.write('\n')
    test_num = np.array([i for i in range(image_num) if i not in train_num])
    with open(test_path, 'w') as f:
        for i in test_num:
            f.write(image[i].split('.')[0])
            f.write('\n')


def get_class_num():
    names =[]
    for scene in sorted(os.listdir(dirpath)):
        infofile_path = os.path.join(dirpath, scene, 'final_state.json')
        infofile = json.load(open(infofile_path))
        for model in infofile:
            name = model['name'].split('-')[0]
            names.append(name)
    print(set(names))

def move_info_file():
    filepath = '/data1/cxg7/dataset/VMRD_3d/train'
    outdir = '/data1/cxg7/dataset/vmrd_v3/Annotations'
    for scene in sorted(os.listdir(filepath)):
        for view in range(1,10):
            infopath = filepath + '/' + str(scene) + '/' + str(view) + '/' + 'info.json'
            shutil.copy(infopath, outdir)
            os.rename(os.path.join(outdir, 'info.json'),os.path.join(outdir, '{}_{}.json'.format(scene, view)))

def move_grasp():
    graspfile = '/data1/cxg7/dataset/vmrd_v3/grasp_origin'
    for scene in sorted(os.listdir(graspfile)):
        for view in range(1,10):
            with open('/data1/cxg7/dataset/vmrd_v3/Grasps/{}_{}.txt'.format(scene, view), 'w') as f:
                graspose = os.path.join(graspfile, scene, '{}.json'.format(view))
                if not os.path.exists(graspose):
                    print('!!!!!!!!!!!!!!!!')
                    print(scene, view)
                    print('去20重新生成！！！！！！')
                    print('!!!!!!!!!!!!!!!!')
                    continue
                data = json.load(open(graspose))
                for obj in data:
                    grasp_box = np.zeros((1,6))
                    grasp_box[:,0], grasp_box[:,1], grasp_box[:,2], grasp_box[:,3], grasp_box[:,4] = obj[1][0][0], obj[1][0][1], obj[1][1][0], obj[1][1][1], obj[1][2]
                    grasp_box[:,5] = int(obj[0])
                    np.savetxt(f, grasp_box,delimiter=" ")
                # pdb.set_trace()

def test_grasp_prepare():
    data_path = '/data1/cxg7/dataset/vmrd_v2/Annotations_origin'
    save_image_ann = '/data1/cxg7/dataset/train_test_t/Annotations'
    save_grasp_ann = '/data1/cxg7/dataset/train_test_t/Grasps'
    image_path = '/data1/cxg7/dataset/train_test_t/JPEGImages'
    for i, filename in enumerate(sorted(os.listdir(data_path))):
        image_ann = {}
        if filename == '00198.xml':
            pdb.set_trace()
        file = os.path.join(data_path, filename)
        tree = ET.parse(file)
        objs = tree.findall('object')
        num_objs = len(objs)
        grasp_ann = np.zeros([num_objs,6])
        image = Image.open(os.path.join(image_path, filename.replace('xml', 'jpg')))
        H,W = image.size[0], image.size[1]
        box = [1,1, H, W]
        image_ann['id'] = i
        image_ann['bbox'] = box
        image_ann['seg_areas'] = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        image_ann_file = [image_ann]
        # for i in range(num_objs):
        #     grasp_ann[i,0], grasp_ann[i,1], grasp_ann[i,2], grasp_ann[i,3], grasp_ann[i,4], grasp_ann[i,5] = \
        #         box[0], box[1], box[2], box[3], int(0), int(-1)
        #     with open(os.path.join(save_grasp_ann, filename.replace('xml', 'txt')), 'w') as f:
        #         np.savetxt(f, grasp_ann, delimiter=" ")

        with open(os.path.join(save_image_ann, filename.replace('xml', 'json')), 'w') as f:
            json.dump(image_ann_file, f)


def vmrd_for_HDA():
    datadir = '/data1/cxg7/dataset/train_test_s/JPEGImages'
    datadir_t = '/data1/cxg7/dataset/train_test_t/JPEGImages'
    savedir_train = '/data1/cxg7/dataset/train_test_s/ImageSets/trainval.txt'
    savedir_test = '/data1/cxg7/dataset/train_test_s/ImageSets/test.txt'
    savedir_train_t = '/data1/cxg7/dataset/train_test_t/ImageSets/trainval.txt'
    savedir_test_t = '/data1/cxg7/dataset/train_test_t/ImageSets/test.txt'
    num = len(os.listdir(datadir))
    num_t = len(os.listdir(datadir_t))
    num_train = random.sample(range(0,num), int(num*0.8))
    num_train_t = random.sample(range(0,num_t), int(num_t*0.8))
    num_test = [i for i in range(num) if i not in num_train]
    num_test_t = [i for i in range(num_t) if i not in num_train_t]
    image = os.listdir(datadir)
    image_t = os.listdir(datadir_t)
    with open(savedir_train, 'w') as f:
        for i in num_train:
            f.write(os.path.join(datadir, image[i]))
            f.write('\n')
    with open(savedir_test, 'w') as f:
        for i in num_test:
            f.write(os.path.join(datadir, image[i]))
            f.write('\n')
    with open(savedir_train_t, 'w') as f:
        for i in num_train_t:
            f.write(os.path.join(datadir_t, image_t[i]))
            f.write('\n')
    with open(savedir_test_t, 'w') as f:
        for i in num_test_t:
            f.write(os.path.join(datadir_t, image_t[i]))
            f.write('\n')

def get_class_num_test():
    datadir = '/data1/cxg7/dataset/train_test_s/Annotations'
    names =[]
    for scene in sorted(os.listdir(datadir)):
        infofile_path = os.path.join(datadir, scene)
        infofile = json.load(open(infofile_path))
        for model in infofile:
            name = model['model_name'].split('-')[0]
            names.append(name)
    print(len(set(names)))

# def get_label_file_for_HDA():
    # datadir = '/data1/cxg7/dataset/train_test_s/Annotations'

def f3():
    datadir = '/data1/ydy/cleaned_3DVMRD/train'
    scenelist = os.listdir(datadir)
    with open('/data1/cxg7/dataset/vmrd_v3/Imageset/train.txt', 'w') as f:
        for scene in scenelist:
            for view in range(9):
                view = str(view+1)
                f.write(os.path.join(datadir, scene, view, 'rgb.jpg'))
                f.write('\n')
def f4():
    datadir = '/data1/cxg7/dataset/vmrd_v3'
    scenelist = os.listdir(datadir)
    for scene in scenelist:
        view_list = os.listdir(os.path.join(datadir, scene))
        if len(view_list) < 5:
            print(scene)
        


########
#results
# ['vessel', 'rocket', 'motorcycle', 'telephone', 'car', 'washer', 'guitar', 
# 'bag', 'bathtub', 'loudspeaker', 'bookshelf', 'microwave', 'bed', 'file', 'ashcan', 
# 'remote control', 'bench', 'bottle', 'computer keyboard', 'cabinet', 'airplane', 'printer', 
# 'bus', 'cellular telephone', 'mug', 'dishwasher', 'pistol', 'camera', 'microphone', 'pot',
#  'earphone', 'birdhouse', 'table', 'mailbox', 'helmet', 'can', 'basket', 'cap']
    
# {'cabinet', 'bag', 'rocket', 'mailbox', 'basket', 'airplane', 'remote control', 'mug', 
# 'motorcycle', 'vessel', 'birdhouse', 'pot', 'guitar', 'bathtub', 'bookshelf', 'helmet', 
# 'camera', 'earphone', 'telephone', 'bed', 'loudspeaker', 'printer', 'cellular telephone', 
# 'bottle', 'table', 'bench', 'microwave', 'dishwasher', 'ashcan', 'bus', 'car', 'washer', 
# 'file', 'computer keyboard', 'microphone', 'can', 'cap', 'pistol'}

# {'cellular telephone', 'remote control', 'earphone', 'mug', 'basket', 'bottle', 
# 'bathtub', 'pot', 'loudspeaker', 'washer', 'bag', 'can', 'mailbox', 'bench', 'car', 'cap', 
# 'bookshelf', 'helmet', 'microphone', 'airplane', 'bus', 'table', 'cabinet', 'telephone', 'motorcycle'}


           


if __name__ == '__main__':
    # VMRD_v3_2_v2()
    f2()
    # get_class_num()
    # move_info_file()
    # move_grasp()
    # test_grasp_prepare()
    # vmrd_for_HDA()
    # get_class_num_test()

    # data = np.loadtxt(open('/data1/cxg7/dataset/train_test_t/Grasps/00022.txt', 'r'))
    # print(data.shape)



