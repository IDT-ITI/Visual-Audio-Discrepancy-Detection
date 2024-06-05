import gc
import os
import csv
import time
import numpy as np

from glob import glob
import pickle

from sklearn.preprocessing import StandardScaler

import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

SEED = 86
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

out_folder = 'models'
epochs = 30
save_epochs = [10, 20, 30]

bs = 32
lr = 0.001
scheduler_min = 0.01
dropout_rate = 0.5
l2_reg = 0.01
l1_reg = 0
sgd_momentum = 0.0  # 0.9 is slightly worse than 0.0
init_xavier = True
print_interval = 5000

APPROACHES = [
    {  # best - Late self-attention scaled
        'name': 'va-ls-10-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 0,
        'mod_self_att': 0,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # baseline - no self-attention (NS) scaled
        'name': 'va-ns-10-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 0,
        'mod_self_att': 0,
        'late_self_att': 0,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # Late self-attention - no data augm. - scaled
        'name': 'va-ls-10-nda-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 0,
        'mod_self_att': 0,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 0,
        'classes': 10,
        'scale': 1
    },
    {  # Single FC layer - scaled
        'name': 'va-ls-10-nxfc-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 0,
        'mod_self_att': 0,
        'late_self_att': 1,
        'final_x_fc': 0,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # visual only - 10 classes - scaled
        'name': 'v-ls-10-s',
        'use': ['vit', 'clip', 'resnet50'],
        'early_self_att': 0,
        'mod_self_att': 0,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # audio only - 10 classes - scaled
        'name': 'a-ls-10-s',
        'use': ['openl3', 'pann', 'iov'],
        'early_self_att': 0,
        'mod_self_att': 0,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # audio only - 3 classes - scaled
        'name': 'a-ls-3-s',
        'use': ['openl3', 'pann', 'iov'],
        'early_self_att': 0,
        'mod_self_att': 0,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 3,
        'scale': 1
    },
    {  # visual only - 3 classes - scaled
        'name': 'v-ls-3-s',
        'use': ['vit', 'clip', 'resnet50'],
        'early_self_att': 0,
        'mod_self_att': 0,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 3,
        'scale': 1
    },
    {  # Early self-attention - scaled
        'name': 'va-es-10-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 1,
        'mod_self_att': 0,
        'late_self_att': 0,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # per-modality (MS) self-attention - scaled
        'name': 'va-hs-10-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 0,
        'mod_self_att': 1,
        'late_self_att': 0,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # ES + LS - scaled
        'name': 'va-es+ls-10-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 1,
        'mod_self_att': 0,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # MS + LS
        'name': 'va-hs+ls-10-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 0,
        'mod_self_att': 1,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # ES + MS - scaled
        'name': 'va-hs+es-10-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 1,
        'mod_self_att': 1,
        'late_self_att': 0,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    },
    {  # ES + MS + LS - scaled
        'name': 'va-es+hs+ls-10-s',
        'use': ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov'],
        'early_self_att': 1,
        'mod_self_att': 1,
        'late_self_att': 1,
        'final_x_fc': 1,
        'data_aug': 1,
        'classes': 10,
        'scale': 1
    }
]

NETS_ALL = ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov']

SHAPES = {
    'vit': 1000,
    'clip': 1024,
    'resnet50': 2048,
    'openl3': 512,
    'pann': 512,
    'iov': 256
}

MAPPING3 = {
    "airport": "indoor",
    "bus": "vehicle",
    "shopping_mall": "indoor",
    "street_pedestrian": "outdoor",
    "street_traffic": "outdoor",
    "metro_station": "indoor",
    "park": "outdoor",
    "public_square": "outdoor",
    "metro": "vehicle",
    "tram": "vehicle",
}

MAPPING10 = {
    "airport": "airport",
    "bus": "bus",
    "shopping_mall": "shopping_mall",
    "street_pedestrian": "street_pedestrian",
    "street_traffic": "street_traffic",
    "metro_station": "metro_station",
    "park": "park",
    "public_square": "public_square",
    "metro": "metro",
    "tram": "tram",
}


def csv2dict(csv_filename):
    with open(csv_filename, 'r') as f:
        dict_reader = csv.DictReader(f, delimiter='\t')
        list_of_dict = list(dict_reader)
    return list_of_dict


class NetCombFinal(nn.Module):
    def __init__(self,
                 v_dim1, v_dim2, v_dim3, a_dim1, a_dim2, a_dim3,
                 early_self_att=0, mod_self_att=0, late_self_att=0,
                 drop_rate=0.2, final_x_fc=0, init_xav=True,
                 out_dim=10, batch_norm=0, no_of_heads=1):
        super(NetCombFinal, self).__init__()

        self.v_dim1 = v_dim1
        self.v_dim2 = v_dim2
        self.v_dim3 = v_dim3
        self.a_dim1 = a_dim1
        self.a_dim2 = a_dim2
        self.a_dim3 = a_dim3

        self.batch_norm = batch_norm
        self.early_self_att = early_self_att
        self.mod_self_att = mod_self_att
        self.late_self_att = late_self_att
        self.final_x_fc = final_x_fc

        if batch_norm:
            if v_dim1:
                self.bn_v1 = nn.BatchNorm1d(v_dim1)
            if v_dim2:
                self.bn_v2 = nn.BatchNorm1d(v_dim2)
            if v_dim3:
                self.bn_v3 = nn.BatchNorm1d(v_dim3)

            if a_dim1:
                self.bn_a1 = nn.BatchNorm1d(a_dim1)
            if a_dim2:
                self.bn_a2 = nn.BatchNorm1d(a_dim2)
            if a_dim3:
                self.bn_a3 = nn.BatchNorm1d(a_dim3)

        if early_self_att:
            if v_dim1:
                self.sa_v1 = nn.MultiheadAttention(v_dim1, no_of_heads, batch_first=True)
            if v_dim2:
                self.sa_v2 = nn.MultiheadAttention(v_dim2, no_of_heads, batch_first=True)
            if v_dim3:
                self.sa_v3 = nn.MultiheadAttention(v_dim3, no_of_heads, batch_first=True)

            if a_dim1:
                self.sa_a1 = nn.MultiheadAttention(a_dim1, no_of_heads, batch_first=True)
            if a_dim2:
                self.sa_a2 = nn.MultiheadAttention(a_dim2, no_of_heads, batch_first=True)
            if a_dim3:
                self.sa_a3 = nn.MultiheadAttention(a_dim3, no_of_heads, batch_first=True)

        f_dim = v_dim1 + v_dim2 + v_dim3 + a_dim1 + a_dim2 + a_dim3

        if mod_self_att:
            v_dim = v_dim1 + v_dim2 + v_dim3
            a_dim = a_dim1 + a_dim2 + a_dim3

            self.sa_hv = nn.MultiheadAttention(v_dim, no_of_heads, batch_first=True)
            self.sa_ha = nn.MultiheadAttention(a_dim, no_of_heads, batch_first=True)

        if late_self_att:
            self.sa_f = nn.MultiheadAttention(f_dim, no_of_heads, batch_first=True)
            if init_xav:
                nn.init.xavier_normal_(self.sa_f.in_proj_weight)
                nn.init.xavier_normal_(self.sa_f.out_proj.weight)

        if final_x_fc:
            self.fc_x = nn.Linear(f_dim, int(f_dim / 2))
            f_dim = int(f_dim / 2)
            if init_xav:
                nn.init.xavier_uniform_(self.fc_x.weight)
                nn.init.zeros_(self.fc_x.bias)

        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(f_dim, out_dim)
        if init_xav:
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, vx1, vx2, vx3, ax1, ax2, ax3):
        # 1st visual input
        if self.v_dim1 > 0:
            if self.batch_norm:
                vx1 = self.bn_v1(vx1)
            if self.early_self_att:
                vx1 = self.sa_v1(vx1, vx1, vx1)[0]

        # 2nd visual input
        if self.v_dim2 > 0:
            if self.batch_norm:
                vx2 = self.bn_v2(vx2)
            if self.early_self_att:
                vx2 = self.sa_v2(vx2, vx2, vx2)[0]

        # 3rd visual input
        if self.v_dim3 > 0:
            if self.batch_norm:
                vx3 = self.bn_v3(vx3)
            if self.early_self_att:
                vx3 = self.sa_v3(vx3, vx3, vx3)[0]

        # 1st audio input
        if self.a_dim1 > 0:
            if self.batch_norm:
                ax1 = self.bn_a1(ax1)
            if self.early_self_att:
                ax1 = self.sa_a1(ax1, ax1, ax1)[0]

        # 2nd audio input
        if self.a_dim2 > 0:
            if self.batch_norm:
                ax2 = self.bn_a2(ax2)
            if self.early_self_att:
                ax2 = self.sa_a2(ax2, ax2, ax2)[0]

        # 3rd audio input
        if self.a_dim3 > 0:
            if self.batch_norm:
                ax3 = self.bn_a3(ax3)
            if self.early_self_att:
                ax3 = self.sa_a3(ax3, ax3, ax3)[0]

        # concat different features
        if self.a_dim1 > 0 and self.v_dim1 > 0:
            # both feature modalities - concat
            vx = torch.cat((vx1, vx2, vx3), dim=1)
            ax = torch.cat((ax1, ax2, ax3), dim=1)
            if self.mod_self_att:
                vx = self.sa_hv(vx, vx, vx)[0]
                ax = self.sa_ha(ax, ax, ax)[0]
            x = torch.cat((vx, ax), dim=1)
        elif self.a_dim1 == 0 and self.v_dim1 > 0:
            # no audio feature use visual features
            vx = torch.cat((vx1, vx2, vx3), dim=1)
            if self.mod_self_att:
                vx = self.sa_hv(vx, vx, vx)[0]
            x = vx
        elif self.a_dim1 > 0 and self.v_dim1 == 0:
            # no visual feature use audio features
            ax = torch.cat((ax1, ax2, ax3), dim=1)
            if self.mod_self_att:
                ax = self.sa_ha(ax, ax, ax)[0]
            x = ax

        # late attention
        if self.late_self_att:
            x = self.sa_f(x, x, x)[0]

        # pre-final fc
        if self.final_x_fc:
            x = self.fc_x(x)

        # final fc
        x = self.dropout(x)
        x = self.fc(x)

        return x


def class_of_test_vid(test_vid_fn):
    pos = -1
    header = test_vid_fn.split('-')[0]
    for xll in range(len(labels)):
        if header in labels[xll]:
            pos = xll
    return pos


# set paths, create file list and load folds
top_dev = os.path.abspath('tau_dataset_development')
top_fold = os.path.join(top_dev, 'meta', 'evaluation_setup')
vid_fns = glob(os.path.join(top_dev, 'video', '*.mp4'))
vid_fns.sort()
for j, vid_fn in enumerate(vid_fns):
    if not os.path.isfile(vid_fn):
        print('!!!', vid_fn, 'not found')
        break
fold_train = csv2dict(os.path.join(top_fold, 'fold1_train.csv'))
fold_test = csv2dict(os.path.join(top_fold, 'fold1_test.csv'))
print(' train fold items:', len(fold_train))
print(' test fold items:', len(fold_test))
fold_train_names = [x['filename_video'].split('/')[1] for x in fold_train]
fold_test_names = [x['filename_video'].split('/')[1] for x in fold_test]

device = torch.device('cuda')

# load logs
done = []
skipped = 0
log_fn = os.path.join(out_folder, 'log.txt')
if os.path.isfile(log_fn) and os.path.isfile(log_fn):
    with open(log_fn, 'r') as fp:
        done = [line.strip() for line in fp.readlines() if line.startswith('---use:')]
else:
    done = []

for approach in APPROACHES:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    i_nets = approach['use']
    i_early_self_att = approach['early_self_att']
    i_mod_self_att = approach['mod_self_att']
    i_late_self_att = approach['late_self_att']
    i_final_x_fc = approach['final_x_fc']
    i_data_aug = approach['data_aug']
    i_scale = approach['scale']
    classes = approach['classes']

    # init mapping
    mapping = MAPPING10
    if classes == 3:
        mapping = MAPPING3

    # find unique labels
    labels = []
    for x in fold_train:
        if mapping[x['scene_label']] not in labels:
            labels.append(mapping[x['scene_label']])

    # create header
    header_info = \
        '---use:' + '-'.join(i_nets) + \
        ',bs:' + str(bs) + ',lr:' + str(lr) + \
        ',es:' + str(i_early_self_att) + \
        ',ms:' + str(i_mod_self_att) + \
        ',ls:' + str(i_late_self_att) + \
        ',ffc:' + str(i_final_x_fc) + \
        ',da:' + str(i_data_aug) + \
        ',do:' + str(dropout_rate) + \
        ',sm:' + str(scheduler_min) + \
        ',sc:' + str(i_scale) + \
        ',seed:' + str(SEED) + \
        ',classes:' + str(classes)

    if header_info in done:
        skipped += 1
        print(' skipping (%d): %s' % (skipped, header_info))
        continue

    print('\n ' + header_info)

    # get labels
    y_train_temp = [labels.index(mapping[x['scene_label']]) for x in fold_train]
    y_test_temp = [labels.index(mapping[x['filename_video'].split('/')[1].split('-')[0]]) for x in fold_test]
    y_train_fns = [x['filename_video'].split('/')[1] for x in fold_train]
    y_test_fns = [x['filename_video'].split('/')[1] for x in fold_test]


    # construct folds maps
    print('mapping... (data augm. =', i_data_aug, ')')
    if i_data_aug:
        map_train = [0] * len(vid_fns) * 10 * 2
        map_test = [0] * len(vid_fns) * 10 * 2
        map_loader_to_vid = []
        test_fns = []
        test_labels = []
        y_train = []
        y_test = []
        for ia in [0, 1]:
            for j, vid_fn in enumerate(vid_fns):
                for k in range(10):
                    vid_fn_x = os.path.basename(vid_fn)
                    if vid_fn_x in fold_train_names:
                        map_train[(ia * len(vid_fns) * 10) + (j * 10) + k] = 1
                        try:
                            ind = y_train_fns.index(vid_fn_x)
                            y_train.append(y_train_temp[ind])
                        except:
                            ind = y_test_fns.index(vid_fn_x)
                            y_train.append(y_test_temp[ind])
                    if vid_fn_x in fold_test_names:
                        map_test[(ia * len(vid_fns) * 10) + (j * 10) + k] = 1
                        ind = y_test_fns.index(vid_fn_x)
                        y_test.append(y_test_temp[ind])
                        map_loader_to_vid.append(y_test_fns.index(vid_fn_x))
                if vid_fn_x in fold_test_names:
                    test_fns.append(vid_fn_x)
                    test_labels.append(class_of_test_vid(vid_fn_x))
    else:
        map_train = [0] * len(vid_fns) * 10
        map_test = [0] * len(vid_fns) * 10
        map_loader_to_vid = []
        test_fns = []
        test_labels = []
        y_train = []
        y_test = []
        for j, vid_fn in enumerate(vid_fns):
            for k in range(10):
                vid_fn_x = os.path.basename(vid_fn)
                if vid_fn_x in fold_train_names:
                    map_train[(j * 10) + k] = 1
                    try:
                        ind = y_train_fns.index(vid_fn_x)
                        y_train.append(y_train_temp[ind])
                    except:
                        ind = y_test_fns.index(vid_fn_x)
                        y_train.append(y_test_temp[ind])
                if vid_fn_x in fold_test_names:
                    map_test[(j * 10) + k] = 1
                    ind = y_test_fns.index(vid_fn_x)
                    y_test.append(y_test_temp[ind])
                    map_loader_to_vid.append(y_test_fns.index(vid_fn_x))
            if vid_fn_x in fold_test_names:
                test_fns.append(vid_fn_x)
                test_labels.append(class_of_test_vid(vid_fn_x))

    print('train samples count: ', sum(map_train))
    print('test samples count:  ', sum(map_test))
    print('samples count sums:  ', sum(map_train) + sum(map_test))
    print('total files:         ', len(vid_fns))

    map_train = np.array([True if x else False for x in map_train])
    map_test = np.array([True if x else False for x in map_test])

    # load features
    feats = {}
    for arch in NETS_ALL:
        # load the model
        fn = os.path.join('features', 'feats_' + arch + '.pkl')
        with open(fn, 'rb') as fp:
            x = pickle.load(fp)

        if arch == 'clip':
            if not i_data_aug:
                feats[arch] = np.zeros((x.shape[0], x.shape[1]), dtype='float32')
                for j in range(x.shape[0]):
                    feats[arch][j, :] = x[j, :].copy()
                del x
            else:
                d = x[0].shape[-1]
                feats[arch] = np.zeros((len(x), d), dtype='float32')
                for j in range(len(x)):
                    feats[arch][j, :] = x[j].copy()
                del x

        if arch in ['openl3', 'pann', 'iov']:
            if i_data_aug:
                feats[arch] = np.vstack((x.copy(), x.copy()))
            else:
                feats[arch] = x.copy()
            del x
        elif arch not in ['clip']:
            if not i_data_aug:
                d = SHAPES[arch]
                feats[arch] = np.zeros((len(x) * 10, d), dtype='float32')
                for j in range(len(x)):
                    for k in range(10):
                        feats[arch][(j * 10) + k, :] = x[j][k]
                del x
            else:
                d = SHAPES[arch]
                feats[arch] = np.zeros((len(x), d), dtype='float32')

                for j in range(len(x)):
                    feats[arch][j, :] = x[j].copy()
                del x
        print('loading: %-20s %9s' % (os.path.basename(fn), feats[arch].shape))

    # scale features
    if i_scale:
        scaler = {}
        for arch in NETS_ALL:
            scaler[arch] = StandardScaler()
            scaler[arch].fit(feats[arch][map_train, :])
            bef_min = np.min(feats[arch])
            bef_max = np.max(feats[arch])

            feats[arch] = scaler[arch].transform(feats[arch])
            aft_min = np.min(feats[arch])
            aft_max = np.max(feats[arch])
            print('scaled %s: [%.3f,%.3f] -> [%.3f,%.3f]' % (arch, bef_min, bef_max, aft_min, aft_max))

    #
    #
    #################################
    # train dataset
    train_l = sum(map_train)

    # visual embeddings
    tensor_vxt1 = torch.Tensor(np.zeros((train_l, SHAPES['vit']))).cuda()
    tensor_vxt2 = torch.Tensor(np.zeros((train_l, SHAPES['clip']))).cuda()
    tensor_vxt3 = torch.Tensor(np.zeros((train_l, SHAPES['resnet50']))).cuda()
    if 'vit' in i_nets:
        tensor_vxt1 = torch.Tensor(feats['vit'][map_train, :]).cuda()
    if 'clip' in i_nets:
        tensor_vxt2 = torch.Tensor(feats['clip'][map_train, :]).cuda()
    if 'resnet50' in i_nets:
        tensor_vxt3 = torch.Tensor(feats['resnet50'][map_train, :]).cuda()

    # audio embeddings
    tensor_axt1 = torch.Tensor(np.zeros((train_l, SHAPES['openl3']))).cuda()
    tensor_axt2 = torch.Tensor(np.zeros((train_l, SHAPES['pann']))).cuda()
    tensor_axt3 = torch.Tensor(np.zeros((train_l, SHAPES['iov']))).cuda()
    if 'openl3' in i_nets:
        tensor_axt1 = torch.Tensor(feats['openl3'][map_train, :]).cuda()
    if 'pann' in i_nets:
        tensor_axt2 = torch.Tensor(feats['pann'][map_train, :]).cuda()
    if 'iov' in i_nets:
        tensor_axt3 = torch.Tensor(feats['iov'][map_train, :]).cuda()

    # labels
    tensor_yt = torch.Tensor(y_train).long().cuda()

    train_dataset = TensorDataset(
        tensor_vxt1, tensor_vxt2, tensor_vxt3,
        tensor_axt1, tensor_axt2, tensor_axt3, tensor_yt)
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)

    #
    #
    #################################
    # validation dataset
    valid_l = sum(map_test)

    # visual embeddings
    tensor_vxv1 = torch.Tensor(np.zeros((valid_l, SHAPES['vit']))).cuda()
    tensor_vxv2 = torch.Tensor(np.zeros((valid_l, SHAPES['clip']))).cuda()
    tensor_vxv3 = torch.Tensor(np.zeros((valid_l, SHAPES['resnet50']))).cuda()
    if 'vit' in i_nets:
        tensor_vxv1 = torch.Tensor(feats['vit'][map_test, :]).cuda()
    if 'clip' in i_nets:
        tensor_vxv2 = torch.Tensor(feats['clip'][map_test, :]).cuda()
    if 'resnet50' in i_nets:
        tensor_vxv3 = torch.Tensor(feats['resnet50'][map_test, :]).cuda()

    # audio embeddings
    tensor_axv1 = torch.Tensor(np.zeros((valid_l, SHAPES['openl3']))).cuda()
    tensor_axv2 = torch.Tensor(np.zeros((valid_l, SHAPES['pann']))).cuda()
    tensor_axv3 = torch.Tensor(np.zeros((valid_l, SHAPES['iov']))).cuda()
    if 'openl3' in i_nets:
        tensor_axv1 = torch.Tensor(feats['openl3'][map_test, :]).cuda()
    if 'pann' in i_nets:
        tensor_axv2 = torch.Tensor(feats['pann'][map_test, :]).cuda()
    if 'iov' in i_nets:
        tensor_axv3 = torch.Tensor(feats['iov'][map_test, :]).cuda()

    # labels
    tensor_yv = torch.Tensor(y_test).long().cuda()

    test_dataset = TensorDataset(
        tensor_vxv1, tensor_vxv2, tensor_vxv3,
        tensor_axv1, tensor_axv2, tensor_axv3, tensor_yv)
    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

    #
    #
    #####################################
    # select features to include
    lv_dim1 = lv_dim2 = lv_dim3 = la_dim1 = la_dim2 = la_dim3 = 0
    if 'vit' in i_nets:
        lv_dim1 = SHAPES['vit']
    if 'clip' in i_nets:
        lv_dim2 = SHAPES['clip']
    if 'resnet50' in i_nets:
        lv_dim3 = SHAPES['resnet50']

    if 'openl3' in i_nets:
        la_dim1 = SHAPES['openl3']
    if 'pann' in i_nets:
        la_dim2 = SHAPES['pann']
    if 'iov' in i_nets:
        la_dim3 = SHAPES['iov']

    model = NetCombFinal(
        lv_dim1, lv_dim2, lv_dim3,
        la_dim1, la_dim2, la_dim3,
        early_self_att=i_early_self_att,
        mod_self_att=i_mod_self_att,
        late_self_att=i_late_self_att,
        final_x_fc=i_final_x_fc,
        drop_rate=dropout_rate,
        out_dim=len(labels),
        init_xav=init_xavier
    )

    model.to(device)
    model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=lr, momentum=sgd_momentum,
                          weight_decay=l2_reg)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(20, int(epochs * 0.8)),
        eta_min=lr * scheduler_min)

    for epoch in range(epochs):

        # train for one epoch
        print(' Epoch %d: ' % (epoch + 1))
        model.train()
        running_loss = 0.0
        last_loss = 0.

        # calculate n_weights
        if l1_reg > 0.0:
            n_weights = 0
            for name, weights in model.named_parameters():
                if 'bias' not in name:
                    n_weights = n_weights + weights.numel()

        for iii, data in enumerate(training_loader):
            # get data
            v_input1, v_input2, v_input3, a_input1, a_input2, a_input3, in_labels = data

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(v_input1, v_input2, v_input3, a_input1, a_input2, a_input3)

            # calculate the loss
            loss = loss_fn(outputs, in_labels)

            # regularize loss using L1 regularization
            if l1_reg > 0.0:
                # Calculate L1 term
                l1_term = torch.tensor(0., requires_grad=True)
                for name, weights in model.named_parameters():
                    if 'bias' not in name:
                        weights_sum = torch.sum(torch.abs(weights))
                        l1_term = l1_term + weights_sum
                l1_term = l1_term / n_weights
                loss = loss - l1_term * l1_reg

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            running_loss += loss.item()
            if iii % print_interval == (print_interval - 1):
                last_loss = running_loss / print_interval
                print('   batch %6d loss: %.7f' % (iii + 1, last_loss))
                running_loss = 0.

        # update lr schedule
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        # evaluate current epoch
        model.eval()

        num_samples = 0
        validation_running_loss = 0.0
        validation_true_y = []
        validation_predictions_gathered = []

        no_of_samples = len(y_test_fns)
        if i_data_aug:
            no_of_samples *= 2

        for iii, valid_data in enumerate(validation_loader):
            v_inputs1, v_input2, v_input3, a_input1, a_input2, a_input3, in_labels = valid_data

            validation_outputs = model(v_inputs1, v_input2, v_input3, a_input1, a_input2, a_input3)

            _, validation_predictions = validation_outputs.max(1)
            num_samples += validation_predictions.size(0)

            validation_true_y += in_labels.tolist()
            validation_predictions_gathered += validation_predictions.tolist()

            validation_loss = loss_fn(validation_outputs, in_labels)
            validation_running_loss += validation_loss.item()

        num_voting_correct = 0
        for iii in range(0, len(validation_true_y), 10):
            istart = iii
            votes = [0] * len(labels)
            for jjj in range(10):
                votes[validation_predictions_gathered[istart + jjj]] += 1
            if np.argmax(votes) == validation_true_y[iii]:
                num_voting_correct += 1

        validation_voting_acc = float(num_voting_correct) / float(no_of_samples) * 100
        validation_running_loss /= float(num_samples)

        print(' validation - acc %.5f - loss %.5f' % (validation_voting_acc, validation_running_loss))
        print(' (lr: %.7f -> %.7f) ' % (before_lr, after_lr))

        if epoch + 1 in save_epochs:
            with open(os.path.join(out_folder, 'log.txt'), 'a') as fp:
                fp.write('%d,%.9f,%.5f\n' % (epoch + 1, last_loss, validation_voting_acc))

            model_fn = 'model_%s_e%2d_acc%.2f.pt' % (
                approach['name'], epoch + 1, validation_voting_acc)
            torch.save(model, os.path.join(out_folder, model_fn))

            del validation_true_y, validation_predictions_gathered, validation_running_loss
            del votes
            del validation_outputs, validation_predictions

    del model
    del loss_fn

    del test_dataset
    del train_dataset

    del tensor_vxt1
    del tensor_vxt2
    del tensor_vxt3
    del tensor_axt1
    del tensor_axt2
    del tensor_axt3
    del tensor_yt

    del tensor_vxv1
    del tensor_vxv2
    del tensor_vxv3
    del tensor_axv1
    del tensor_axv2
    del tensor_axv3
    del tensor_yv

    del validation_loader
    del training_loader

    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(60)
