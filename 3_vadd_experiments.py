import gc
import os
import csv
import time
import numpy as np
import random

from glob import glob
import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


random.seed(5)
np.random.seed(5)
torch.manual_seed(5)

log_prefix = 'vadd_exps'

NETS_ALL = ['vit', 'clip', 'resnet50', 'openl3', 'pann', 'iov']

SHAPES = {
    'vit': 1000,
    'clip': 1024,
    'resnet50': 2048,
    'openl3': 512,
    'pann': 512,
    'iov': 256
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
                 out_dim=10, batch_norm=0, act_after_fc=0):
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
        self.act_after_fc = act_after_fc

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
                self.sa_v1 = nn.MultiheadAttention(v_dim1, 1, batch_first=True)
            if v_dim2:
                self.sa_v2 = nn.MultiheadAttention(v_dim2, 1, batch_first=True)
            if v_dim3:
                self.sa_v3 = nn.MultiheadAttention(v_dim3, 1, batch_first=True)

            if a_dim1:
                self.sa_a1 = nn.MultiheadAttention(a_dim1, 1, batch_first=True)
            if a_dim2:
                self.sa_a2 = nn.MultiheadAttention(a_dim2, 1, batch_first=True)
            if a_dim3:
                self.sa_a3 = nn.MultiheadAttention(a_dim3, 1, batch_first=True)

        f_dim = v_dim1 + v_dim2 + v_dim3 + a_dim1 + a_dim2 + a_dim3

        if mod_self_att:
            v_dim = v_dim1 + v_dim2 + v_dim3
            a_dim = a_dim1 + a_dim2 + a_dim3

            self.sa_hv = nn.MultiheadAttention(v_dim, 1, batch_first=True)
            self.sa_ha = nn.MultiheadAttention(a_dim, 1, batch_first=True)

        if late_self_att:
            self.sa_f = nn.MultiheadAttention(f_dim, 1, batch_first=True)
            if init_xav:
                nn.init.xavier_normal_(self.sa_f.in_proj_weight)
                nn.init.xavier_normal_(self.sa_f.out_proj.weight)

        if final_x_fc:
            self.fc_x = nn.Linear(f_dim, int(f_dim / 2))
            f_dim = int(f_dim / 2)
            if init_xav:
                nn.init.xavier_uniform_(self.fc_x.weight)
                nn.init.zeros_(self.fc_x.bias)

        if act_after_fc:
            self.a_fc_x = nn.PReLU(num_parameters=f_dim)

        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(f_dim, out_dim)
        if init_xav:
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, vx1, vx2, vx3, ax1, ax2, ax3):
        # first visual input
        if self.v_dim1:
            if self.batch_norm:
                vx1 = self.bn_v1(vx1)
            if self.early_self_att:
                vx1 = self.sa_v1(vx1, vx1, vx1)[0]

        # second visual input
        if self.v_dim2:
            if self.batch_norm:
                vx2 = self.bn_v2(vx2)
            if self.early_self_att:
                vx2 = self.sa_v2(vx2, vx2, vx2)[0]

        # third visual input
        if self.v_dim3:
            if self.batch_norm:
                vx3 = self.bn_v3(vx3)
            if self.early_self_att:
                vx3 = self.sa_v3(vx3, vx3, vx3)[0]

        # 1st audio input
        if self.a_dim1:
            if self.batch_norm:
                ax1 = self.bn_a1(ax1)
            if self.early_self_att:
                ax1 = self.sa_a1(ax1, ax1, ax1)[0]

        # 2nd audio input
        if self.a_dim2:
            if self.batch_norm:
                ax2 = self.bn_a2(ax2)
            if self.early_self_att:
                ax2 = self.sa_a2(ax2, ax2, ax2)[0]

        # 3rd audio input
        if self.a_dim3:
            if self.batch_norm:
                ax3 = self.bn_a3(ax3)
            if self.early_self_att:
                ax3 = self.sa_a3(ax3, ax3, ax3)[0]

        # concat
        if self.mod_self_att:
            if self.v_dim1:
                vx = torch.cat((vx1, vx2, vx3), dim=1)
                vx = self.sa_hv(vx, vx, vx)[0]

            if self.a_dim1:
                ax = torch.cat((ax1, ax2, ax3), dim=1)
                ax = self.sa_ha(ax, ax, ax)[0]

            if self.a_dim1 and self.v_dim1:
                x = torch.cat((vx, ax), dim=1)
            elif not self.a_dim1 and self.v_dim1:
                x = vx
            elif self.a_dim1 and not self.v_dim1:
                x = ax
        else:
            if self.a_dim1 and self.v_dim1:
                x = torch.cat((vx1, vx2, vx3, ax1, ax2, ax3), dim=1)
            elif not self.a_dim1 and self.v_dim1:
                x = torch.cat((vx1, vx2, vx3), dim=1)
            elif self.a_dim1 and not self.v_dim1:
                x = torch.cat((ax1, ax2, ax3), dim=1)

        # late attention
        if self.late_self_att:
            x = self.sa_f(x, x, x)[0]

        # pre-final fc
        if self.final_x_fc:
            x = self.fc_x(x)
            if self.act_after_fc:
                x = self.a_fc_x(x)

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


def validate_one_epoch(c_nets):
    num_correct = 0
    num_samples = 0
    running_loss = 0.0

    for i, valid_data in enumerate(validation_loader):

        v_inputs1, v_input2, v_input3, v_input4, a_input1, a_input2, a_input3, in_labels = valid_data

        if 'resnet50' not in c_nets:
            v_input3 = None
        if 'effnet' not in c_nets:
            v_input4 = None
        if 'iov' not in c_nets:
            a_input3 = None

        outputs = model(v_inputs1, v_input2, v_input3, v_input4, a_input1, a_input2, a_input3)

        _, predictions = outputs.max(1)
        num_correct += float((predictions == in_labels).sum())
        num_samples += predictions.size(0)

        loss = loss_fn(outputs, in_labels)
        running_loss += loss.item()

    valid_acc = float(num_correct) / float(num_samples) * 100
    running_loss /= float(num_samples)

    return valid_acc, running_loss


def validate_one_epoch_voting(c_nets, XF):
    num_correct = 0
    num_samples = 0
    running_loss = 0.0

    global map_loader_to_vid

    true_y = []
    pred_gathered = []

    no_of_samples = len(y_test_fns)
    if XF:
       no_of_samples *= 2
    votes = np.zeros((no_of_samples, len(labels)))
    # filled = [0] * no_of_samples
    divider = 10
    if XF:
       divider = 20

    upper = -1
    for i, valid_data in enumerate(validation_loader):

        v_inputs1, v_input2, v_input3, v_input4, a_input1, a_input2, a_input3, in_labels = valid_data

        if 'resnet50' not in c_nets:
            v_input3 = None
        if 'effnet' not in c_nets:
            v_input4 = None
        if 'iov' not in c_nets:
            a_input3 = None

        outputs = model(v_inputs1, v_input2, v_input3, v_input4, a_input1, a_input2, a_input3)

        _, predictions = outputs.max(1)
        num_correct += float((predictions == in_labels).sum())
        num_samples += predictions.size(0)

        true_y += in_labels.tolist()
        pred_gathered += predictions.tolist()

        loss = loss_fn(outputs, in_labels)
        running_loss += loss.item()

    num_voting_correct = 0
    for i in range(0, len(true_y), 10):
        istart = i
        iend = i + 9
        votes = [0] * len(labels)
        for j in range(10):
            votes[pred_gathered[istart+j]] += 1
        if np.argmax(votes) == true_y[i]:
            num_voting_correct += 1

    valid_voting_acc = float(num_voting_correct) / float(no_of_samples) * 100
    valid_acc = float(num_correct) / float(num_samples) * 100
    running_loss /= float(num_samples)

    return valid_voting_acc, valid_acc, running_loss


# Load models
models = {'v': {}}
models['v'][10] = NetCombFinal(v_dim1=SHAPES['vit'], v_dim2=SHAPES['clip'], v_dim3=SHAPES['resnet50'],
                               a_dim1=0, a_dim2=0, a_dim3=0,
                               final_x_fc=1, late_self_att=1, out_dim=10)
models['v'][10].load_state_dict(torch.load(os.path.join('models', 'model_a-ls-3.pt')))
models['v'][10].eval()

models['v'][3] = NetCombFinal(v_dim1=SHAPES['vit'], v_dim2=SHAPES['clip'], v_dim3=SHAPES['resnet50'],
                              a_dim1=0, a_dim2=0, a_dim3=0,
                              final_x_fc=1, late_self_att=1, out_dim=3)
models['v'][3].load_state_dict(torch.load(os.path.join('models', 'model_a-ls-3_e10.pt')))
models['v'][3].eval()

models['a'] = {}
models['a'][10] = NetCombFinal(v_dim1=0, v_dim2=0, v_dim3=0,
                               a_dim1=SHAPES['openl3'], a_dim2=SHAPES['pann'], a_dim3=SHAPES['iov'],
                               final_x_fc=1, late_self_att=1, out_dim=10)
models['a'][10].load_state_dict(torch.load(os.path.join('models', 'model_a-ls-10.pt')))
models['a'][10].eval()

models['a'][3] = NetCombFinal(v_dim1=0, v_dim2=0, v_dim3=0,
                              a_dim1=SHAPES['openl3'], a_dim2=SHAPES['pann'], a_dim3=SHAPES['iov'],
                              final_x_fc=1, late_self_att=1, out_dim=3)
models['a'][3].load_state_dict(torch.load(os.path.join('models', 'model_a-ls-3_e10.pt')))


device = torch.device('cuda')
for modality in ['v', 'a']:
    for classes in [10, 3]:
        models[modality][classes].eval()
        models[modality][classes].to(device)
        models[modality][classes].cuda()

# set paths, create file list and load folds
top_dev = os.path.abspath('tau_dataset_development')
top_fold = os.path.join(top_dev, 'meta', 'evaluation_setup')
vid_fns = glob(os.path.join(top_dev, 'video', '*.mp4'))
vid_fns.sort()
for j, vid_fn in enumerate(vid_fns):
    if not os.path.isfile(vid_fn):
        print('!!!', vid_fn, 'not found')
        break
print(' all paths checked!')
fold_train = csv2dict(os.path.join(top_fold, 'fold1_train.csv'))
fold_test = csv2dict(os.path.join(top_fold, 'fold1_test.csv'))
print(' train fold items:', len(fold_train))
print(' test fold items:', len(fold_test))
fold_train_names = [x['filename_video'].split('/')[1] for x in fold_train]
fold_test_names = [x['filename_video'].split('/')[1] for x in fold_test]


for i_scenario, classes in enumerate([10, 3]):

    # init mapping
    if classes == 3:
        mapping = {
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
        pass
    elif classes == 10:
        mapping = {
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

    # find unique labels
    labels = []
    for x in fold_train:
        if mapping[x['scene_label']] not in labels:
            labels.append(mapping[x['scene_label']])
            print(len(labels), labels[-1])

    # get labels
    y_train_temp = [labels.index(mapping[x['scene_label']]) for x in fold_train]
    y_test_temp = [labels.index(mapping[x['filename_video'].split('/')[1].split('-')[0]]) for x in fold_test]
    y_train_fns = [x['filename_video'].split('/')[1] for x in fold_train]
    y_test_fns = [x['filename_video'].split('/')[1] for x in fold_test]

    print('\n ~~~ Splitting and syncing')

    if XXF > 0.0:
        xx_samples = 0
        random.seed(11)
        test_set_len = len(fold_test_names)
        limit_use = int(test_set_len * XXF)
        random_indexes = random.sample(range(0, test_set_len), limit_use)
        cur_index = -1
        for i in range(len(fold_test_names)):
            if i in random_indexes:
                fold_train_names.append(fold_test_names[i])
                xx_samples += 1
        print('test_set_len:        ', test_set_len)
        print('use limit:           ', limit_use)
        print('xx_samples:          ', xx_samples)
        print('...')

    # construct folds maps
    if XF:
        map_train = [0] * len(vid_fns) * 10 * 2
        map_test = [0] * len(vid_fns) * 10 * 2
        map_loader_to_vid = []
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
    else:
        map_train = [0] * len(vid_fns) * 10
        map_test = [0] * len(vid_fns) * 10
        map_loader_to_vid = []
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
                # if vid_fn_x in fold_test_names:
                #     map_test[(j * 10) + k] = 1
                #     ind = y_test_fns.index(vid_fn_x)
                #     y_test.append(y_test_temp[ind])
                #     map_loader_to_vid.append(y_test_temp[ind])
                if vid_fn_x in fold_test_names:
                    map_test[(j * 10) + k] = 1
                    ind = y_test_fns.index(vid_fn_x)
                    y_test.append(y_test_temp[ind])
                    map_loader_to_vid.append(y_test_fns.index(vid_fn_x))

    print('train samples count: ', sum(map_train))
    print('test samples count:  ', sum(map_test))
    print('samples count sums:  ', sum(map_train) + sum(map_test))
    print('total files:         ', len(vid_fns))
    if XF:
        print('total samples:       ', len(vid_fns) * 10 * 2)
    else:
        print('total samples:       ', len(vid_fns) * 10)
    map_train = np.array([True if x else False for x in map_train])
    map_test = np.array([True if x else False for x in map_test])

    # load features
    print('\n ~~~ Loading')
    feats = {}
    for arch in NETS_ALL:
        # load the model
        if XF and arch in ['vit', 'clip', 'resnet50', 'effnet', ]:
            fn = os.path.join('features', 'feats_' + arch + '_x.pkl')
        else:
            fn = os.path.join('features', 'feats_' + arch + '.pkl')

        print('loading', fn)
        with open(fn, 'rb') as fp:
            x = pickle.load(fp)

        if arch == 'clip':
            if not XF:
                feats[arch] = np.zeros((x.shape[0], x.shape[1]), dtype='float32')
                print('arch: %-18s - shape: %9s' % (arch, feats[arch].shape))
                for j in range(x.shape[0]):
                    feats[arch][j, :] = x[j, :].copy()
                del x
            else:
                d = x[0].shape[-1]
                feats[arch] = np.zeros((len(x), d), dtype='float32')
                print('arch: %-18s - shape: %9s' % (arch, feats[arch].shape))
                for j in range(len(x)):
                    feats[arch][j, :] = x[j].copy()
                del x

        if arch in ['pann', 'openl3', 'openl32', 'iov']:
            if XF:
                feats[arch] = np.vstack((x.copy(), x.copy()))
            else:
                feats[arch] = x.copy()
            print('arch: %-18s - shape: %9s' % (arch, x.shape))
            del x
        elif arch not in ['clip']:
            if (not XF):
                d = SHAPES[arch]
                feats[arch] = np.zeros((len(x)*10, d), dtype='float32')
                print('arch: %-18s - shape: %9s' % (arch, feats[arch].shape))
                for j in range(len(x)):
                    for k in range(10):
                        feats[arch][(j*10)+k, :] = x[j][k]
                del x
            else:
                d = SHAPES[arch]
                feats[arch] = np.zeros((len(x), d), dtype='float32')
                print('arch: %-18s - shape: %9s' % (arch, feats[arch].shape))
                for j in range(len(x)):
                    feats[arch][j, :] = x[j].copy()
                del x

    header_info = \
        '---use:' + '-'.join(i_nets) + \
        ',bs:' + str(BS) + ',lr:' + str(LR) + \
        ',hu:' + str(i_hu) + ',int_fcs:' + str(i_int_fcs) + \
        ',act:' + str(i_use_act) + ',att:' + str(i_use_att) + \
        ',bnb:' + str(0) + ',bna:' + str(0) + \
        ',ffc:' + str(i_final_x_fc) + ',fatt:' + str(i_final_att) + \
        ',hfc:' + str(i_hier_fc) + ',hat:' + str(i_hier_att) + \
        ',act_after_fc:' + str(i_act_after_fc) + \
        ',xf:' + str(XF) + \
        ',classes:' + str(len(labels))

    print(header_info)

    # visual embeddings
    tensor_vxv1 = torch.Tensor(
        feats['vit'][map_test, :]).cuda()
    tensor_vxv2 = torch.Tensor(
        feats['clip'][map_test, :]).cuda()
    tensor_vxv3 = torch.Tensor(
        feats['resnet50'][map_test, :]).cuda()
    vdim1 = SHAPES['vit']
    vdim2 = SHAPES['clip']
    vdim3 = SHAPES['resnet50']

    # audio embeddings
    tensor_axv1 = torch.Tensor(
        feats['openl3'][map_test, :]).cuda()
    tensor_axv2 = torch.Tensor(
        feats['pann'][map_test, :]).cuda()
    tensor_axv3 = torch.Tensor(
        feats['iov'][map_test, :]).cuda()
    adim1 = SHAPES['openl3']
    adim2 = SHAPES['pann']
    adim3 = SHAPES['iov']

    # labels
    tensor_yv = torch.Tensor(y_test).long().cuda()

    # visual validation set
    test_visual_dataset = TensorDataset(
        tensor_vxv1, tensor_vxv2, tensor_vxv3, tensor_yv)
    validation_visual_loader = torch.utils.data.DataLoader(
        test_visual_dataset,  batch_size=BS, shuffle=False)
    print('Visual validation set: %d' % (len(test_visual_dataset)))

    # audio validation set
    test_audio_dataset = TensorDataset(
        tensor_axv1, tensor_axv2, tensor_axv3, tensor_yv)
    validation_audio_loader = torch.utils.data.DataLoader(
        test_audio_dataset, batch_size=BS, shuffle=False)
    print('Visual validation set: %d' % (len(test_audio_dataset)))

    model = NetCombFinal(
        vdim1, vdim2, vdim3, 0,
        adim1, adim2, adim3,
        int_fcs=i_int_fcs, use_act=i_use_act,
        use_att=i_use_att, weighted_concat=0,
        hier_fc=i_hier_fc, hier_att=i_hier_att,
        final_att=i_final_att, final_x_fc=i_final_x_fc,
        use_bn_bef=0, use_bn_after=0,
        out_dim=len(labels), drop_rate=dropout_rate,
        final_trans=0,
    )

    model.to(device)
    model.cuda()
    model.train(False)
    vacc, temp_foo, vloss = validate_one_epoch_voting(i_nets, XF)

    time.sleep(20)

    del model
    del vacc
    del vloss

    del test_dataset

    del tensor_vxv1
    del tensor_vxv2
    del tensor_vxv3
    del tensor_vxv4
    del tensor_axv1
    del tensor_axv2
    del tensor_axv3
    del tensor_yv

    del validation_loader

    time.sleep(20)

    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)
