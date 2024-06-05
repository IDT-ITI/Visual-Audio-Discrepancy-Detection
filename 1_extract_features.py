import os
import sys
import time
import csv 

import numpy as np
import cv2
from PIL import Image
from glob import glob
import pickle

import random

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn

import imutils
from imutils.video import FileVideoStream

import open_clip

DO_DCNNS = True
DO_CLIP = True
device = torch.device('cuda')
features = {}


def adjust_contrast(img_in, contrast=127):
	contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

	alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
	gamma = 127 * (1 - alpha)

	# The function addWeighted calculates
	# the weighted sum of two arrays
	cal = cv2.addWeighted(img_in, alpha, img_in, 0, gamma)

	return cal


def adjust_img(img, brightness=255, contrast=127):
	brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
	contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

	if brightness != 0:
		if brightness > 0:
			shadow = brightness
			maxi = 255
		else:
			shadow = 0
			maxi = 255 + brightness
		al_pha = (maxi - shadow) / 255
		ga_mma = shadow

		# The function addWeighted
		# calculates the weighted sum
		# of two arrays
		cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)
	else:
		cal = img

	if contrast != 0:
		alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
		gamma = 127 * (1 - alpha)

		# The function addWeighted calculates
		# the weighted sum of two arrays
		cal = cv2.addWeighted(cal, alpha, cal, 0, gamma)

	return cal


def csv2dict(fn):
	with open(fn, 'r') as f:
		dict_reader = csv.DictReader(f, delimiter='\t')
		list_of_dict = list(dict_reader)
	return list_of_dict


def recursion_change_bn(module):
	# hacky way to deal with the Pytorch 1.0 update
	if isinstance(module, torch.nn.BatchNorm2d):
		module.track_running_stats = 1
	else:
		for i, (name, module1) in enumerate(module._modules.items()):
			module1 = recursion_change_bn(module1)
	return module


def hook_feature(module, input, output):
	features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def init_labels():
	# prepare all the labels
	# scene category relevant
	classes = []
	with open(os.path.join('places365_models', 'categories_places365.txt')) as class_file:
		for line in class_file:
			classes.append(line.strip().split(' ')[0][3:])
	classes = tuple(classes)

	# indoor and outdoor relevant
	labels_IO = []
	with open(os.path.join('places365_models', 'IO_places365.txt')) as f:
		lines = f.readlines()
		for line in lines:
			items = line.rstrip().split()
			labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
	labels_IO = np.array(labels_IO)

	# scene attribute relevant
	with open(os.path.join('places365_models', 'labels_sunattribute.txt')) as f:
		lines = f.readlines()
		labels_attribute = [item.rstrip() for item in lines]
	W_attribute = np.load(os.path.join('places365_models', 'W_sceneattribute_wideresnet18.npy'))

	return classes, labels_IO, labels_attribute, W_attribute


def init_model(arch):

	if arch == 'wideresnet':
		# this model has a last conv feature map as 14x14
		model_file = 'places365_models/wideresnet18_places365.pth.tar'
		sys.path.append('/ssd/TAU/places365_models/')
		import wideresnet
		l_model = wideresnet.resnet18(num_classes=365)
		# checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
		checkpoint = torch.load(model_file, map_location=device)
		state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
		l_model.load_state_dict(state_dict)

		# hacky way to deal with the upgraded batchnorm2D and avgpool layers...
		for i, (name, module) in enumerate(l_model._modules.items()):
			module = recursion_change_bn(l_model)
		l_model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
		l_model.to(device)
		l_model.cuda()
		l_model.eval()

		# load the image transformer
		tf = trn.Compose([
			trn.Resize((224, 224)),
			trn.ToTensor(),
			trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		return l_model, tf

	elif arch == 'effnet':
		# model = models.efficientnet_b5(pretrained=True, weights=models.EfficientNet_B5_Weights.DEFAULT)
		l_model = models.efficientnet_v2_l(pretrained=True, weights=models.EfficientNet_V2_L_Weights.DEFAULT)
		l_model.to(device)
		l_model.cuda()
		l_model.eval()

		# load the image transformer
		tf = models.EfficientNet_V2_L_Weights.DEFAULT.transforms()

		return l_model, tf

	elif arch == 'vit':
		l_model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
		l_model.to(device)
		l_model.cuda()
		l_model.eval()

		# load the image transformer
		tf = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()

		return l_model, tf

	else:
		# load the pre-trained weights
		model_file = 'places365_models/%s_places365.pth.tar' % arch
		l_model = models.__dict__[arch](num_classes=365)
		checkpoint = torch.load(model_file, map_location=device)
		state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
		l_model.load_state_dict(state_dict)
		l_model.to(device)
		l_model.cuda()
		l_model.eval()

		# load the image transformer
		centre_crop = trn.Compose([
				trn.Resize((256, 256)),
				trn.CenterCrop(224),
				trn.ToTensor(),
				trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		return l_model, centre_crop


def get_activation(name):
	def hook(model, input, output):
		features[name] = output.detach()
	return hook


# load folds
top_dev = os.path.abspath('tau_dataset_development')
top_eval = os.path.abspath('./evaluation')
top_fold = os.path.join('.', 'tau_dataset_development', 'meta', 'evaluation_setup')
fold_train = csv2dict(os.path.join(top_fold, 'fold1_train.csv'))
fold_test = csv2dict(os.path.join(top_fold, 'fold1_test.csv'))
fold_eval = csv2dict(os.path.join(top_fold, 'fold1_evaluate.csv'))
print(' train fold items:', len(fold_train))
print(' test fold items:', len(fold_test))
print(' eval fold items:', len(fold_eval))

# get list of folds filenames
labels = []
for x in fold_train:
	if x['scene_label'] not in labels:
		labels.append(x['scene_label'])
		print(len(labels), labels[-1])

# load the labels
classes, labels_IO, labels_attribute, W_attribute = init_labels()

# create list
vid_fns = glob(os.path.join(top_dev, 'video', '*.mp4'))
vid_fns.sort()
for j, vid_fn in enumerate(vid_fns):
	if not os.path.isfile(vid_fn):
		print('!!!', vid_fn, 'not found')
		break
out_path = 'features'

print(' all paths checked!')

if DO_DCNNS:
	for i, arch in enumerate(['resnet50', 'vit', 'effnet']):
		# load the model
		model, tf = init_model(arch)

		d = 0
		if arch == 'wideresnet':
			model.avgpool.register_forward_hook(get_activation('avgpool'))
			act_name = 'avgpool'
			d = 512
		elif arch == 'resnet50':
			model.avgpool.register_forward_hook(get_activation('avgpool'))
			act_name = 'avgpool'
			d = 2048
		elif arch == 'effnet':
			model.avgpool.register_forward_hook(get_activation('avgpool'))
			act_name = 'avgpool'
			d = 1280
		elif arch == 'vit':
			model.heads.head.register_forward_hook(get_activation('ln'))
			act_name = 'ln'
			d = 1000

		feats = []

		with torch.no_grad():
			for augment in [0, 1]:
				for j, vid_fn in enumerate(vid_fns):

					start = time.perf_counter()

					# load video
					cv_images = []
					fvs = FileVideoStream(vid_fn).start()
					fps = 30
					count = 1
					while fvs.more():
						frame = fvs.read()
						if count % fps == 0:
							cv_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
						count += 1
					fvs.stop()
					if len(cv_images) != 10:
						input('...')

					# forward image
					for k, cv_image in enumerate(cv_images):
						if augment == 1:
							cv_image = cv2.flip(cv_image, 1)
							cv_image = adjust_img(cv_image,
												  brightness=255+int(random.uniform(-1, 1)*40),
												  contrast=127+int(random.uniform(-1, 1)*40))
							cv_image = imutils.rotate(cv_image, int(random.uniform(-1, 1)*5))
						foo = model(V(tf(Image.fromarray(cv_image))).unsqueeze(0).cuda())

						# register
						inter = features[act_name]
						feats.append(inter.squeeze().detach().cpu().numpy().copy())

					end = time.perf_counter()
					print(' %s, %d/%d, avg t: %.3fs' % (arch, len(feats), 2*10*len(vid_fns), end - start))

		with open(os.path.join(out_path, 'feats_'+arch+'_x.pkl'), 'wb') as fp:
			pickle.dump(feats, fp, protocol=pickle.HIGHEST_PROTOCOL)

		del model
		del tf

#
#
# ##############
# Clip
#

if DO_CLIP:
	print(' initializing clip model')
	clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
	tokenizer = open_clip.get_tokenizer('ViT-H-14')
	clip_model.to(device)

	feats = []

	with torch.no_grad():
		for augment in [0, 1]:
			for j, vid_fn in enumerate(vid_fns):
				start = time.perf_counter()

				# load video
				cv_images = []
				fvs = FileVideoStream(vid_fn).start()
				fps = 30
				count = 1
				while fvs.more():
					frame = fvs.read()
					if count % fps == 0:
						cv_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
					count += 1
				fvs.stop()
				if len(cv_images) != 10:
					input('...')

				# forward image
				input_img = []
				for k, cv_image in enumerate(cv_images):
					if augment == 1:
						cv_image = cv2.flip(cv_image, 1)
						cv_image = adjust_img(cv_image,
											brightness=255+int(random.uniform(-1, 1)*40),
											contrast=127+int(random.uniform(-1, 1)*40))
						cv_image = imutils.rotate(cv_image, int(random.uniform(-1, 1)*5))
					input_img.append(clip_preprocess(Image.fromarray(cv_image)))
				input_images = torch.stack(input_img, dim=0)
				img_feats = clip_model.encode_image(input_images.cuda())

				# register
				img_feats = img_feats.squeeze().detach().cpu().numpy().copy()
				for k in range(len(cv_images)):
					feats.append(img_feats[k,:])

				end = time.perf_counter()
				print(' %s, %d/%d, avg t: %.3fs' % ('CLIP', len(feats), 2*10*len(vid_fns), end - start))

	with open(os.path.join(out_path, 'feats_clip_x.pkl'), 'wb') as fp:
		pickle.dump(feats, fp, protocol=pickle.HIGHEST_PROTOCOL)
