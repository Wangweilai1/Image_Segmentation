import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from network import U_Net
import cv2
from PIL import Image
from torch.autograd import Variable 

def getContourAndCalculateArea(imgOrg, imgMask, bin_thread = 125, color_contour = (0, 0, 255), text_contour = (0, 255, 0)):
    imgMaskGray = cv2.GaussianBlur(imgMask, (3, 3), 0)
    _, binary =  cv2.threshold(imgMaskGray, bin_thread, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_ratio, w_ratio = imgOrg.shape[0] / imgMask.shape[0], imgOrg.shape[1] / imgMask.shape[1]
    idx = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if(area < 100):
            continue
        contour = contour * (w_ratio, h_ratio)
        bbox = contour.astype('int32').reshape(-1)
        if len(bbox) <= 4:
            continue
        cv2.drawContours(imgOrg, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, color_contour, 5)
        index = "{0}".format(idx)
        idx += 1
        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
        _ = cv2.putText(imgOrg, index, (bbox[0], bbox[1]), font, 2, text_contour, 3)
    return imgOrg

class SolverTest(object):
	def __init__(self, config, test_loader = []):

		# Data loader
		self.test_loader = test_loader

		# Models
		self.unet_road = None
		self.unet_build = None

		# Path
		self.result_path = config.result_path

		self.device = torch.device('cuda:{0}'.format(config.gpu_id) if config.gpu_id != -1 and torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet_road = U_Net(img_ch=3,output_ch=1)
			self.unet_build = U_Net(img_ch=3,output_ch=1)
		else:
			print("Model Type is Error!")
			
		self.unet_road.to(self.device)
		self.unet_build.to(self.device)

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def test(self, build_model_path, road_model_path):
		"""Test encoder, generator"""
		#===================================== Test ====================================#
		#self.unet_build = torch.jit.load(build_model_path, map_location=self.device)
		#self.unet_road = torch.jit.load(road_model_path, map_location=self.device)
		self.unet_build.load_state_dict(torch.load(build_model_path, map_location=self.device))
		self.unet_road.load_state_dict(torch.load(road_model_path, map_location=self.device))
		print('%s is Successfully Loaded from %s, %s'%(self.model_type, build_model_path, road_model_path))
		self.unet_build.train(False)
		self.unet_road.train(False)
		self.unet_build.eval()
		self.unet_road.eval()

		example = torch.rand(1, 3, 384, 384)
		example = Variable(example.cuda())
		traced_script_module = torch.jit.trace(self.unet_build, (example))
		traced_script_module.save("./Aerial_Building.pt")

		for _, (images, filename) in enumerate(self.test_loader):
			print(filename[0])
			imgOrg = cv2.imread("./images/" + filename[0])
			startTime = time.time()
			imgTensor = images.to(self.device)
			buildTensor = F.sigmoid(self.unet_build(imgTensor))
			roadTensor = F.sigmoid(self.unet_road(imgTensor))
			buildTensor = buildTensor.squeeze(0).squeeze(0)
			roadTensor = roadTensor.squeeze(0).squeeze(0)
			buildTensor = self.to_data(buildTensor)
			roadTensor = self.to_data(roadTensor)
			build_mask = buildTensor.numpy() * 255
			road_mask = roadTensor.numpy() * 255
			print("Use Time: {0:0.3f}s".format(time.time() - startTime))
			build_mask = Image.fromarray(build_mask.astype(np.uint8))
			road_mask = Image.fromarray(road_mask.astype(np.uint8))
			build_mask = np.asarray(build_mask)
			road_mask = np.asarray(road_mask)
			build_mask = build_mask.reshape(build_mask.shape[0], build_mask.shape[1], 1)
			imgOut_Build = getContourAndCalculateArea(imgOrg, build_mask, bin_thread=125, color_contour=(0, 0, 255), text_contour=(0, 255, 0))
			imgOut = getContourAndCalculateArea(imgOut_Build, road_mask, bin_thread=75, color_contour=(255, 0, 0), text_contour=(0, 255, 255))
			cv2.imwrite(self.result_path + "/" + filename[0], imgOut)