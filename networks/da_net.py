import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torchvision.models as models
from networks import all_networks

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator(nn.Module):
	def __init__(self, nout=3, last_layer_activation=True, hidden_in=5120):
		super(Discriminator, self).__init__()
		self.last_layer_activation = last_layer_activation
		self.main = nn.Sequential(
			nn.Conv2d(3,32,3,1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32,32,3,2, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32,64,3,1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64,64,3,2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64,128,3,1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128,128,3,2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128,256,3,1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256,256,3,2, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256,512,3,1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,2, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,2, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512,512,3,2, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			Flatten(),
			nn.Linear(hidden_in,1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024,512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512,nout),
			)
		self.final_act = nn.LogSoftmax(dim=1)
	
	def forward(self, image):
		prob = self.main(image)
		if self.last_layer_activation:
			prob = self.final_act(prob)
		return prob