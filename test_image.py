import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_3_800.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

start = time.process_time()
out = model(image)
elapsed = (time.process_time() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
print(IMAGE_NAME)
#print(IMAGE_NAME[-4:])
out_img.save('HW4_result/out_srf' + str(UPSCALE_FACTOR) + '_xtraaug_800/' + IMAGE_NAME[-6:-4] + "_pred.png")
