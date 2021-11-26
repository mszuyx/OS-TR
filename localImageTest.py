import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
# import onnx
# print(onnx.__version__)

ref_path_ = '/home/ros/OS_TR/ref_4.jpg'
query_path_ = '/home/ros/OS_TR/query_2.jpg'
model_path_ = '/home/ros/OS_TR/log/dtd_dtd_weighted_bce_banded_0.001/snapshot-epoch_2021-11-25-16:42:11_texture.pth'
model = torch.load(model_path_)
model.eval()

ref_img = np.asarray(Image.open(ref_path_).convert('RGB').resize((256,256)))/255.0 #.convert('RGB')
query_img = np.asarray(Image.open(query_path_).resize((256,256)))/255.0
ref_tensor = (torch.from_numpy(ref_img).permute(2,0,1)).unsqueeze(0)
query_tensor = (torch.from_numpy(query_img).permute(2,0,1)).unsqueeze(0)

# ref_out = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
# query_out = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('/home/ros/OS_TR/ref_2.jpg', ref_out)
# cv2.imwrite('/home/ros/OS_TR/query_2.jpg', query_out)

transform_zk = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667))
        ])
ref_tensor = transform_zk(ref_img).unsqueeze(0).float()
query_tensor = transform_zk(query_img).unsqueeze(0).float()

if torch.cuda.is_available():
    query_tensor = query_tensor.cuda()
    ref_tensor = ref_tensor.cuda()

scores = model(query_tensor, ref_tensor)
# scores[scores >= 0.5] = 1
# scores[scores < 0.5] = 0
seg = scores[0, 0, :, :]#.long()
pred = seg.data.cpu().numpy()

fig = plt.figure(0)
ax = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(query_tensor[0].permute(1, 2, 0).data.cpu().numpy())
ax.set_title('Query')
ax.axis('off')
ax = fig.add_subplot(1, 3, 2)
imgplot = plt.imshow(ref_tensor[0].permute(1, 2, 0).data.cpu().numpy())
ax.set_title('Reference')
ax.axis('off')
ax = fig.add_subplot(1, 3, 3)
imgplot = plt.imshow(pred)
ax.set_title('Prediction')
ax.axis('off')

plt.figure(1)
plt.imshow(query_img)
plt.imshow(pred, alpha=0.5, cmap=plt.get_cmap("RdBu"))
plt.show()


