import torch
import torchvision

img = torchvision.io.read_image('./asset/slice-result/input.jpg')

channel, height, width = img.shape
eyes = torch.zeros(channel, height, width,dtype=torch.uint8)
face = torch.zeros(channel, height, width,dtype=torch.uint8)
mouth = torch.zeros(channel, height, width,dtype=torch.uint8)
nose = torch.zeros(channel, height, width,dtype=torch.uint8)

# eyes
x, y =  50,102
width, height = 146, 46
eyes[:, y:y+height, x:x+width] = img[:, y:y+height, x:x+width]
torchvision.io.write_jpeg(eyes, './asset/slice-result/eyes.jpg')

# face
x, y =  50,102
width, height = 146, 102
face[:, y:y+height, x:x+width] = img[:, y:y+height, x:x+width]
torchvision.io.write_jpeg(face, './asset/slice-result/face.jpg')

# mouth
x, y =  50,174
width, height = 146, 46
mouth[:, y:y+height, x:x+width] = img[:, y:y+height, x:x+width]
torchvision.io.write_jpeg(mouth, './asset/slice-result/mouth.jpg')

# nose
x, y =  86,134
width, height = 86, 46
nose[:, y:y+height, x:x+width] = img[:, y:y+height, x:x+width]
torchvision.io.write_jpeg(nose, './asset/slice-result/nose.jpg')
