import os
import tifffile
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from skimage import transform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def load_and_resize_kidney_images(image_folder_path, target_size=(128, 128)):
    image_files = sorted([file for file in os.listdir(image_folder_path) if file.endswith(".tif")])
    image_data = []
    for file in image_files:
        image = tifffile.imread(os.path.join(image_folder_path, file)).astype(np.float32)
        resized_image = transform.resize(image, target_size, mode='constant')
        image_data.append(resized_image)
    return torch.from_numpy(np.stack(image_data))

image_folder_path = r'/kaggle/input/blood-vessel-segmentation/train/kidney_3_sparse/images'
data = load_and_resize_kidney_images(image_folder_path, target_size=(128, 128))
data = data.unsqueeze(1).float()
labels = torch.from_numpy(np.random.randint(0, 2, size=data.shape)).float()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = data.to(device)
labels = labels.to(device)

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True)

val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, pin_memory=True)

labels_folder_path = r'/kaggle/input/blood-vessel-segmentation/train/kidney_3_sparse/labels'
label_images = []
for file in sorted(os.listdir(labels_folder_path)):
    if file.endswith(".tif"):
        label_image = tifffile.imread(os.path.join(labels_folder_path, file))
        label_images.append(torch.from_numpy(label_image).float())

batch_size = 1000
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.grid(False)

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.zaxis.label.set_color('black')

ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.tick_params(axis='z', colors='black')

for batch_start in range(0, len(label_images), batch_size):
    predicted_labels_batch = torch.stack(label_images[batch_start:batch_start + batch_size], dim=0)
    predicted_labels_batch = predicted_labels_batch.to(device)
    
    for i in range(predicted_labels_batch.shape[0]):
        non_zero_coords = predicted_labels_batch[i].nonzero()
        
        if non_zero_coords.numel() > 0:
            if non_zero_coords.dim() == 2:
                x, y = non_zero_coords.unbind(1)
                z = torch.full_like(x, fill_value=i + batch_start)
            elif non_zero_coords.dim() == 1:
                x, y = non_zero_coords.unbind(0)
                z = torch.full_like(x, fill_value=i + batch_start)
            else:
                x, y, z = non_zero_coords.unbind(1)
            
            ax.scatter(x.cpu(), y.cpu(), z.cpu(), c='blue', marker='o', s=2, alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Combined 3D Segmentation Result for All Images')

plt.ion()
plt.show()
