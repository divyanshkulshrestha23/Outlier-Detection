import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import math

location1 = "Enter File Path Here"
location2 = "Enter File Path Here"
percent = "Enter Estimated Percentage of Outliers Here"

def YOLO_Outliers_Isolation_Forest(image_location, labels_location, percentage):

  base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
  model = Model(inputs=base_model.input, outputs=base_model.output)
  
  patch_features = []
  patch_info = []
  
  for filename in os.listdir(image_location):
      if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
          continue
  
      img_path = os.path.join(image_location, filename)
      label_path = os.path.join(label_location, filename.rsplit('.', 1)[0] + '.txt')
  
      if not os.path.exists(label_path):
          continue
  
      img = cv2.imread(img_path)
      if img is None:
          continue
      h, w = img.shape[:2]
  
      with open(label_path, 'r') as f:
          for line in f:
              parts = line.strip().split()
              if len(parts) != 5:
                  continue
              _, cx, cy, bw, bh = map(float, parts)
              x1 = int((cx - bw / 2) * w)
              y1 = int((cy - bh / 2) * h)
              x2 = int((cx + bw / 2) * w)
              y2 = int((cy + bh / 2) * h)
  
              x1, y1 = max(0, x1), max(0, y1)
              x2, y2 = min(w, x2), min(h, y2)
  
              patch = img[y1:y2, x1:x2]
              if patch.shape[0] < min_patch_size or patch.shape[1] < min_patch_size:
                  continue
  
              patch_resized = cv2.resize(patch, resize_dim)
              patch_array = preprocess_input(img_to_array(patch_resized))
              patch_array = np.expand_dims(patch_array, axis=0)
              features = cnn_model.predict(patch_array, verbose=0).flatten()
  
              patch_features.append(features)
              patch_info.append((img_path, (x1, y1, x2, y2)))
  
  features = np.array(patch_features)
  
  pca = PCA(n_components=50)
  X_pca = pca.fit_transform(features)
  
  clf = IsolationForest(contamination= percentage/100, random_state=42)  
  preds = clf.fit_predict(X_pca)
  outlier_indices = np.where(preds == -1)[0]
  
  num_outliers = len(outlier_indices)
  cols = 5  # Number of images per row
  rows = math.ceil(num_outliers / cols)
  
  plt.figure(figsize=(cols * 3, rows * 3))  # Scale size by number of rows/cols
  
  for i, idx in enumerate(outlier_indices):
      img = cv2.imread(raw_images[idx])
      img = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB)
      plt.subplot(rows, cols, i + 1)
      plt.imshow(img)
      plt.title(f"Outlier #{i + 1}")
      plt.axis('off')
  
  plt.suptitle("All Detected Outlier Images", fontsize=16)
  plt.tight_layout()
  plt.show()


YOLO_Outliers_Isolation_Forest(location1, location2, percent)

