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

image_location = "Enter File Path Here"
percent = "Enter Estimated Percentage of Outliers Here"

def outliers_Using_Isolation_Forest(location, percentage):
  image_paths = [os.path.join(location, f) for f in os.listdir(location) if f.endswith('jpeg')]
  features = []
  raw_images = []
  
  image_paths = [os.path.join(location, f) for f in os.listdir(location) if f.endswith('jpeg')]
  features = []
  raw_images = []
  
  for path in image_paths:
      img = cv2.imread(path)
      if img is None:
          continue
  
      img = cv2.resize(img, (224, 224))
      img_array = img_to_array(img)
      img_array = preprocess_input(img_array)
      img_array = np.expand_dims(img_array, axis=0)
  
      feat = model.predict(img_array, verbose=0).flatten()
      features.append(feat)
      raw_images.append(path)
  
  features = np.array(features)
  
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


outliers_using_Isolation_Forest(image_location, percent)




