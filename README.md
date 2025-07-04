# Outlier-Detection
The Outlier Detection Notebooks use two prominent algorithms - Isolation Forest and Local Outlier Factor - to detect outliers in a contaminated dataset. Both models are suitable for identifying mislabeled or rare samples when the majority of data is presumed to be clean.

## Configurable Outlier Percentage
Both notebooks include an input cell where the user can define the estimated percentage of outliers in the dataset. This value is passed to the `contamination` parameter of the respective models:

  * In Isolation Forest, it defines the expected proportion of anomalies in the dataset.

  * In Local Outlier Factor, it informs the threshold for determining which points are considered outliers.

## How the Code Works
Each notebook is designed to perform unsupervised outlier detection by leveraging visual features extracted from images using a pretrained Convolutional Neural Network (CNN). The core steps of the workflow are outlined below:

### 1. Image Loading
The notebook reads all image files from a user-specified directory. It filters for supported file formats (jpeg) and loads them using OpenCV.

```python
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.jpeg')]
```

### 2. Feature Extraction using CNN
Each image is resized and passed through a pretrained CNN model — such as MobileNetV2 — to extract high-level feature embeddings. These features capture essential visual characteristics such as texture, shape, and color distribution.

The CNN model used is MobileNetV2 (with weights pretrained on ImageNet), and the output is the global average pooled embedding from the final convolutional layers.

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)
```

The extracted features are then flattened and stored for anomaly detection.

### 3. (Optional) Dimensionality Reduction
To improve runtime and reduce noise in high-dimensional feature vectors, Principal Component Analysis (PCA) is applied to reduce the features to a manageable number of dimensions (commonly 30–50).

```python
pca = PCA(n_components=50)
X_pca = pca.fit_transform(features)
```

### 4. Outlier Detection Using Isolation Forest or LOF
Isolation Forest:
Learns a model to isolate anomalies by randomly selecting features and split values.

Samples that are easier to isolate (i.e., require fewer splits) are more likely to be outliers.

```python
clf = LocalOutlierFactor(n_neighbors=20, contamination= contamination/100)
preds = clf.fit_predict(X_pca)
```

Local Outlier Factor (LOF):
Measures the local density deviation of a given sample compared to its neighbors.

Points in regions of lower density than their neighbors are flagged as outliers.

```python
clf = LocalOutlierFactor(n_neighbors=20, contamination=user_contamination)
preds = clf.fit_predict(X_pca)
outlier_indices = np.where(preds == -1)[0]
```

### 5. Result Interpretation and Visualization
Detected outliers are returned as indices or file paths.
```python
import math

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
```


The notebook visualizes a subset of flagged outlier images using matplotlib for user inspection.

