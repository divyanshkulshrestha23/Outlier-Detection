# Outlier-Detection
The Outlier Detection Notebooks use two prominent algorithms - Isolation Forest and Local Outlier Factor - to detect outliers in a contaminated dataset. Both models are suitable for identifying mislabeled or rare samples when the majority of data is presumed to be clean.

## Configurable Outlier Percentage
Both notebooks include an input cell where the user can define the estimated `percentage` of outliers in the dataset. This value is passed to the `contamination` parameter of the respective models:

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
clf = IsolationForest(contamination= percentage/100, random_state=42)
preds = clf.fit_predict(X_pca)
outlier_indices = np.where(preds == -1)[0]
```

Local Outlier Factor (LOF):
Measures the local density deviation of a given sample compared to its neighbors.

Points in regions of lower density than their neighbors are flagged as outliers.

```python
lof = LocalOutlierFactor(n_neighbors=20, contamination= percentage / 100)
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

# Annotation Verification

If you have YOLO-format annotation files, the other two files can work with that data too.

## How YOLO Patch Extraction Works

Patches from full-sized images using YOLO-format annotation files serve as the input units for anomaly detection, allowing the model to identify suspicious or unusual object instances, rather than evaluating entire images.

### 1. Image and Annotation Matching
The script first scans the images/ and labels/ directories and matches image files (e.g., sample1.jpg) with their corresponding annotation files (e.g., sample1.txt). Each annotation file is expected to follow the YOLO format, where each line represents a single object using five fields:

```php
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized between 0 and 1 with respect to the image dimensions.

### 2. Conversion to Pixel Coordinates
For each object in the annotation file, the normalized bounding box coordinates are converted into actual pixel values using the image’s width and height. This step calculates the top-left (x1, y1) and bottom-right (x2, y2) corners of the bounding box:

```python
x1 = int((cx - bw / 2) * image_width)
y1 = int((cy - bh / 2) * image_height)
x2 = int((cx + bw / 2) * image_width)
y2 = int((cy + bh / 2) * image_height)
```

### 3. Patch Extraction and Filtering
Using the computed coordinates, the image is sliced to extract a rectangular patch corresponding to the object. Patches that are too small (e.g., less than 10×10 pixels) are discarded, as they typically do not contain meaningful visual information and may introduce noise into the model.

Each valid patch is then resized to a standard input size (e.g., 224×224 pixels) to match the expected dimensions for CNN feature extraction.

### 4. Preprocessing for CNN Feature Extraction
Each resized patch undergoes preprocessing using preprocess_input() from Keras (specific to the CNN model being used, such as MobileNetV2). This ensures pixel values are properly scaled and formatted. The preprocessed patch is converted into a NumPy array and passed to the CNN model to extract high-level feature representations.


### 5. Metadata Tracking
For every patch extracted, the script stores metadata including the image path and bounding box coordinates. This information is used later to match detected outlier patches back to their original image and location and visualize the patch in the final output.

```python
patch = img[y1:y2, x1:x2]
if patch.shape[0] < min_patch_size or patch.shape[1] < min_patch_size:
  continue
patch_resized = cv2.resize(patch, resize_dim)
patch_array = preprocess_input(img_to_array(patch_resized))
patch_array = np.expand_dims(patch_array, axis=0)
features = cnn_model.predict(patch_array, verbose=0).flatten()
patch_features.append(features)
patch_info.append((img_path, (x1, y1, x2, y2)))
```

