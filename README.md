# Severity-Assessment-and-Classification-of-Skin-Disease
A hybrid framework for **skin disease severity classification** using EfficientNet-B0 for lesion segmentation and feature extraction. Grad-CAM improves interpretability by highlighting critical regions, while a Random Forest classifier enables accurate and reliable severity assessment for clinical decision support.

#Python Code

#Finding the Optimal Deep Feature Extractor Using EfficientNet- 
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2 

# Load EfficientNet backbone
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3)) 
x = GlobalAveragePooling2D()(base.output) x = Dropout(0.3)(x)
output = Dense(9, activation="softmax")(x) 
model_eff =
model_eff.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) 

# Train the model
train_gen = ImageDataGenerator(rescale=1/255.) train_ds = train_gen.flow_from_directory( 
"skin_lesion_processed/train", 
target_size=(224,224), batch_size=32, class_mode="categorical" 
) 
val_ds = train_gen.flow_from_directory(
18 
 
Model(inputs= 	base.input, outputs=output) 

"skin_lesion_processed/val", 
target_size=(224,224), batch_size=32, class_mode="categorical" 
)
model_eff.fit(train_ds, validation_data=val_ds, epochs=20) 
model_eff.save("best_efficientnet.keras")
feature_model = Model(inputs=model_eff.input, outputs=model_eff.layers[-2].output) 
data_path = "skin_lesion_processed/train" rows = [] 
for label in os.listdir(data_path):
class_path = os.path.join(data_path, label) 
for img_name in os.listdir(class_path):
img_path = os.path.join(class_path, img_name) 
img = cv2.imread(img_path) 
img = cv2.resize(img, (224,224)) img = img.astype("float32") / 255.0 
feat = feature_model.predict(np.expand_dims(img, axis=0))[0] 
row = list(feat) + [label] rows.append(row) 
df = pd.DataFrame(rows) df.to_csv("deep_features.csv", index=False) 
4.2.2 Finding the Region of Interest (ROI) Through Image Segmentation- 
import cv2
import numpy as np 
def segment_roi(img): 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) blur = cv2.GaussianBlur(gray, (5,5), 0) 
_, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
white_ratio = np.mean(mask == 255) if white_ratio < 0.5: 
mask = 255 - mask 
kernel = np.ones((7,7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
return mask 
4.2.3 Finding the Changes in Skin Lesion Attributes – 
def extract_features(img, mask): roi_pixels = img[mask == 255] 
mean_color = np.mean(roi_pixels, axis=0) area = np.sum(mask == 255) 
ys, xs = np.where(mask == 255) if len(xs) > 0: 
width = xs.max() - xs.min()
height = ys.max() - ys.min() aspect_ratio = width / (height + 1e-6) 
else:
aspect_ratio = 0 
return [area, mean_color[0], mean_color[1], mean_color[2], aspect_ratio] 

# Finding the Most Suitable Classifier- 
import pandas as pd
from sklearn.model_selection import train_test_split from sklearn.preprocessing import LabelEncoder from xgboost import XGBClassifier
import joblib 
df = pd.read_csv("deep_features.csv") X = df.drop("label", axis=1) 
y = df["label"] 
le = LabelEncoder()
y_encoded = le.fit_transform(y) 
X_train, X_test, y_train, y_test = train_test_split( 
X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded ) 
model_xgb = XGBClassifier( objective="multi:softmax", num_class=9, n_estimators=500, learning_rate=0.05, max_depth=6 
)
model_xgb.fit(X_train, y_train) joblib.dump(model_xgb, "xgb_classifier.pkl") 
joblib.dump(le, "label_encoder.pkl") 
4.2.5 Finding the Interpretability of Predictions- 
import tensorflow as tf import numpy as np import cv2 
def grad_cam(model, img_array, layer_name="top_conv"): 
grad_model = tf.keras.models.Model(
[model.inputs], [model.get_layer(layer_name).output, model.output] 
) 
with tf.GradientTape() as tape:
conv_outputs, predictions = grad_model(img_array) class_index = tf.argmax(predictions[0])
loss = predictions[:, class_index] 
grads = tape.gradient(loss, conv_outputs)[0] conv_outputs = conv_outputs[0] 
weights = tf.reduce_mean(grads, axis=(0,1)) cam = np.zeros(conv_outputs.shape[:2]) 
for i, w in enumerate(weights): cam += w * conv_outputs[:,:,i] 
cam = cv2.resize(cam.numpy(), (224,224)) cam = np.maximum(cam, 0)
cam = cam / cam.max() 
return cam 

#Finding the Model’s Behaviour Using Single-Image Prediction- 
import matplotlib.pyplot as plt 
model = joblib.load("xgb_classifier.pkl")
le = joblib.load("label_encoder.pkl")
eff_model = tf.keras.models.load_model("best_efficientnet.keras") 
def predict_and_show(img_path, true_label=None): 
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# ROI + hand-crafted features
mask = segment_roi(img_rgb)
physical = extract_features(img_rgb, mask) 
# Deep features
img_resized = cv2.resize(img_rgb, (224,224)) / 255.0
deep_feat = feature_model.predict(np.expand_dims(img_resized, axis=0))[0] 
# Combine 1280 + 5
total_features = np.hstack([deep_feat, physical]).reshape(1, -1) 
pred_class = model.predict(total_features)[0] pred_label = le.inverse_transform([pred_class])[0] 
# Grad-CAM
cam = grad_cam(eff_model, np.expand_dims(img_resized, axis=0))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) overlay = cv2.addWeighted(img_resized, 0.5, heatmap/255.0, 0.5, 0) 
# Display plt.figure(figsize=(10,5)) 
 
plt.subplot(1,2,1) plt.imshow(img_rgb) plt.title(f"Actual: {true_label}") plt.axis("off") 
plt.subplot(1,2,2) plt.imshow(overlay) plt.title(f"Predicted: {pred_label}") plt.axis("off") 
plt.show() 
<img width="468" height="646" alt="image" src="https://github.com/user-attachments/assets/e6baca96-f896-4528-a7da-c1397c0224c8" />
