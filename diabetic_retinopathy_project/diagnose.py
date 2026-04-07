import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).parent
CSV = ROOT/'dataset'/'train.csv'
TRAIN_DIR = ROOT/'dataset'/'train_images'
MODEL_PATH = ROOT/'model'/'dr_model.h5'
IMG_SIZE=(224,224)
CLASSES=['No DR','Mild','Moderate','Severe','Proliferative']

if not CSV.exists():
    print('CSV missing:',CSV)
    raise SystemExit

df=pd.read_csv(CSV)
filenames=[]
labels=[]
for _,row in df.iterrows():
    p=TRAIN_DIR/row['id']
    if not p.exists():
        continue
    filenames.append(str(p))
    labels.append(int(row['diagnosis']))

X=[]
for fp in filenames:
    img=Image.open(fp).convert('RGB').resize(IMG_SIZE)
    arr=np.asarray(img,dtype=np.float32)/255.0
    X.append(arr)
X=np.stack(X,axis=0)
y=np.array(labels)

# stratified split
train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=42)
X_val=X[val_idx]
y_val=y[val_idx]
filenames_val=[filenames[i] for i in val_idx]

# load model
try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print('Loaded model from', MODEL_PATH)
except Exception as e:
    print('Failed to load model directly:', e)
    import train_model as tm
    model = tm.build_model(input_shape=(224,224,3))
    model.load_weights(str(MODEL_PATH))
    print('Loaded weights into new model')

# predict
preds = model.predict(X_val, batch_size=32)
pred_labels = np.argmax(preds, axis=1)

# confusion matrix
cm = confusion_matrix(y_val, pred_labels)
print('Confusion Matrix:')
print(cm)
print('\nClassification Report:')
print(classification_report(y_val, pred_labels, target_names=CLASSES, digits=4))

# top misclassified Proliferative (true label 4)
mis_idx = [i for i,(t,p) in enumerate(zip(y_val,pred_labels)) if t==4 and p!=4]
if not mis_idx:
    print('No misclassified Proliferative images in validation set')
else:
    records=[]
    for i in mis_idx:
        conf=float(preds[i][pred_labels[i]])
        records.append((conf, filenames_val[i], y_val[i], pred_labels[i]))
    records.sort(reverse=True)
    print('\nTop 10 misclassified Proliferative_DR images (by predicted-class confidence):')
    for conf, fname, true, pred in records[:10]:
        print(f'{Path(fname).name}  true={CLASSES[true]} pred={CLASSES[pred]} conf={conf:.4f} path={fname}')
