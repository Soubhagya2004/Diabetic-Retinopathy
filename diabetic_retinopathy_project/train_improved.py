from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import math

ROOT = Path(__file__).parent
CSV_PATH = ROOT / 'dataset' / 'train.csv'
TRAIN_DIR = ROOT / 'dataset' / 'train_images'
MODEL_DIR = ROOT / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'dr_efficientnetb0.h5'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 5
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']


def load_csv():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f'CSV missing: {CSV_PATH}')
    df = pd.read_csv(CSV_PATH)
    # ensure label column
    if 'id' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'id'})
    paths = []
    labels = []
    for _, row in df.iterrows():
        p = TRAIN_DIR / row['id']
        if not p.exists():
            continue
        paths.append(str(p))
        labels.append(int(row['diagnosis']))
    return np.array(paths), np.array(labels, dtype=np.int32)


def np_preprocess(path_str):
    # path_str is a bytes object from tf.numpy_function
    path = path_str.decode('utf-8') if isinstance(path_str, bytes) else path_str
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # return a zero image if loading failed
        out = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32)
        return out
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # apply CLAHE on the V channel for contrast enhancement
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    except Exception:
        pass

    # circular crop: mask outside central circle to reduce background
    try:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        radius = int(min(center) * 0.95)
        Y, X = np.ogrid[:h, :w]
        dist_from_center = (X - center[0])**2 + (Y - center[1])**2
        mask = dist_from_center <= radius*radius
        # convert to float and apply mask with mean background
        imgf = img.astype(np.float32)
        bg = imgf.mean(axis=(0,1))
        imgf[~mask] = bg
        img = imgf.astype(np.uint8)
    except Exception:
        pass

    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    return arr


def tf_preprocess(path, label, augment=False):
    # wrap numpy preprocessing
    img = tf.numpy_function(np_preprocess, [path], tf.float32)
    img.set_shape([IMG_SIZE[1], IMG_SIZE[0], 3])
    lbl = tf.one_hot(label, NUM_CLASSES)

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        # random rotation ±15 degrees (tensorflow_addons optional)
        if 'tfa' in globals():
            angle = tf.random.uniform([], -15.0, 15.0) * math.pi / 180.0
            img = tfa.image.rotate(img, angles=angle, fill_mode='reflect')
        # random zoom via central crop/resize
        scale = tf.random.uniform([], 0.9, 1.05)
        new_h = tf.cast(scale * IMG_SIZE[1], tf.int32)
        new_w = tf.cast(scale * IMG_SIZE[0], tf.int32)
        img = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
        img = tf.image.resize(img, IMG_SIZE)

    return img, lbl


def build_dataset(paths, labels, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths.astype('str'), labels.astype('int32')))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(lambda p, l: tf_preprocess(p, l, augment=augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_model(input_shape=(224,224,3)):
    inputs = tf.keras.Input(shape=input_shape)
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, base


def main(epochs=20):
    paths, labels = load_csv()
    if len(paths) == 0:
        raise RuntimeError('No images found')

    # stratified split
    train_p, val_p, train_y, val_y = train_test_split(paths, labels, test_size=0.12, stratify=labels, random_state=42)

    # compute class weights to handle imbalance
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
    class_weights = {int(i): float(w) for i, w in enumerate(cw)}
    print('Class weights:', class_weights)

    # build datasets
    try:
        import tensorflow_addons as tfa
        globals()['tfa'] = tfa
    except Exception:
        print('tensorflow_addons not available; rotation augmentation skipped')

    ds_train = build_dataset(train_p, train_y, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    ds_val = build_dataset(val_p, val_y, batch_size=BATCH_SIZE, shuffle=False, augment=False)

    model, base = build_model(input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3))
    base.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor='val_recall', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ]

    print('Training head...')
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # fine-tune: unfreeze top layers of base
    try:
        for layer in base.layers[-40:]:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Recall(name='recall')],
        )
        print('Fine-tuning top layers...')
        history2 = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=math.ceil(epochs/2),
            callbacks=callbacks,
            class_weight=class_weights,
        )
    except Exception as e:
        print('Fine-tune failed:', e)

    # save final model
    model.save(MODEL_PATH)
    print('Saved model to', MODEL_PATH)

    # evaluation on validation set
    X_val = []
    y_val = []
    for p, lbl in zip(val_p, val_y):
        arr = np_preprocess(p)
        X_val.append(arr)
        y_val.append(int(lbl))
    X_val = np.stack(X_val, axis=0)
    preds = model.predict(X_val, batch_size=BATCH_SIZE)
    pred_labels = np.argmax(preds, axis=1)

    cm = confusion_matrix(y_val, pred_labels)
    print('\nConfusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(y_val, pred_labels, target_names=CLASSES, digits=4))


if __name__ == '__main__':
    main(epochs=20)
