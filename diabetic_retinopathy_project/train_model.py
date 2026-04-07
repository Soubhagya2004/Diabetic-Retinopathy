from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import math
import tensorflow as tf
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except Exception:
    HAS_ALBUMENTATIONS = False
import cv2


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train_images"
CSV_PATH = DATA_DIR / "train.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "dr_model.h5"

# optional archive folder (from provided dataset)
ARCHIVE_COLORED = TRAIN_DIR / "archive" / "colored_images"

IMG_SIZE = (224, 224)
NUM_SYNTHETIC = 25
CLASSES = [0, 1, 2, 3, 4]


def ensure_dirs():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset(n=NUM_SYNTHETIC):
    print(f"Generating {n} synthetic retinal images for testing...")
    rows = []
    for i in range(1, n + 1):
        fn = f"img_{i:04d}.png"
        path = TRAIN_DIR / fn

        # create a synthetic retina-like image
        img = Image.new("RGB", IMG_SIZE, (10, 10, 20))
        draw = ImageDraw.Draw(img)

        # add a central bright circle
        cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
        radius = random.randint(60, 90)
        for r in range(radius, 0, -6):
            color = (
                min(255, 30 + r // 2 + random.randint(0, 30)),
                min(255, 20 + r // 3 + random.randint(0, 30)),
                min(255, 10 + random.randint(0, 20)),
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        # add some blood-vessel-like lines
        for _ in range(random.randint(5, 12)):
            x1 = random.randint(0, IMG_SIZE[0])
            y1 = random.randint(0, IMG_SIZE[1])
            x2 = random.randint(0, IMG_SIZE[0])
            y2 = random.randint(0, IMG_SIZE[1])
            width = random.randint(1, 3)
            draw.line([x1, y1, x2, y2], fill=(120, 10, 10), width=width)

        img.save(path)

        label = random.choice(CLASSES)
        rows.append({"id": fn, "diagnosis": int(label)})

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Synthetic dataset created: {len(rows)} images -> {CSV_PATH}")


def create_csv_from_archive():
    """Scan `archive/colored_images` and create a CSV with relative paths and numeric labels.

    Folder-to-label mapping used here is:
      No_DR -> 0
      Mild -> 1
      Moderate -> 2
      Severe -> 3
      Proliferate_DR -> 4
    """
    if not ARCHIVE_COLORED.exists():
        return False

    mapping = {
        "No_DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferate_DR": 4,
    }

    rows = []
    for class_name, label in mapping.items():
        class_dir = ARCHIVE_COLORED / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.iterdir():
            if img_path.is_file():
                # store relative path from TRAIN_DIR so existing loader works
                rel_path = img_path.relative_to(TRAIN_DIR)
                rows.append({"id": str(rel_path), "diagnosis": int(label)})

    if not rows:
        return False

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV created from archive: {CSV_PATH} ({len(rows)} images)")
    return True


def load_dataset():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = TRAIN_DIR / row["id"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(int(row["diagnosis"]))

    if not images:
        raise RuntimeError("No images found while loading dataset.")

    X = np.stack(images, axis=0)
    y = tf.keras.utils.to_categorical(labels, num_classes=5)
    return X, y


def build_model(input_shape=(224, 224, 3)):
    # Transfer learning with MobileNetV2 as feature extractor
    # Attempt to download ImageNet weights to Keras cache using a certifi-based SSL context
    weights_fname = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    weights_url = (
        "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"
        + weights_fname
    )
    cache_dir = Path.home() / ".keras" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / weights_fname

    if not cache_path.exists():
        print(f"Weights not found in cache. Attempting to download to {cache_path}")
        try:
            # try certifi-based context first
            try:
                import ssl, certifi, urllib.request

                ctx = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(weights_url, context=ctx) as resp, open(cache_path, "wb") as out:
                    out.write(resp.read())
                print("Downloaded weights using certifi CA bundle.")
            except Exception:
                import ssl, urllib.request

                # fallback: disable verification (insecure)
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(weights_url, context=ctx) as resp, open(cache_path, "wb") as out:
                    out.write(resp.read())
                print("Downloaded weights with SSL verification disabled (insecure).")
        except Exception as e:
            print("Warning: failed to download ImageNet weights:", e)

    try:
        # build base without automatic imagenet download, then load weights if available
        base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
        if cache_path.exists():
            try:
                base.load_weights(str(cache_path))
                print("Loaded MobileNetV2 weights from cache.")
            except Exception as e:
                print("Warning: failed to load cached weights, continuing with random init:", e)
        base.trainable = False
    except Exception as e:
        print("Error creating MobileNetV2 base. Falling back to simple random-init base:", e)
        base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
        base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # attach base for easy access when fine-tuning
    model.base_model = base
    return model


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        fl = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return loss


def mixup_generator(generator, alpha=0.2):
    while True:
        x1, y1 = next(generator)
        x2, y2 = next(generator)
        # handle possible mismatched last-batch sizes by trimming to min batch
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        m = min(n1, n2)
        if m == 0:
            continue
        if m < n1:
            x1 = x1[:m]
            y1 = y1[:m]
        if m < n2:
            x2 = x2[:m]
            y2 = y2[:m]
        lam = np.random.beta(alpha, alpha)
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        yield x, y


def aug_batch_generator(X, y, batch_size, augmenter=None, shuffle=True):
    """Yield batches from X,y applying `augmenter` if provided.

    `X` expected in float32 [0,1]. If `augmenter` is an albumentations.Compose,
    images are converted to uint8 0-255 for augmentation and back.
    """
    n = len(X)
    idxs = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idxs)
        for i in range(0, n, batch_size):
            batch_idx = idxs[i : i + batch_size]
            batch_x = X[batch_idx].copy()
            batch_y = y[batch_idx]
            if augmenter is not None and HAS_ALBUMENTATIONS:
                aug_imgs = []
                for img in batch_x:
                    img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                    augmented = augmenter(image=img_u8)
                    aug_img = augmented["image"]
                    aug_imgs.append(aug_img.astype(np.float32) / 255.0)
                batch_x = np.stack(aug_imgs, axis=0)
            elif augmenter is not None and not HAS_ALBUMENTATIONS:
                # fallback: apply simple random cutout and brightness jitter
                for j in range(len(batch_x)):
                    img = batch_x[j]
                    # brightness jitter
                    factor = 1.0 + np.random.uniform(-0.2, 0.2)
                    img = np.clip(img * factor, 0.0, 1.0)
                    # random cutout
                    if np.random.rand() < 0.5:
                        h, w = img.shape[:2]
                        ch = int(h * np.random.uniform(0.1, 0.3))
                        cw = int(w * np.random.uniform(0.1, 0.3))
                        y0 = np.random.randint(0, h - ch + 1)
                        x0 = np.random.randint(0, w - cw + 1)
                        img[y0 : y0 + ch, x0 : x0 + cw, :] = 0.5
                    batch_x[j] = img

            yield batch_x, batch_y


def make_augmenter():
    if not HAS_ALBUMENTATIONS:
        return True
    aug = A.Compose(
        [
            A.RandomResizedCrop(height=IMG_SIZE[1], width=IMG_SIZE[0], scale=(0.8, 1.0), p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.6),
            A.OneOf([A.GaussNoise(var_limit=(10.0, 50.0)), A.ISONoise()], p=0.4),
            A.OneOf([A.MotionBlur(3), A.MedianBlur(3), A.Blur(3)], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Cutout(num_holes=1, max_h_size=int(IMG_SIZE[1] * 0.25), max_w_size=int(IMG_SIZE[0] * 0.25), p=0.4),
        ]
    )
    return aug


def compute_sample_weights_from_class_weights(y_onehot, class_weight):
    y_int = np.argmax(y_onehot, axis=1)
    sample_weights = np.array([float(class_weight[int(c)]) for c in y_int], dtype=np.float32)
    return sample_weights


def _albumentations_augment(img_np, augmenter):
    # img_np: uint8 HWC
    augmented = augmenter(image=img_np)
    return augmented["image"].astype(np.float32) / 255.0


def tf_augment_image(img, augmenter):
    # img: float32 [0,1]
    if HAS_ALBUMENTATIONS and augmenter is not None:
        def _fn(x):
            x_u8 = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
            out = _albumentations_augment(x_u8, augmenter)
            return out

        aug = tf.numpy_function(_fn, [img], tf.float32)
        aug.set_shape([IMG_SIZE[1], IMG_SIZE[0], 3])
        return aug
    else:
        # fallback TF augmentations on [0,1]
        x = img
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        return x


def mixup_batch_tf(batch_x, batch_y, batch_sw, alpha=0.2):
    # batch_x: [B,H,W,C], batch_y: [B,classes], batch_sw: [B]
    B = tf.shape(batch_x)[0]
    # sample beta via two gamma samples
    g1 = tf.random.gamma(shape=(B,), alpha=alpha)
    g2 = tf.random.gamma(shape=(B,), alpha=alpha)
    lam = g1 / (g1 + g2)
    lam_x = tf.reshape(lam, (B, 1, 1, 1))
    lam_y = tf.reshape(lam, (B, 1))

    idx = tf.random.shuffle(tf.range(B))
    x2 = tf.gather(batch_x, idx)
    y2 = tf.gather(batch_y, idx)
    sw2 = tf.gather(batch_sw, idx)

    mixed_x = lam_x * batch_x + (1.0 - lam_x) * x2
    mixed_y = lam_y * batch_y + (1.0 - lam_y) * y2
    mixed_sw = lam * batch_sw + (1.0 - lam) * sw2
    return mixed_x, mixed_y, mixed_sw


def build_tf_dataset(X, y, batch_size, augmenter=None, shuffle=True, mixup=False, class_weight=None):
    # X: float32 [0,1], y: one-hot
    sample_weights = None
    if class_weight is not None:
        sample_weights = compute_sample_weights_from_class_weights(y, class_weight)

    ds = tf.data.Dataset.from_tensor_slices((X, y, sample_weights if sample_weights is not None else np.zeros((X.shape[0],), dtype=np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, X.shape[0]))

    def _map_augment(x, yv, sw):
        x_aug = tf_augment_image(x, augmenter)
        yv = tf.cast(yv, tf.float32)
        sw = tf.cast(sw, tf.float32)
        return x_aug, yv, sw

    ds = ds.map(_map_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if mixup:
        def _map_mixup(bx, by, bsw):
            mx, my, msw = mixup_batch_tf(bx, by, bsw)
            return mx, my, msw

        ds = ds.map(_map_mixup, num_parallel_calls=tf.data.AUTOTUNE)

    # For training, yield (x,y, sample_weight); for validation, sample_weight can be ignored
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main(epochs=5, use_array_fit=False, use_dataset=False):
    ensure_dirs()

    # If an `archive/colored_images` dataset exists, create CSV from it
    if ARCHIVE_COLORED.exists():
        created = create_csv_from_archive()
        if not created:
            print("Found archive folder but no images were added to CSV.")
    # use_array_fit is passed as parameter

    # If dataset missing or CSV empty, create synthetic
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0 or len(list(TRAIN_DIR.iterdir())) == 0:
        generate_synthetic_dataset()

    print("Loading dataset...")
    try:
        X, y = load_dataset()
    except Exception as e:
        print("Dataset load failed:", e)
        print("Generating synthetic dataset and reloading...")
        generate_synthetic_dataset()
        X, y = load_dataset()

    print(f"Dataset loaded: X={X.shape}, y={y.shape}")

    # compute integer labels and class weights to address imbalance
    y_int = np.argmax(y, axis=1)
    total = len(y_int)
    class_counts = np.bincount(y_int, minlength=5)
    class_weight = {}
    for i, c in enumerate(class_counts):
        if c <= 0:
            class_weight[i] = 1.0
        else:
            class_weight[i] = float(total) / (len(class_counts) * c)

    print("Class distribution:")
    for i, c in enumerate(class_counts):
        print(f"  class {i}: count={c}, weight={class_weight[i]:.3f}")

    # Stratified train/val split
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, stratify=y_int, random_state=42
        )
    except Exception:
        # fallback: simple split
        n_val = max(1, int(0.1 * len(X)))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

    batch_size = 32
    # build augmenter (albumentations if available, otherwise simple fallback)
    augmenter = make_augmenter()

    print(f"use_array_fit={use_array_fit}, use_dataset={use_dataset}")

    # Oversample minority classes by repeating samples until all classes match the max count
    y_train_int = np.argmax(y_train, axis=1)
    unique, counts = np.unique(y_train_int, return_counts=True)
    max_count = int(counts.max())
    resampled_X = []
    resampled_y = []
    for cls in range(y.shape[1]):
        cls_idx = np.where(y_train_int == cls)[0]
        if len(cls_idx) == 0:
            continue
        # how many samples to add
        n_needed = max_count - len(cls_idx)
        # sample with replacement
        if n_needed > 0:
            add_idx = np.random.choice(cls_idx, size=n_needed, replace=True)
            all_idx = np.concatenate([cls_idx, add_idx])
        else:
            all_idx = cls_idx
        resampled_X.append(X_train[all_idx])
        resampled_y.append(y_train[all_idx])

    X_resampled = np.concatenate(resampled_X, axis=0)
    y_resampled = np.concatenate(resampled_y, axis=0)

    # shuffle resampled dataset
    perm = np.random.permutation(len(X_resampled))
    X_resampled = X_resampled[perm]
    y_resampled = y_resampled[perm]

    # use augmented batch generator (supports albumentations or simple fallback)
    # Build training input depending on mode
    if use_dataset:
        ds_train = build_tf_dataset(X_resampled, y_resampled, batch_size=batch_size, augmenter=augmenter, shuffle=True, mixup=True, class_weight=class_weight)
        ds_val = build_tf_dataset(X_val, y_val, batch_size=batch_size, augmenter=None, shuffle=False, mixup=False, class_weight=None)
        train_source = 'dataset'
    else:
        if not use_array_fit:
            train_gen = aug_batch_generator(X_resampled, y_resampled, batch_size=batch_size, augmenter=augmenter)
            val_gen = aug_batch_generator(X_val, y_val, batch_size=batch_size, augmenter=None, shuffle=False)
        else:
            train_gen = None
            val_gen = None
        ds_train = None
        ds_val = None
        train_source = 'array' if use_array_fit else 'generator'

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # If a model exists, load weights and continue training
    try:
        if MODEL_PATH.exists():
            model.load_weights(str(MODEL_PATH))
            print(f"Loaded existing weights from {MODEL_PATH}, continuing training.")
    except Exception as e:
        print("Could not load existing weights, starting from scratch:", e)
    print(model.summary())

    # recompile with focal loss and a small LR (ensure compiled with focal loss before training)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    # prepare mixup generator or array-based training
    steps_per_epoch = math.ceil(len(X_resampled) / batch_size)
    val_steps = math.ceil(len(X_val) / batch_size)

    initial_epochs = min(5, epochs)
    print(f"Phase 1: training head for {initial_epochs} epochs...")

    if use_dataset:
        print("Training from tf.data dataset with sample weights and mixup...")
        history1 = model.fit(
            ds_train,
            epochs=initial_epochs,
            validation_data=ds_val,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
        )
    elif use_array_fit:
        history1 = model.fit(
            X_resampled,
            y_resampled,
            batch_size=batch_size,
            epochs=initial_epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            shuffle=True,
        )
    else:
        mixup_gen = mixup_generator(train_gen, alpha=0.2)
        try:
            history1 = model.fit(
                mixup_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=initial_epochs,
                validation_data=val_gen,
                validation_steps=val_steps,
                callbacks=callbacks,
                class_weight=class_weight,
            )
        except ValueError as e:
            print("class_weight unsupported for generator input, retrying without class_weight:", e)
            history1 = model.fit(
                mixup_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=initial_epochs,
                validation_data=val_gen,
                validation_steps=val_steps,
                callbacks=callbacks,
            )

    # Phase 2: fine-tune top layers if more epochs requested
    history2 = None
    if epochs > initial_epochs:
        fine_tune_epochs = epochs - initial_epochs
        # unfreeze top layers of the base model
        try:
            base = model.base_model
            for layer in base.layers[:-100]:
                layer.trainable = False
            for layer in base.layers[-100:]:
                layer.trainable = True
            print("Unfroze top 100 layers of base model for fine-tuning.")
        except Exception:
            try:
                model.base_model.trainable = True
                print("Unfroze entire base model for fine-tuning.")
            except Exception:
                print("Could not unfreeze base model layers; skipping fine-tune phase.")

        # lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=["accuracy"],
        )

        print(f"Phase 2: fine-tuning for {fine_tune_epochs} epochs...")
        if use_dataset:
            history2 = model.fit(
                ds_train,
                epochs=fine_tune_epochs,
                validation_data=ds_val,
                callbacks=callbacks,
                initial_epoch=initial_epochs,
                steps_per_epoch=steps_per_epoch,
            )
        elif use_array_fit:
            history2 = model.fit(
                X_resampled,
                y_resampled,
                batch_size=batch_size,
                epochs=fine_tune_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weight,
                shuffle=True,
            )
        else:
            try:
                history2 = model.fit(
                    mixup_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=fine_tune_epochs,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    callbacks=callbacks,
                    class_weight=class_weight,
                )
            except ValueError as e:
                print("class_weight unsupported for generator input in fine-tune, retrying without class_weight:", e)
                history2 = model.fit(
                    mixup_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=fine_tune_epochs,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    callbacks=callbacks,
                )

    # Save final model
    try:
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print("Could not save model via model.save(), attempting save_weights():", e)
        try:
            model.save_weights(str(MODEL_PATH))
            print(f"Weights saved to {MODEL_PATH}")
        except Exception as e2:
            print("Failed to save weights:", e2)

    # Final evaluation on full dataset
    try:
        X_all, y_all = load_dataset()
        eval_res = model.evaluate(X_all, y_all, batch_size=batch_size, verbose=1)
        print("Final evaluation on full dataset:", eval_res)
    except Exception as e:
        print("Could not evaluate on full dataset:", e)


if __name__ == "__main__":
    main(epochs=20)
