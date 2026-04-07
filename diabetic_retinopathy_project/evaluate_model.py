import sys
import numpy as np
from train_model import load_dataset
from sklearn.model_selection import train_test_split
from app import load_model_if_available, CLASSES
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    X, y = load_dataset()
    y_int = np.argmax(y, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y_int, random_state=42
    )

    model = load_model_if_available()
    if model is None:
        print("ERROR: model not available")
        sys.exit(2)

    print("Running predictions on validation set...")
    preds = model.predict(X_val, batch_size=32, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_val, axis=1)

    acc = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(true_labels, pred_labels, target_names=CLASSES, digits=4))
    print("\nConfusion matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)

    # show per-class counts
    unique, counts = np.unique(true_labels, return_counts=True)
    print('\nValidation class counts:')
    for u, c in zip(unique, counts):
        print(f"  class {u} ({CLASSES[u]}): {c}")


if __name__ == '__main__':
    main()
