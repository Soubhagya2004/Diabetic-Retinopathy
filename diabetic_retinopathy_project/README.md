# Diabetic Retinopathy Detection

Production-ready, end-to-end example that trains a simple CNN to classify retinal images into 5 APTOS-style stages (0-4), saves the model, and exposes a Flask web app for uploading images and viewing predictions.

Project structure

diabetic_retinopathy_project/
│
├── dataset/
│   ├── train_images/
│   └── train.csv
│
├── model/
│   └── dr_model.h5
│
├── uploads/
│
├── templates/
│   └── index.html
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

How to run

1. Create and activate a Python environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate    # Windows (PowerShell)
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (this will auto-generate 25 synthetic retinal images if dataset is missing):

```bash
python train_model.py
```

This trains for 5 epochs and saves the model to `model/dr_model.h5`.

4. Run the Flask app:

```bash
python app.py
```

Open: http://127.0.0.1:5000

Dataset

- Script expects APTOS-style labels (0-4) in `dataset/train.csv` with columns `id,diagnosis` and images in `dataset/train_images/`.
- If the dataset is not provided, the training script will auto-generate 25 synthetic test images and a CSV so you can run and test locally.

Screenshots

- Screenshot of the web UI after uploading an image (replace with your own):

  - Upload form and prediction result will appear at: http://127.0.0.1:5000

Notes

- The web app will not attempt to load an invalid/empty `dr_model.h5`; run training first.
- The project uses a simple CNN (Conv → Pool → Conv → Pool → Dense) to remain lightweight for local testing.
