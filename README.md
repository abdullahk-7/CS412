# 🎬 MovieLens 1M — Deep Learning Recommender Playground

Welcome to a hands-on, research-friendly environment for exploring deep learning-based recommender systems. This repository enables rapid prototyping, benchmarking, and enhancement of collaborative filtering models using the MovieLens 1M dataset.
## 🚀 Quick Start Guide
	1.	Install Required Packages
   pip install -r requirements.txt
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2.	Download the Dataset
	•	Download MovieLens 1M from GroupLens
	•	Place the ratings.dat file inside:(https://grouplens.org/datasets/movielens/1m/)
   - Place `ratings.dat` in `data/movielens_1m/`
	3.	Run an Experiment
On macOS/Linux:
python src/evaluate.py
On Windows:
run_experiment.bat
---
🧩 Project Components
	•	Data Preprocessing:
src/data_loader.py – Loads and preprocesses the MovieLens data for training-ready format.
	•	Model Architectures:
	•	src/models/ncf_mlp.py – Neural Collaborative Filtering (MLP-based)
	•	src/models/autoencoder.py – Denoising Autoencoder for recommendation tasks
	•	Training Engine:
src/train.py – Core logic for model training and loss tracking.
	•	Experiment Pipeline:
src/evaluate.py – Full pipeline to load data, train models, and evaluate performance.
## 📁 Project Structure
```
CS_412A3/
├── data/
│   └── movielens_1m/
│       └── ratings.dat
├── results/
│   └── results.txt
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── train.py
│   └── models/
│       ├── ncf_mlp.py
│       └── autoencoder.py
├── requirements.txt
└── run_experiment.bat
🔍 What You Can Explore
	•	Compare Models:
Benchmark different architectures like NCF and autoencoder.
	•	Expand Functionality:
Easily integrate your custom models or loss functions.
	•	Inspect Results:
MAE scores are stored in results/results.txt for quick evaluation.
	•	Learn by Doing:
Clean, well-documented code designed for educational and research exploration.
📊 Sample Output

Upon completion, results are logged in results/results.txt:
Model         MAE
NCF_MLP       0.XXXX
Autoencoder   0.XXXX
```

---

🛠️ Customize & Extend

Here are some ideas to enhance the playground:
	•	Implement new architectures (e.g., Matrix Factorization, GNNs)
	•	Apply dropout, weight decay, or custom regularization
	•	Experiment with learning rates and optimizers
	•	Add alternative metrics like RMSE or Precision@K
---

🧠 Common Issues
	•	Missing Data?
Ensure ratings.dat is correctly placed in data/movielens_1m/
	•	Unexpected Index Errors?
User and item IDs are remapped internally—double-check dataset integrity.
	•	No GPU?
The framework gracefully falls back to CPU.

🤝 Acknowledgements
	•	MovieLens 1M Dataset by GroupLens
	•	Built using PyTorch, pandas, and scikit-learn

This repository is designed for learning and experimentation. Feel free to fork it, modify it, and make it your own!


