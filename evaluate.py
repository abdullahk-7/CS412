import os
import sys
from typing import Dict

import torch
import pandas as pd

# Add parent directory to Python path for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import prepare_movielens_loaders
from src.models.ncf_mlp import NCF_MLP
from src.models.autoencoder import AutoencoderCF
from src.train import fit_ncf, fit_autoencoder


def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                   model_type: str, n_users: int, n_movies: int, device: str = 'cpu') -> float:
    """
    Evaluates a given model by computing Mean Absolute Error (MAE).
    """
    model.eval()
    total_error, count = 0.0, 0

    with torch.no_grad():
        if model_type == 'ncf':
            for users, items, ratings in loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                predictions = model(users, items)
                total_error += torch.abs(predictions - ratings).sum().item()
                count += len(ratings)

        elif model_type == 'autoencoder':
            user_matrix = torch.zeros(n_users, n_movies).to(device)
            for user, item, rating in loader.dataset:
                user_matrix[user, item] = rating

            reconstructed = model(user_matrix)

            for users, items, ratings in loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                predictions = reconstructed[users, items]
                total_error += torch.abs(predictions - ratings).sum().item()
                count += len(ratings)

    return total_error / count if count > 0 else float('nan')


def save_results(results: Dict[str, float], filepath: str = 'results/results.txt') -> None:
    """
    Saves evaluation metrics to a results file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write('Model\tMAE\n')
        for model_name, mae in results.items():
            f.write(f'{model_name}\t{mae:.4f}\n')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, test_loader, n_users, n_movies = prepare_movielens_loaders()

    # Train & evaluate NCF model
    ncf_model = NCF_MLP(n_users, n_movies).to(device)
    ncf_model = fit_ncf(ncf_model, train_loader, device=device)
    ncf_mae = evaluate_model(ncf_model, test_loader, 'ncf', n_users, n_movies, device=device)

    # Train & evaluate Autoencoder model
    auto_model = AutoencoderCF(n_movies).to(device)
    auto_model = fit_autoencoder(auto_model, train_loader, n_users, n_movies, device=device)
    auto_mae = evaluate_model(auto_model, test_loader, 'autoencoder', n_users, n_movies, device=device)

    # Store results
    results = {
        'NCF_MLP': ncf_mae,
        'Autoencoder': auto_mae
    }
    save_results(results)


if __name__ == '__main__':
    main()