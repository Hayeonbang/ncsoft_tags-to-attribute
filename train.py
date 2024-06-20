import argparse
import os
import pandas as pd
import math

import torch
import torch.optim as optim

from model import MLP, load_model
from utils import calculate_metrics, plot_metrics, plot_loss_curves, save_model
from dataset import create_data_loaders
from data_preprocess import data_preprocess, data_encoder, attribute_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and validate Text-to-Attribute model')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--data_path', type=str, default='./dataset/tagging_results.csv', help='Path to the tag data CSV file')
    parser.add_argument('--attribute_path', type=str, default='./dataset/attribute_results.csv', help='Path to the attribute CSV file')
    parser.add_argument('--processed_attribute_path', type=str, default='./dataset/attribute_preprocessed.csv', help='Path to the processed attribute CSV file')
    parser.add_argument('--model_save_path', type=str, default='./save/trained_model.pth', help='Path to save the trained model')
    return parser.parse_args()

def load_data(data_path, attribute_path, processed_attribute_path):
    data_df = pd.read_csv(data_path)
    attribute_df = pd.read_csv(attribute_path)
    data_processed, attribute_processed = data_preprocess(data_df, attribute_df)
    data_vector = data_encoder(data_processed)
    attribute_processed = pd.read_csv(processed_attribute_path)
    return data_processed, data_vector, attribute_processed

def get_attribute_ranges(attribute_processed):
    voice_min = attribute_processed['normalized_voice_number'].min()
    voice_max = attribute_processed['normalized_voice_number'].max()
    density_min = attribute_processed['normalized_rhythmic_density'].min()
    density_max = attribute_processed['normalized_rhythmic_density'].max()
    ioi_min = attribute_processed['normalized_mean_ioi'].min()
    ioi_max = attribute_processed['normalized_mean_ioi'].max()
    rhythmic_intensity_min = attribute_processed['normalized_rhythmic_intensity'].min()
    rhythmic_intensity_max = attribute_processed['normalized_rhythmic_intensity'].max()
    velocity_min = attribute_processed['normalized_average_velocity'].min()
    velocity_max = attribute_processed['normalized_average_velocity'].max()
    return voice_min, voice_max, density_min, density_max, ioi_min, ioi_max, rhythmic_intensity_min, rhythmic_intensity_max, velocity_min, velocity_max

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for features, attributes, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, attributes)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0
        total_rmse = 0
        num_batches = 0
        with torch.no_grad():
            for features, attributes, _ in val_loader:
                outputs = model(features)
                loss = criterion(outputs, attributes)
                total_val_loss += loss.item()
                rmse = math.sqrt(loss.item())
                total_rmse += rmse
                num_batches += 1

        val_loss = total_val_loss / len(val_loader)
        avg_rmse = total_rmse / num_batches
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation RMSE: {avg_rmse:.4f}')

    return train_losses, val_losses, best_model

def main():
    args = parse_arguments()
    try:
        data_processed, data_vector, attribute_processed = load_data(args.data_path, args.attribute_path, args.processed_attribute_path)
        attribute_ranges = get_attribute_ranges(attribute_processed)
        attribute_tensor = attribute_loader(attribute_processed, *attribute_ranges)
        rel_paths = data_processed['rel_path'].tolist()

        model = MLP()
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        train_loader, val_loader, test_loader = create_data_loaders(data_vector, attribute_tensor, rel_paths, args.batch_size)

        train_losses, val_losses, best_model = train_and_validate(model, train_loader, val_loader, criterion, optimizer, args.num_epochs)
        save_model(best_model, args.model_save_path)
        plot_loss_curves(train_losses, val_losses)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()