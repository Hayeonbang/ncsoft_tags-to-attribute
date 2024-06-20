import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from data_preprocess import data_preprocess, data_encoder, attribute_loader
from model import MLP, load_model
from dataset import create_data_loaders
from utils import calculate_metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate Text-to-Attribute model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--data_path', type=str, default='./dataset/tagging_results.csv', help='Path to the data CSV file')
    parser.add_argument('--attribute_path', type=str, default='./dataset/attribute_results.csv', help='Path to the attribute CSV file')
    parser.add_argument('--processed_attribute_path', type=str, default='./dataset/attribute_preprocessed.csv', help='Path to the processed attribute CSV file')
    parser.add_argument('--model_path', type=str, default='./save/trained_model.pth', help='Path to the trained model')
    parser.add_argument('--results_path', type=str, default='./save/predicted_results.csv', help='Path to save the evaluation results')
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

def evaluate_model(model, test_loader, attribute_ranges, results_path):
    model.eval()
    total_metrics = {
        'voice': [0, 0, 0],
        'density': [0, 0, 0],
        'ioi': [0, 0, 0],
        'intensity': [0, 0, 0],
        'velocity': [0, 0, 0]
    }
    total_samples = 0
    predictions = []
    actuals = []
    rel_paths_all = []

    with torch.no_grad():
        for features, attributes, rel_paths in test_loader:
            outputs = model(features)
            predictions.extend(outputs.tolist())
            actuals.extend(attributes.tolist())
            rel_paths_all.extend(rel_paths)

            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            if attributes.dim() == 1:
                attributes = attributes.unsqueeze(0)

            (mse_voice, mae_voice, rmse_voice),\
            (mse_density, mae_density, rmse_density),\
            (mse_ioi, mae_ioi, rmse_ioi),\
            (mse_intensity, mae_intensity, rmse_intensity),\
            (mse_velocity, mae_velocity, rmse_velocity) = calculate_metrics(outputs, attributes)

            total_metrics['voice'][0] += mse_voice * features.size(0)
            total_metrics['voice'][1] += mae_voice * features.size(0)
            total_metrics['voice'][2] += rmse_voice * features.size(0)

            total_metrics['density'][0] += mse_density * features.size(0)
            total_metrics['density'][1] += mae_density * features.size(0)
            total_metrics['density'][2] += rmse_density * features.size(0)

            total_metrics['ioi'][0] += mse_ioi * features.size(0)
            total_metrics['ioi'][1] += mae_ioi * features.size(0)
            total_metrics['ioi'][2] += rmse_ioi * features.size(0)
            
            total_metrics['intensity'][0] += mse_intensity * features.size(0)
            total_metrics['intensity'][1] += mae_intensity * features.size(0)
            total_metrics['intensity'][2] += rmse_intensity * features.size(0)

            total_metrics['velocity'][0] += mse_velocity * features.size(0)
            total_metrics['velocity'][1] += mae_velocity * features.size(0)
            total_metrics['velocity'][2] += rmse_velocity * features.size(0)

            total_samples += features.size(0)

    for attr_name in total_metrics.keys():
        for i in range(3):
            total_metrics[attr_name][i] /= total_samples

    save_results(predictions, actuals, rel_paths_all, attribute_ranges, results_path)
    return total_metrics

def save_results(predictions, actuals, rel_paths_all, attribute_ranges, results_path):
    voice_min, voice_max, density_min, density_max, ioi_min, ioi_max, rhythmic_intensity_min, rhythmic_intensity_max, velocity_min, velocity_max = attribute_ranges
    
    predictions_array = np.array(predictions)
    actuals_array = np.array(actuals)

    results_df = pd.DataFrame(data={
        "Rel_Path": rel_paths_all,
        "Predicted_Voice_Number": predictions_array[:, 0] * (voice_max - voice_min) + voice_min,
        "Actual_Voice_Number": actuals_array[:, 0] * (voice_max - voice_min) + voice_min,
        "Predicted_Rhythmic_Density": predictions_array[:, 1] * (density_max - density_min) + density_min,
        "Actual_Rhythmic_Density": actuals_array[:, 1] * (density_max - density_min) + density_min,
        "Predicted_Mean_IOI": predictions_array[:, 2] * (ioi_max - ioi_min) + ioi_min,
        "Actual_Mean_IOI": actuals_array[:, 2] * (ioi_max - ioi_min) + ioi_min,
        "Predicted_Rhythmic_Intensity": predictions_array[:, 3] * (rhythmic_intensity_max - rhythmic_intensity_min) + rhythmic_intensity_min,
        "Actual_Rhythmic_Intensity": actuals_array[:, 3] * (rhythmic_intensity_max - rhythmic_intensity_min) + rhythmic_intensity_min,
        "Predicted_Average_Velocity": predictions_array[:, 4] * (velocity_max - velocity_min) + velocity_min,
        "Actual_Average_Velocity": actuals_array[:, 4] * (velocity_max - velocity_min) + velocity_min,
        "Normalized_Predicted_Voice_Number": predictions_array[:, 0],
        "Normalized_Actual_Voice_Number": actuals_array[:, 0],
        "Normalized_Predicted_Rhythmic_Density": predictions_array[:, 1],
        "Normalized_Actual_Rhythmic_Density": actuals_array[:, 1],
        "Normalized_Predicted_Mean_IOI": predictions_array[:, 2],
        "Normalized_Actual_Mean_IOI": actuals_array[:, 2],
        "Normalized_Predicted_Rhythmic_Intensity": predictions_array[:, 3],
        "Normalized_Actual_Rhythmic_Intensity": actuals_array[:, 3],
        "Normalized_Predicted_Average_Velocity": predictions_array[:, 4],
        "Normalized_Actual_Average_Velocity": actuals_array[:, 4]
    })

    results_df.to_csv(results_path, index=False)

def main():
    args = parse_arguments()
    try:
        data_processed, data_vector, attribute_processed = load_data(args.data_path, args.attribute_path, args.processed_attribute_path)
        attribute_ranges = get_attribute_ranges(attribute_processed)

        attribute_tensor = attribute_loader(attribute_processed, *attribute_ranges)
        rel_paths = data_processed['rel_path'].tolist()

        model = MLP()
        load_model(model, args.model_path)

        _, _, test_loader = create_data_loaders(data_vector, attribute_tensor, rel_paths, args.batch_size)

        metrics = evaluate_model(model, test_loader, attribute_ranges, args.results_path)

        for attr_name, attr_metrics in metrics.items():
            print(f"{attr_name.capitalize()} - Final test MSE: {attr_metrics[0]:.4f}")
            print(f"{attr_name.capitalize()} - Final test MAE: {attr_metrics[1]:.4f}")
            print(f"{attr_name.capitalize()} - Final test RMSE: {attr_metrics[2]:.4f}")


    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
