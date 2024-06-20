import pandas as pd
import numpy as np
import torch

def data_preprocess(data_df, attribute_df):
    data_sorted = data_df.sort_values(by='rel_path').reset_index(drop=True)
    attribute_sorted = attribute_df.sort_values(by='rel_path').reset_index(drop=True)
    
    attribute_sorted = attribute_sorted[attribute_sorted['mean_ioi'] < 10].reset_index(drop=True)
    
    average_voice_number_min = attribute_sorted['voice_number'].min()
    average_voice_number_max = attribute_sorted['voice_number'].max()
    attribute_sorted['normalized_voice_number'] = (attribute_sorted['voice_number'] - average_voice_number_min) / (average_voice_number_max - average_voice_number_min)
    
    rhythmic_density_min = attribute_sorted['rhythmic_density'].min()
    rhythmic_density_max = attribute_sorted['rhythmic_density'].max()
    attribute_sorted['normalized_rhythmic_density'] = (attribute_sorted['rhythmic_density'] - rhythmic_density_min) / (rhythmic_density_max - rhythmic_density_min)
    
    mean_ioi_min = attribute_sorted['mean_ioi'].min()
    mean_ioi_max = attribute_sorted['mean_ioi'].max()
    attribute_sorted['normalized_mean_ioi'] = (attribute_sorted['mean_ioi'] - mean_ioi_min) / (mean_ioi_max - mean_ioi_min)
    
    rhythmic_intensity_min = attribute_sorted['rhythmic_intensity'].min()
    rhythmic_intensity_max = attribute_sorted['rhythmic_intensity'].max()
    attribute_sorted['normalized_rhythmic_intensity'] = (attribute_sorted['rhythmic_intensity'] - rhythmic_intensity_min) / (rhythmic_intensity_max - rhythmic_intensity_min)

    average_velocity_min = attribute_sorted['average_velocity'].min()
    average_velocity_max = attribute_sorted['average_velocity'].max()
    attribute_sorted['normalized_average_velocity'] = (attribute_sorted['average_velocity'] - average_velocity_min) / (average_velocity_max - average_velocity_min)
    
    common_rel_path = set(data_sorted['rel_path']).intersection(set(attribute_sorted['rel_path']))
    data_final = data_sorted[data_sorted['rel_path'].isin(common_rel_path)].reset_index(drop=True)
    attribute_final = attribute_sorted[attribute_sorted['rel_path'].isin(common_rel_path)].reset_index(drop=True)
    
    attribute_final.to_csv('./dataset/attribute_preprocessed.csv', index=False)
    
    return data_final, attribute_final

def data_encoder(data_df, threshold=0.55):
    df_values = data_df.loc[:, 'jazz':'classical']
    normalized_values = (df_values - df_values.min()) / (df_values.max() - df_values.min())
    encoded_values = np.where(normalized_values >= threshold, 1, 0)
    tensor_data = torch.tensor(encoded_values, dtype=torch.float32)
    return tensor_data

def attribute_loader(attribute_df, voice_min, voice_max, density_min, density_max, ioi_min, ioi_max, velocity_min, velocity_max, rhythmic_intensity_min, rhythmic_intensity_max):
    voice_numbers = attribute_df['normalized_voice_number']
    rhythmic_densities = attribute_df['normalized_rhythmic_density']
    mean_iois = attribute_df['normalized_mean_ioi']
    rhythmic_intensities = attribute_df['normalized_rhythmic_intensity']
    average_velocities = attribute_df['normalized_average_velocity']
    
    normalized_voice_numbers = (voice_numbers - voice_min) / (voice_max - voice_min)
    normalized_rhythmic_densities = (rhythmic_densities - density_min) / (density_max - density_min)
    normalized_rhythmic_intensities = (rhythmic_intensities - rhythmic_intensity_min) / (rhythmic_intensity_max - rhythmic_intensity_min)
    normalized_mean_iois = (mean_iois - ioi_min) / (ioi_max - ioi_min)
    normalized_average_velocities = (average_velocities - velocity_min) / (velocity_max - velocity_min)
    
    normalized_attributes = pd.concat([normalized_voice_numbers, normalized_rhythmic_densities, normalized_mean_iois, normalized_rhythmic_intensities, normalized_average_velocities], axis=1)
    tensor_attribute = torch.tensor(normalized_attributes.values, dtype=torch.float32)
    return tensor_attribute
