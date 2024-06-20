import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_metrics(outputs, attributes):
    # MSE
    mse_voice = ((outputs[:, 0] - attributes[:, 0]) ** 2).mean()
    mse_density = ((outputs[:, 1] - attributes[:, 1]) ** 2).mean()
    mse_ioi = ((outputs[:, 2] - attributes[:, 2]) ** 2).mean()
    mse_intensity = ((outputs[:, 3] - attributes[:, 3]) ** 2).mean()
    mse_velocity = ((outputs[:, 4] - attributes[:, 4]) ** 2).mean()

    # MAE
    mae_voice = (outputs[:, 0] - attributes[:, 0]).abs().mean()
    mae_density = (outputs[:, 1] - attributes[:, 1]).abs().mean()
    mae_ioi = (outputs[:, 2] - attributes[:, 2]).abs().mean()
    mae_intensity = (outputs[:, 3] - attributes[:, 3]).abs().mean()
    mae_velocity = (outputs[:, 4] - attributes[:, 4]).abs().mean()

    # RMSE
    rmse_voice = torch.sqrt(mse_voice)
    rmse_density = torch.sqrt(mse_density)
    rmse_ioi = torch.sqrt(mse_ioi)
    rmse_intensity = torch.sqrt(mse_intensity)
    rmse_velocity = torch.sqrt(mse_velocity)

    # 중간 값 출력
    print(f"MAE Voice: {mae_voice.item()}, RMSE Voice: {rmse_voice.item()}")
    print(f"MAE Density: {mae_density.item()}, RMSE Density: {rmse_density.item()}")
    print(f"MAE IOI: {mae_ioi.item()}, RMSE IOI: {rmse_ioi.item()}")
    print(f"MAE Intensity: {mae_intensity.item()}, RMSE Intensity: {rmse_intensity.item()}")
    print(f"MAE Velocity: {mae_velocity.item()}, RMSE Velocity: {rmse_velocity.item()}")


    return (mse_voice.item(), mae_voice.item(), rmse_voice.item()),\
        (mse_density.item(), mae_density.item(), rmse_density.item()),\
        (mse_ioi.item(), mae_ioi.item(), rmse_ioi.item()),\
        (mse_intensity.item(), mae_intensity.item(), rmse_intensity.item()),\
        (mse_velocity.item(), mae_velocity.item(), rmse_velocity.item())

def plot_metrics(metrics, title, ylim=None, colors=None):
    attributes = ['Voice Number', 'Rhythmic Density', 'Mean IOI', 'Rhythmic Intensity', 'Average Velocity']
    x = np.arange(len(attributes))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(10, 8))
    rects = ax.bar(x, metrics, width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(attributes, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    if ylim:
        ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout()
    plt.savefig(f'./save/{title}.png', dpi=300)
    plt.show()

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./save/loss_curves.png')

def save_model(state_dict, path):
    torch.save(state_dict, path)
