#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import torch
import json
from datetime import datetime


def save_model_parameters(model, save_path, model_name):
    """
    Save model parameters to file
    Args:
        model: PyTorch model
        save_path: directory to save the model
        model_name: name of the model file
    """
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    # print(f"Model parameters saved to: {model_path}")


def save_all_client_models(client_models, save_dir, round_num, algorithm_name):
    """
    Save all client model parameters
    Args:
        client_models: list of client models or their state_dicts
        save_dir: base directory to save models
        round_num: training round number
        algorithm_name: name of the federated learning algorithm
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    client_dir = os.path.join(save_dir, f"{algorithm_name}_{timestamp}", f"round_{round_num}", "clients")
    os.makedirs(client_dir, exist_ok=True)
    
    for i, model in enumerate(client_models):
        if isinstance(model, dict):  # if it's already a state_dict
            model_path = os.path.join(client_dir, f"client_{i}.pth")
            torch.save(model, model_path)
        else:  # if it's a model object
            model_path = os.path.join(client_dir, f"client_{i}.pth")
            torch.save(model.state_dict(), model_path)
        # print(f"Client {i} parameters saved to: {model_path}")
    
    return client_dir


def save_server_model(server_model, save_dir, round_num, algorithm_name):
    """
    Save server model parameters
    Args:
        server_model: server model or its state_dict
        save_dir: base directory to save models
        round_num: training round number
        algorithm_name: name of the federated learning algorithm
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    server_dir = os.path.join(save_dir, f"{algorithm_name}_{timestamp}", f"round_{round_num}", "server")
    os.makedirs(server_dir, exist_ok=True)
    
    if isinstance(server_model, dict):  # if it's already a state_dict
        model_path = os.path.join(server_dir, "server_model.pth")
        torch.save(server_model, model_path)
    else:  # if it's a model object
        model_path = os.path.join(server_dir, "server_model.pth")
        torch.save(server_model.state_dict(), model_path)
    
    # print(f"Server model parameters saved to: {model_path}")
    return server_dir


def save_training_info(save_dir, algorithm_name, args, final_accuracy):
    """
    Save training information and hyperparameters
    Args:
        save_dir: base directory
        algorithm_name: name of the algorithm
        args: training arguments
        final_accuracy: final test accuracy
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    info_dir = os.path.join(save_dir, f"{algorithm_name}_{timestamp}")
    os.makedirs(info_dir, exist_ok=True)
    
    training_info = {
        "algorithm": algorithm_name,
        "epochs": args.epochs,
        "num_users": args.num_users,
        "frac": args.frac,
        "local_ep": args.local_ep,
        "local_bs": args.local_bs,
        "lr": args.lr,
        "model": args.model,
        "dataset": args.dataset,
        "final_accuracy": final_accuracy,
        "timestamp": timestamp
    }
    
    info_path = os.path.join(info_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=4)
    
    print(f"Training info saved to: {info_path}")
    return info_dir
