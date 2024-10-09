# script to test dynamics with the cluster
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Test dynamics with the cluster')
    parser.add_argument('-n', '--number', type=str, required=True, help='Number of the experiment')
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()
    exp_number = args.number
    config_path = args.config

    #create a directory inside mg_data/#exp_number
    os.makedirs(f"mg_data/{exp_number}", exist_ok=True)

    #create a random plot and save it in the directory
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig(f"mg_data/{exp_number}/plot{exp_number}.png")
    

    print(f"Experiment number: {exp_number}")
    print(f"Config file path: {config_path}")