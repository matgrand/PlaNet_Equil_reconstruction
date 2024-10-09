# script to test dynamics with the cluster
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # assign to exp_number the value of $SLURM_JOB_ID
    exp_number = os.environ["SLURM_JOB_ID"]
    
    #create a directory inside mg_data/#exp_number
    os.makedirs(f"mg_data/{exp_number}", exist_ok=True)

    #create a random plot and save it in the directory
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig(f"mg_data/{exp_number}/plot{exp_number}.png")
    

    print(f"Experiment number: {exp_number}")