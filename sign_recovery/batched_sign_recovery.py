# Imports
import os
import sign_recovery
import multiprocessing
from itertools import product

# ========== Global Settings ========== #
model_name = "cifar10_3x256_64_10_float64"  # Name of the model to be analyzed
model_path = f"../data/{model_name}.keras"  # Path to the .keras file containing the model
duals_path = f"../data/dual_points_{model_name}"  # Path to precomputed dual points
LAYERIDS = [1, 2, 3, 4]  # layer IDs to analyze
NEURONIDS = range(256)  # neuron IDs to analyze
analyzeWiggleSensitivity = 'False'  # Record the sensitivity to the wiggle at the target layer
analyzeSpeed = 'False'  # Record the rate of change of future layer neurons
nDebug = 'False'  # Set to True to skip logfile
nThreads = 10  # Number of threads to be used for analyzing multiple neurons in parallel
# ==================================== #

def run_analysis(params):
    """Wrapper function to run sign_recovery.main with given parameters."""
    layerID, neuronID = params
    try:
        args = [
            '--model', model_path,
            '--layerID', str(layerID),
            '--neuronID', str(neuronID),
            '--filepath_load_x0', duals_path,
            '--analyzeWiggleSensitivity', analyzeWiggleSensitivity,
            '--analyzeSpeed', analyzeSpeed,
            '--nDebug', nDebug
        ]
        print(f"🚀 Starting analysis for Layer {layerID}, Neuron {neuronID}")
        sign_recovery.main(args)
        print(f"✅ Finished analysis for Layer {layerID}, Neuron {neuronID}")
    except Exception as e:
        print(f"❌ Error in Layer {layerID}, Neuron {neuronID}: {e}")

def main():
    """Main function to set up and run the multiprocessing pool."""
    # Create a list of all parameter combinations
    tasks = list(product(LAYERIDS, NEURONIDS))
    
    print("="*50)
    print("Starting batched sign recovery...")
    print(f"Model: {model_name}")
    print(f"Layers to analyze: {LAYERIDS}")
    print(f"Neurons per layer: {len(list(NEURONIDS))}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Using {nThreads} threads.")
    print("="*50)

# Set up multiprocessing
if nThreads > 1:
        # Define a simple error handler for the pool
    def error_handler(e): 
            print(f"Pool Error: {e}")
            print(f"Cause: {e.__cause__}")

        # Use a multiprocessing pool to run tasks in parallel
        with multiprocessing.Pool(processes=nThreads) as pool:
            pool.map(run_analysis, tasks)
    else: 
        # Run tasks sequentially if only one thread is specified
        for task in tasks:
            run_analysis(task)
            
    print("\n" + "="*50)
    print("🎉 All tasks completed!")
    print("="*50)


if __name__ == '__main__':
    # On Windows, multiprocessing programs must be guarded by this check
    multiprocessing.freeze_support()
    main()
