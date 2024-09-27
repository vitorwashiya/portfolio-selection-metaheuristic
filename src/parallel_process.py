import concurrent.futures
from tqdm import tqdm
import traceback
import pickle
import gc
from utils import find_weights_gurobi, get_combinations, get_returns_df


grid = get_combinations()
data = get_returns_df()

dat_fim_tre = "2021-01-01"
data = data[:dat_fim_tre]
results = []

def process_parameters(data, parameters):
    try:
        df = find_weights_gurobi(data, parameters)
        return {
            "parameters": parameters,
            "df": df
        }
    except Exception as e:
        print(f"Error processing parameters {parameters}: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    chunk_size = 20
    for i in tqdm(range(0, len(grid), chunk_size), total=(len(grid) + chunk_size - 1) // chunk_size):
        chunk = grid[i:i + chunk_size]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_parameters, data, parameters) for parameters in chunk]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    results.append(result)
                gc.collect()
            del futures
            gc.collect()

    with open('../data/results.pkl', 'wb') as f:
        pickle.dump(results, f)