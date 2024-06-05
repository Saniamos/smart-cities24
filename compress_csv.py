import os
import pandas as pd
import glob
from pqdm.processes import pqdm

def process_file(file_name):
    # Read the CSV file
    df = pd.read_csv(file_name)
    # Save as parquet file
    df.to_parquet(file_name.replace('.csv', ".parquet"))
    # Remove the original file
    os.remove(file_name)

if __name__ == "__main__":
    files = glob.glob("./recordings/**/*.csv", recursive=True)
    pqdm(files, process_file, n_jobs=4)