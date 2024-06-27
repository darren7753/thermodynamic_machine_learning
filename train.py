import os
import pandas as pd
from pycaret.regression import setup, compare_models, create_model, save_model
from dotenv import load_dotenv
from pymongo import MongoClient

# Function to load data from MongoDB
def load_from_mongo(collection):
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    if "_id" in df.columns:
        df.drop("_id", axis=1, inplace=True)
    return df

# Function to train and save models
def train_and_save_models(collection_map, target_map):
    load_dotenv(".env")

    # MongoDB Connections
    client = MongoClient(
        os.getenv("MONGO_CONNECTION_STRING"),
        serverSelectionTimeoutMS=300000
    )
    db = client[os.getenv("MONGO_DATABASE_NAME")]

    # Load data from MongoDB collections
    dataframes = {}
    for data_name, collection_name in collection_map.items():
        collection = db[collection_name]
        dataframes[data_name] = load_from_mongo(collection)

    # Create directory structure
    base_dir = "Models"
    os.makedirs(base_dir, exist_ok=True)

    # Train models for each dataframe
    for name, df in dataframes.items():
        target = target_map[name]
        print(f"Training models for {name}...")

        # Debug: Print shape and head of dataframe
        print(f"Data shape for {name}: {df.shape}")
        print(f"Data columns for {name}: {df.columns}")
        print(f"Data head for {name}:\n{df.head()}")

        # Setup PyCaret
        reg = setup(data=df, target=target, verbose=False, session_id=123)

        # Compare models
        best_model = compare_models()
        results = reg.pull()

        # Debug: Print results
        print(f"Comparison results for {name}:\n{results}")

        # Check if results are empty
        if results.empty:
            print(f"No models were trained successfully for {name}. Skipping...")
            continue

        # Create subdirectory for the dataset
        model_dir = f"{base_dir}/{name}"
        os.makedirs(model_dir, exist_ok=True)

        # Save comparison results to CSV inside the subdirectory
        results.to_csv(f"{model_dir}/model_comparison.csv")

        # Create and save models
        for model_name in results.index:
            model = create_model(model_name)
            save_model(model, os.path.join(model_dir, model_name))

        # Save the best model name
        with open(f"{model_dir}/best_model.txt", "w") as f:
            f.write(results.index[0])

        print(f"Models for {name} saved successfully.")

    print("All models trained and saved successfully.")

# Main function to run the script
def run():
    # Define MongoDB collection names and corresponding targets
    collection_map = {
        "ENGINE": os.getenv("MONGO_COLLECTION_NAME_ENGINE"),
        "SAVONIUS": os.getenv("MONGO_COLLECTION_NAME_SAVONIUS"),
        "CRANK_MECHANISM": os.getenv("MONGO_COLLECTION_NAME_CRANK_MECHANISM"),
        "PLTU": os.getenv("MONGO_COLLECTION_NAME_PLTU")
    }

    target_map = {
        "ENGINE": "Effisiensi Panas (%)",
        "SAVONIUS": "Rasio Kepesatan (l)",
        "CRANK_MECHANISM": "Kerugian Gesekan %",
        "PLTU": "output generator (Watt)"
    }

    # Execute training and saving of models
    train_and_save_models(collection_map, target_map)

# Entry point to the script
if __name__ == "__main__":
    run()