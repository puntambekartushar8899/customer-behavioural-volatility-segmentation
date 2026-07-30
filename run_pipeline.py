from pathlib import Path
import pandas as pd

from src.preprocessing import load_raw_data, clean_online_retail_data, save_cleaned_data
from src.features import create_rfm_features, create_volatility_features, create_rfm_volatility_features
from src.clustering import scale_features, add_cluster_labels
from src.evaluation import compare_models, create_persona_summary, assign_persona_labels


BASE_DIR = Path(__file__).resolve().parent

raw_path = BASE_DIR / "data" / "raw" / "Online Retail.xlsx"

processed_dir = BASE_DIR / "data" / "processed"
tables_dir = BASE_DIR / "outputs" / "tables"

processed_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)


print("Loading raw dataset...")
raw_df = load_raw_data(raw_path)

print("Cleaning dataset...")
cleaned_df = clean_online_retail_data(raw_df)
save_cleaned_data(cleaned_df, processed_dir / "cleaned_online_retail.csv")

print("Creating RFM features...")
rfm = create_rfm_features(cleaned_df)
rfm.to_csv(processed_dir / "rfm_features.csv")

print("Creating volatility features...")
volatility = create_volatility_features(cleaned_df)
volatility.to_csv(processed_dir / "volatility_features.csv")

print("Creating RFM + volatility dataset...")
rfm_volatility = create_rfm_volatility_features(rfm, volatility)
rfm_volatility.to_csv(processed_dir / "rfm_volatility_features.csv")

print("Scaling feature sets...")
rfm_scaled, _ = scale_features(rfm)
rfm_vol_scaled, _ = scale_features(rfm_volatility)

print("Running clustering...")
rfm_clustered = add_cluster_labels(rfm, rfm_scaled, n_clusters=4)
rfm_vol_clustered = add_cluster_labels(rfm_volatility, rfm_vol_scaled, n_clusters=4)

rfm_clustered.to_csv(processed_dir / "rfm_clustered.csv")
rfm_vol_clustered.to_csv(processed_dir / "rfm_volatility_clustered.csv")

print("Calculating metrics...")
metrics = compare_models(rfm_scaled, rfm_clustered, rfm_vol_scaled, rfm_vol_clustered)
metrics.to_csv(tables_dir / "clustering_metrics.csv", index=False)

print("Creating customer personas...")
personas = create_persona_summary(rfm_vol_clustered)
personas = assign_persona_labels(personas)
personas.to_csv(tables_dir / "labelled_customer_personas.csv")

print("Pipeline completed successfully.")
print(metrics)
print(personas)