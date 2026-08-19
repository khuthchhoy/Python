"""CoreML Model Exporter for On-Device iOS Inference.
Converts the trained Python ensemble models to Apple CoreML (.mlpackage) format.
"""

from pathlib import Path
import numpy as np
import coremltools as ct
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from stock_predictor.config import PredictionConfig
from stock_predictor.data.synthetic import generate_synthetic_stock_data
from stock_predictor.features.pipeline import FeaturePipeline


def export_models_to_coreml(output_dir: str = "ios/CoreML"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("1. Generating training dataset for CoreML export...")
    config = PredictionConfig(forecast_horizon=5)
    df = generate_synthetic_stock_data(n_days=500, seed=42)
    pipeline = FeaturePipeline(config)
    dataset = pipeline.prepare_dataset(df, horizon=5)
    
    X = dataset["X"].values
    y = dataset["y_return"].values
    feature_names = dataset["feature_names"]
    
    print("2. Training RandomForest Regressor for CoreML compatibility...")
    rf_model = RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42)
    rf_model.fit(X, y)
    
    print("3. Converting model to Apple CoreML (.mlpackage)...")
    coreml_model = ct.converters.sklearn.convert(
        rf_model,
        input_features=feature_names,
        output_feature_names="predicted_5d_log_return"
    )
    
    coreml_model.author = "AI Quantitative Engine"
    coreml_model.short_description = "Predicts 1-week ahead stock log return from 35+ technical features"
    
    model_save_path = out_path / "StockReturnPredictor.mlpackage"
    coreml_model.save(str(model_save_path))
    print(f"✅ CoreML model successfully saved to: {model_save_path}")
    print("You can now drag and drop StockReturnPredictor.mlpackage directly into Xcode!")


if __name__ == "__main__":
    export_models_to_coreml()
