import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = ["temp", "humidity", "wind_speed", "pressure", "description_len"]
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = BASE_DIR / "training_data.csv"
MODEL_FILE = BASE_DIR / "clothing_model.pkl"


def build_synthetic_dataset() -> pd.DataFrame:
    rows = [
        # hot weather
        (36, 45, 4, 1008, 10, "Wear breathable fabrics, shorts, and sunglasses for hot weather.", "Bright summer shades like sky blue, mint green, or peach."),
        (33, 50, 5, 1009, 12, "Wear breathable fabrics, shorts, and sunglasses for hot weather.", "Bright summer shades like sky blue, mint green, or peach."),
        (31, 60, 3, 1007, 8, "Wear breathable fabrics, shorts, and sunglasses for hot weather.", "Bright summer shades like sky blue, mint green, or peach."),
        # cold weather
        (10, 70, 4, 1018, 9, "Choose a warm coat, scarf, and layers for cold weather.", "Warm/deep tones like maroon, navy, charcoal, or olive."),
        (7, 75, 6, 1020, 11, "Choose a warm coat, scarf, and layers for cold weather.", "Warm/deep tones like maroon, navy, charcoal, or olive."),
        (13, 65, 5, 1016, 10, "Choose a warm coat, scarf, and layers for cold weather.", "Warm/deep tones like maroon, navy, charcoal, or olive."),
        # rainy weather
        (24, 90, 8, 1005, 13, "Carry a waterproof jacket and wear comfortable shoes for rainy weather.", "Darker practical colors like navy, black, or deep gray."),
        (27, 88, 10, 1004, 15, "Carry a waterproof jacket and wear comfortable shoes for rainy weather.", "Darker practical colors like navy, black, or deep gray."),
        (22, 92, 7, 1006, 14, "Carry a waterproof jacket and wear comfortable shoes for rainy weather.", "Darker practical colors like navy, black, or deep gray."),
        # windy/mild
        (26, 55, 12, 1010, 9, "A comfortable outfit with a windbreaker is ideal for windy weather.", "Neutral sporty colors like gray, olive, and denim blue."),
        (23, 58, 11, 1012, 10, "A comfortable outfit with a windbreaker is ideal for windy weather.", "Neutral sporty colors like gray, olive, and denim blue."),
        (20, 60, 9, 1013, 11, "A comfortable outfit with a windbreaker is ideal for windy weather.", "Neutral sporty colors like gray, olive, and denim blue."),
        # clear/mild
        (28, 52, 4, 1011, 9, "A light layer with a jacket is a good choice for mild weather.", "Light colors like white, beige, or pastel blue."),
        (25, 57, 5, 1012, 11, "A light layer with a jacket is a good choice for mild weather.", "Light colors like white, beige, or pastel blue."),
        (21, 62, 4, 1014, 10, "A light layer with a jacket is a good choice for mild weather.", "Light colors like white, beige, or pastel blue."),
    ]

    return pd.DataFrame(
        rows,
        columns=FEATURE_COLUMNS + ["clothing_label", "color_label"],
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        missing = [c for c in FEATURE_COLUMNS + ["clothing_label", "color_label"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset is missing columns: {missing}. Required columns are "
                f"{FEATURE_COLUMNS + ['clothing_label', 'color_label']}"
            )
        return df

    df = build_synthetic_dataset()
    df.to_csv(path, index=False)
    return df


def train_and_save(data_path: Path, model_path: Path) -> None:
    df = load_dataset(data_path)
    x = df[FEATURE_COLUMNS]
    y_clothing_text = df["clothing_label"]
    y_color_text = df["color_label"]

    clothing_encoder = LabelEncoder()
    color_encoder = LabelEncoder()
    y_clothing = clothing_encoder.fit_transform(y_clothing_text)
    y_color = color_encoder.fit_transform(y_color_text)

    x_train, x_test, y_train_c, y_test_c = train_test_split(
        x, y_clothing, test_size=0.25, random_state=42
    )
    y_color_series = pd.Series(y_color, index=x.index)
    y_train_color = y_color_series.loc[x_train.index].to_numpy()
    y_test_color = y_color_series.loc[x_test.index].to_numpy()

    clothing_model = RandomForestClassifier(n_estimators=300, random_state=42)
    color_model = RandomForestClassifier(n_estimators=300, random_state=42)

    clothing_model.fit(x_train.values, y_train_c)
    color_model.fit(x_train.values, y_train_color)

    pred_c = clothing_model.predict(x_test.values)
    pred_color = color_model.predict(x_test.values)

    print("Clothing accuracy:", round(accuracy_score(y_test_c, pred_c), 3))
    print("Color accuracy:", round(accuracy_score(y_test_color, pred_color), 3))
    print("\nClothing classification report:")
    print(classification_report(y_test_c, pred_c, zero_division=0))
    print("\nColor classification report:")
    print(classification_report(y_test_color, pred_color, zero_division=0))

    model_bundle = {
        "clothing_model": clothing_model,
        "color_model": color_model,
        "clothing_encoder": clothing_encoder,
        "color_encoder": color_encoder,
        "feature_columns": FEATURE_COLUMNS,
    }
    with model_path.open("wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\nSaved model to: {model_path.resolve()}")
    if data_path.exists():
        print(f"Training data used: {data_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE), help="CSV file for training data.")
    parser.add_argument("--out", default=str(MODEL_FILE), help="Output pickle model path.")
    args = parser.parse_args()

    train_and_save(Path(args.data), Path(args.out))


if __name__ == "__main__":
    main()
