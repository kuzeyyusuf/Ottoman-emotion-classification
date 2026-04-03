import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


VALID_EMOTIONS = {
    "cekince",
    "istek",
    "kizginlik",
    "merak",
    "minnettarlik",
    "saygi",
    "umut",
    "uzuntu"
}


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8")

    required_columns = {"text", "emotion"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Gerekli sütunlar bulunamadı. "
            f"Beklenen: {required_columns}, Bulunan: {set(df.columns)}"
        )

    df = df.dropna(subset=["text", "emotion"]).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip()

    df = df[(df["text"] != "") & (df["emotion"] != "")].copy()

    df = df[
        ~(
            (df["text"].str.lower() == "text") &
            (df["emotion"].str.lower() == "emotion")
        )
    ].copy()

    df = df[df["emotion"].isin(VALID_EMOTIONS)].copy()

    df = df.drop_duplicates(subset=["text", "emotion"]).copy()

    return df


def save_classification_report(y_test, y_pred, output_path: str):
    report = classification_report(y_test, y_pred, zero_division=0)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def save_confusion_matrix(y_test, y_pred, labels, output_path: str):
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("reports", exist_ok=True)

    train_df = load_and_clean_data("data.csv")
    test_df = load_and_clean_data("test_real.csv")

    print("Train veri sayısı:", len(train_df))
    print("\nTrain emotion dağılımı:")
    print(train_df["emotion"].value_counts())

    print("\nTest veri sayısı:", len(test_df))
    print("\nTest emotion dağılımı:")
    print(test_df["emotion"].value_counts())

    X_train = train_df["text"]
    y_train = train_df["emotion"]

    X_test = test_df["text"]
    y_test = test_df["emotion"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=10000
        )),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("EMOTION MODEL SONUÇLARI (GERÇEK TEST)")
    print("=" * 50)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    errors = pd.DataFrame({
        "text": X_test.values,
        "true": y_test.values,
        "pred": y_pred
    })

    errors = errors[errors["true"] != errors["pred"]].copy()

    print("\n--- Hatalı Tahmin Örnekleri ---")
    print(errors.head(20))

    errors.to_csv("reports/sample_errors.csv", index=False, encoding="utf-8-sig")
    save_classification_report(y_test, y_pred, "reports/classification_report.txt")
    save_confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(VALID_EMOTIONS),
        output_path="reports/confusion_matrix.png"
    )

    with open("emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nModel kaydedildi: emotion_model.pkl")
    print("Rapor kaydedildi: reports/classification_report.txt")
    print("Hatalar kaydedildi: reports/sample_errors.csv")
    print("Confusion matrix kaydedildi: reports/confusion_matrix.png")


if __name__ == "__main__":
    main()