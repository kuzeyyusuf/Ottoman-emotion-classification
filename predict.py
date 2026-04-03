import pickle

with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Duygu tahmin sistemi hazır.")
print("Çıkmak için q yaz.\n")

while True:
    text = input("Bir cümle gir: ").strip()

    if text.lower() == "q":
        print("Program sonlandırıldı.")
        break

    if not text:
        print("Lütfen boş bir cümle girme.\n")
        continue

    prediction = model.predict([text])[0]

    # LinearSVC için skorlar
    scores = model.decision_function([text])[0]
    labels = model.classes_

    ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)

    print(f"\nTahmin edilen duygu: {prediction}")
    print("En yakın 3 sınıf:")

    for label, score in ranked[:3]:
        print(f"- {label}: {score:.4f}")

    print()