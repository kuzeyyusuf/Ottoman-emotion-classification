import pickle
import gradio as gr

with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_emotion(text):
    text = text.strip()

    if not text:
        return "Lütfen bir cümle giriniz.", [["-", 0.0]]

    prediction = model.predict([text])[0]

    scores = model.decision_function([text])[0]
    labels = model.classes_

    ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)

    top_results = []
    for label, score in ranked:
        top_results.append([label, round(float(score), 4)])

    return prediction, top_results


demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Osmanlıca / Klasik Türkçe üsluplu bir cümle giriniz..."
    ),
    outputs=[
        gr.Textbox(label="Tahmin Edilen Duygu"),
        gr.Dataframe(
            headers=["Duygu", "Skor"],
            datatype=["str", "number"],
            label="Sınıf Skorları"
        )
    ],
    title="Osmanlıca Türkçesi Duygu Sınıflandırma Demo",
    description=(
        "Bu demo, Osmanlıca / Klasik Türkçe üsluplu ifadelerde duygu tahmini yapar."
    ),
    examples=[
        ["İnşallah murad olunan netice hâsıl olur."],
        ["Bu hususta bir miktar tereddüdüm vardır."],
        ["Zât-ı âlinize şükranlarımı arz ederim."],
        ["Meramımı arz etmeme müsaade buyurur musunuz?"]
    ]
)

if __name__ == "__main__":
    demo.launch()