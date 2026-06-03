"""
app.py
------
Gradio web interface for real-time fake news detection.
Supports both TF-IDF and DistilBERT models.
"""

import os
import gradio as gr

# Lazy-load models so the app starts fast
_tfidf_pipeline = None
_bert_model = None
_bert_tokenizer = None


def get_tfidf():
    global _tfidf_pipeline
    if _tfidf_pipeline is None:
        from tfidf_model import load_model
        _tfidf_pipeline = load_model()
    return _tfidf_pipeline


def get_bert():
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        from bert_model import load_model
        _bert_model, _bert_tokenizer = load_model()
    return _bert_model, _bert_tokenizer


def predict(article_text: str, model_choice: str) -> tuple[str, str, str]:
    """
    Run prediction and return (verdict, confidence, breakdown).
    """
    if not article_text or len(article_text.strip()) < 20:
        return "⚠️ Please enter a longer article (at least 20 characters).", "", ""

    try:
        if model_choice == "TF-IDF + Logistic Regression":
            from tfidf_model import predict as tfidf_predict
            result = tfidf_predict(article_text, pipeline=get_tfidf())
        else:
            from bert_model import predict as bert_predict
            model, tokenizer = get_bert()
            result = bert_predict(article_text, model=model, tokenizer=tokenizer)

        label = result["label"]
        confidence = result["confidence"]
        fake_prob = result["fake_prob"]
        real_prob = result["real_prob"]

        emoji = "✅" if label == "Real" else "🚨"
        verdict = f"{emoji} **{label} News**"
        conf_str = f"{confidence * 100:.1f}%"
        breakdown = (
            f"🔴 Fake probability: {fake_prob * 100:.1f}%\n"
            f"🟢 Real probability: {real_prob * 100:.1f}%"
        )
        return verdict, conf_str, breakdown

    except FileNotFoundError as e:
        return (
            "❌ Model not found. Please train the model first.",
            "",
            str(e),
        )


# --- Gradio UI ---

DESCRIPTION = """
## 📰 Fake News Detector
Paste a news article below and select a model to classify it as **Real** or **Fake**.

> **Models available:**
> - **TF-IDF + LR** — Fast, CPU-friendly baseline
> - **DistilBERT** — Deep learning model, higher accuracy (~96%)
"""

EXAMPLES = [
    ["Scientists have discovered a new species of deep-sea fish near the Mariana Trench, exhibiting bioluminescent properties never seen before. The findings were published in Nature this week.", "TF-IDF + Logistic Regression"],
    ["BREAKING: Government puts secret mind-control chemicals in tap water — doctors REFUSE to talk about it. Share before they delete this!", "TF-IDF + Logistic Regression"],
]

with gr.Blocks(title="Fake News Detector") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            article_input = gr.Textbox(
                label="Article Text",
                placeholder="Paste your news article here...",
                lines=10,
            )
            model_selector = gr.Radio(
                choices=["TF-IDF + Logistic Regression", "DistilBERT (fine-tuned)"],
                value="TF-IDF + Logistic Regression",
                label="Model",
            )
            submit_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column(scale=2):
            verdict_out = gr.Markdown(label="Verdict")
            confidence_out = gr.Textbox(label="Confidence", interactive=False)
            breakdown_out = gr.Textbox(label="Probability Breakdown", lines=3, interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=[article_input, model_selector])

    submit_btn.click(
        fn=predict,
        inputs=[article_input, model_selector],
        outputs=[verdict_out, confidence_out, breakdown_out],
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
