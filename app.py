import gradio as gr
from transformers import pipeline
import warnings
import spaces

warnings.filterwarnings("ignore")

# Load models at the start
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@spaces.GPU
def analyze_journal(text):
    if not text.strip():
        return "Please enter a journal entry.", "N/A"
    
    # Take first 2000 characters to prevent token limits
    safe_text = text[:2000]
    
    summary = summarizer(safe_text, max_length=130, min_length=30, do_sample=False)
    sentiment = sentiment_analyzer(safe_text)
    
    sentiment_text = f"{sentiment[0]['label']} (Confidence: {sentiment[0]['score']:.2%})"
    summary_text = summary[0]['summary_text'].strip()
    
    return summary_text, sentiment_text

# Build the Gradio UI
demo = gr.Interface(
    fn=analyze_journal,
    inputs=gr.Textbox(lines=10, label="Your Journal Entry", placeholder="Write your deeply personal thoughts here..."),
    outputs=[
        gr.Textbox(label="AI Summary"),
        gr.Textbox(label="Emotional Sentiment")
    ],
    title="Journal Buddy 📔",
    description="A completely open-source AI journal analyzer. (Built for Hacktoberfest)",
    allow_flagging="never"  # Ensure no data is saved!
)

if __name__ == "__main__":
    demo.launch()
