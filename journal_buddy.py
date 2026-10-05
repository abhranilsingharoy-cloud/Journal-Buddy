import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Suppress Hugging Face symlink warnings on Windows and TensorFlow logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import pipeline
import warnings

# Suppress model loading warnings for a cleaner CLI
warnings.filterwarnings("ignore")

console = Console()

class JournalAnalyzer:
    """A local privacy-first AI text analyzer for journal entries."""
    
    def __init__(self) -> None:
        self.summarizer = None
        self.sentiment_analyzer = None
        
    def load_models(self) -> None:
        """Loads the Hugging Face models into memory."""
        try:
            self.summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6"
            )
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        except Exception as e:
            logging.error(f"Failed to load AI models: {e}")
            sys.exit(1)

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyzes text and returns summary and sentiment."""
        if not text.strip():
            raise ValueError("Provided text is empty.")
            
        # Chunking: Taking the first 2000 chars to respect distilbart token limits
        safe_text = text[:2000]
        
        summary = self.summarizer(safe_text, max_length=130, min_length=30, do_sample=False)
        sentiment = self.sentiment_analyzer(safe_text)
        
        return {
            "summary": summary[0]["summary_text"].strip(),
            "sentiment_label": sentiment[0]["label"],
            "sentiment_score": sentiment[0]["score"]
        }

def read_file(file_path: Path) -> str:
    """Reads the content of a file."""
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] The file {file_path} does not exist.")
        sys.exit(1)
        
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(
        prog="JournalBuddy",
        description="A completely offline, privacy-first journal analyzer using open-source AI."
    )
    parser.add_argument("file", type=Path, help="Path to the journal text file to analyze.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    console.print(Panel.fit("[bold blue]Journal Buddy[/bold blue] 📔\n[dim]100% Local & Private AI Analysis[/dim]"))
    
    text_content = read_file(args.file)
    analyzer = JournalAnalyzer()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Loading local AI models... (may take a moment on first run)", total=None)
        analyzer.load_models()
        
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Analyzing journal deeply... (Running completely offline)", total=None)
        try:
            results = analyzer.analyze(text_content)
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)

    # Display Results
    sentiment_color = "green" if results['sentiment_label'] == "POSITIVE" else "yellow"
    
    console.print("\n[bold]📝 Summary[/bold]")
    console.print(f"  {results['summary']}")
    
    console.print("\n[bold]🧠 Emotional Sentiment[/bold]")
    console.print(f"  Overall Vibe: [{sentiment_color}]{results['sentiment_label']}[/{sentiment_color}] "
                  f"(Confidence: {results['sentiment_score']:.2%})\n")

if __name__ == "__main__":
    main()
