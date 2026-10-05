---
title: Journal Buddy
emoji: 🦀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.43.0
app_file: app.py
pinned: false
---

# Journal Buddy 📔

A completely offline, privacy-first journal analyzer powered by Open-Source AI.

Built for the **Hacktoberfest Weekend Challenge: Build for a Friend**.

## Why this exists
Many people love journaling for mental health but want a quick summary and emotional check-in at the end of the week. However, sending deeply personal journal entries to cloud APIs (like ChatGPT or Claude) is a massive privacy risk.

**Journal Buddy** solves this by using Hugging Face open-weight models (`transformers`) to process your journal entries 100% locally on your own machine. Your data never leaves your hard drive.

## Features
- **100% Local Inference**: Runs entirely on your CPU/GPU without internet access (after initial model download).
- **Automated Summarization**: Condenses long journal entries into a short recap.
- **Sentiment Analysis**: Detects the overall emotional vibe of the entry.
- **Beautiful CLI**: Uses the `rich` library for a gorgeous, responsive terminal interface.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhranilsingharoy-cloud/Journal-Buddy.git
   cd Journal-Buddy
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Simply pass a text file containing your journal entry to the CLI:

```bash
python journal_buddy.py sample_journal.txt
```

### Example Output:
![Example CLI Output](https://via.placeholder.com/600x300.png?text=Rich+Terminal+Output+Here)

## Tech Stack
- **Python 3.9+**
- **Hugging Face `transformers`**
- **PyTorch**
- **Rich** (for the CLI UI)
