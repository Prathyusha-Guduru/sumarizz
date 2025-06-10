# Sumarizz

Comprehensive project report can be found [here](https://docs.google.com/document/d/1J9SLABuK7BQJpB9im6jp0KQTjX2i5o1-t7ichxiPjig/edit?usp=sharing)

Demo can be found [here](https://youtu.be/I1HCtGaJ0xw)

**Convert scientific papers into Gen Z–style summaries with ease.**

Sumarizz provides fine‑tuned language models that transform dense academic texts into punchy, slang‑infused recaps. Whether you’re a busy student, researcher, or science enthusiast, Sumarizz helps you grasp the highlights—fast.

---

## 🚀 Features

* **High‑quality summaries**: Fine‑tuned on scientific articles to preserve key insights.
* **Gen Z slang style**: Casual, engaging tone that resonates with younger audiences.
* **Large‑document support**: Automatically parses PDFs up to 16,000 tokens, trimming non‑essential sections.
* **Interactive demo**: Try it live with a Gradio web app.
* **Hugging Face integration**: Pre‑trained models hosted for instant inference.

---

## 📦 Installation

1. **Clone the repo**:

   ```bash
   git clone https://github.com/your-org/Sumarizz.git
   cd Sumarizz
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Project Structure

```
Sumarizz/
├── demo/               # Gradio demo and helper scripts
│   ├── demo.ipynb      # Launch Gradio interface
│   ├── pdf_parser.py   # Extract and clean text from PDFs
│   └── secret.toml     # AWS & other keys (gitignored)
├── models/             # Configuration and LoRA adapters
├── scripts/            # Data preprocessing and scraping scripts
│   ├── parse_pdfs.py   # PDF → raw text → CSV
│   ├── scrape_data.py   # Scrape paper metadata & summaries
│   └── build_prefs.py  # Generate preference pairs for DPO
├── training/           # Fine‑tuning & DPO pipelines
├── requirements.txt
└── README.md           # You are here!
```

---

## 🛠️ Modules Overview

### 1. PDF Parsing

* **Purpose**: Cleanly extract text from academic PDFs.
* **Workflow**:

  1. Load PDFs from a directory.
  2. Remove low‑priority sections (`References`, `Contributors`, `Affiliations`).
  3. Split or trim content to fit LED’s 16k‑token limit.
  4. Save structured output to CSV for downstream processing.

### 2. Dataset Scraping

* **Sources**:

  * [aimodels.fyi](https://aimodels.fyi/) (\~140 papers)
  * [PapersWithCode SOTA](https://paperswithcode.com/) (\~500 papers)
* **Insights**:

  * Smaller dataset (138 entries) yielded minimal ROUGE gains.
  * Expanded dataset (503 entries) improved scores but required quality filtering.

### 3. Preference‑Pair Generation

* **Goal**: Create winning/losing summary pairs for DPO fine‑tuning.
* **Process**: Compare outputs from different LoRA variants to label preferences.

### 4. Fine‑Tuning Strategies

* **LoRA Exploration**: Tested various LoRA configurations; selected best by ROUGE improvement.
* **DPO Refinement**: Further polished the chosen LoRA model with Direct Preference Optimization.

### 5. Demo

* **Gradio App**: Quick web interface for uploading PDFs and receiving Gen Z summaries.
* **Components**:

  * `pdf_parser.py`: Preprocesses PDFs on the fly.
  * `demo.ipynb`: Notebook to launch the Gradio server.

---

## 🎮 Quickstart Demo

1. **Navigate** to the `demo/` folder:

   ```bash
   cd demo
   ```
2. **Install** any extra demo-specific requirements (if needed).
3. **Run** the demo notebook or script:

   ```bash
   jupyter lab demo.ipynb
   ```
4. **Copy** the Gradio link and paste it in your browser to try Sumarizz.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for:

* Bug reports and feature requests.
* Data quality improvements.
* UI/UX enhancements.

Before you submit:

* Ensure code passes linting and tests.
* Add or update documentation as needed.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

