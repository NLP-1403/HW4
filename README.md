# Persian Gender Neutralization (NLP HW4)

This repository contains the code and report for **Homework 4** of the *Natural Language Processing* course (Spring 2024, Sharif University of Technology).  
The project focuses on **gender neutralization for Persian text**: collecting and preparing a dataset, and then performing **parameter-efficient fine-tuning (PEFT)** using **LoRA** (with **int8 quantization**) on a **Llama 3** model to rewrite gender-biased sentences into more gender-neutral alternatives.

## Repository Contents

- **`HW4_Dataset.ipynb`**  
  Dataset preparation pipeline, including:
  - Collecting Persian text samples from **X (Twitter)** using a Selenium-based scraper
  - Using **LLM-based labeling** (ChatGPT) to create paired examples of *biased input → neutralized target*
  - Reformatting samples into an instruction-style dataset (fields like `dialogue` and `summary`)

- **`HW4_FineTuning.ipynb`**  
  Fine-tuning and evaluation pipeline, including:
  - Installing dependencies (`llama-recipes`, `evaluate`, `bert_score`, `hazm`, …)
  - Hugging Face authentication (requires access to **Meta Llama** weights)
  - **LoRA fine-tuning** with **8-bit / int8** loading to reduce VRAM usage
  - Notes on adapting **context length** to prevent out-of-memory (OOM) errors on smaller GPUs

- **`document/`**  
  LaTeX sources for the course report (e.g., `Report.tex` and the custom class).

- **`stopwords/`**  
  Utility resources (e.g., Persian punctuation/diacritics list in `chars.txt`) used during text normalization/cleaning.

## Dataset

The resulting dataset is published on Hugging Face:

- https://huggingface.co/datasets/AmirMohammadFakhimi/gender_neutralize

## How to Run

### 1) Environment

These notebooks were run with **Python 3.x** (the fine-tuning notebook metadata indicates Python 3.10).  
Recommended: use a fresh environment (conda/venv) and run the notebooks in order.

### 2) Dataset preparation (`HW4_Dataset.ipynb`)

Key requirements:
- `pandas`, `tqdm`
- `hazm` for Persian normalization
- Optional: credentials/config for X scraping (the notebook reads a local `.env` with a `[TWITTER]` section)

Notes:
- The notebook uses (or references) a Selenium-based X/Twitter scraper and executes it via shell commands.
- If you do not plan to scrape data yourself, you can skip scraping steps and focus on the released Hugging Face dataset.

### 3) Fine-tuning (`HW4_FineTuning.ipynb`)

Key requirements:
- GPU strongly recommended (the notebook mentions running on **T4 ×2**)
- Hugging Face login with permission to access **Llama 3** weights

The notebook installs required packages (including `llama-recipes`) and runs PEFT fine-tuning with LoRA and int8 quantization.

**OOM troubleshooting:** if you hit CUDA out-of-memory, reduce the training `context_length` in the notebook configuration as suggested in the notebook notes.

## Method (High-level)

1. **Data collection** from Persian social media queries and curated prompts.
2. **LLM-assisted labeling** to obtain neutralized targets for biased inputs.
3. **Instruction-style formatting** for supervised fine-tuning.
4. **PEFT fine-tuning (LoRA)** on Llama 3 with memory optimizations (int8).
5. **Evaluation** using common text metrics (e.g., BERTScore) and Persian processing utilities (Hazm).

## License

This repository includes a `LICENSE` file—please refer to it for usage terms.

## Acknowledgments

- Course: NLP (Sharif University of Technology, Spring 2024)
- External tooling referenced in the dataset notebook includes a Selenium-based X/Twitter scraping project.
- Hugging Face ecosystem and `llama-recipes` for fine-tuning utilities.
