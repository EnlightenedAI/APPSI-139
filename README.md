# APPSI-139: A Parallel Corpus of English Application Privacy Policy Summarization and Interpretation

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2604.27550)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

## 📖 Introduction

To address the challenges of understanding complex and legally dense privacy policies, we introduce **APPSI-139**, the first parallel corpus of English privacy policies annotated by legal experts, aimed at providing user-friendly interpretations. 

We also propose the **TCSI-pp-V2** framework, a multi-task hybrid summarization model that effectively balances computational efficiency with accuracy. Evaluations show that our system outperforms general-purpose models (e.g., GPT-4o) in both readability and reliability. APPSI-139 empowers users to make more informed privacy decisions, significantly enhancing the comprehension and interpretation of digital privacy policies.

> ⚖️ **Annotation Quality:** The annotation was conducted by five experts holding master’s degrees in law and official lawyer certifications. Annotators underwent systematic training and guideline walkthroughs to ensure high consistency and reliability. 
> * [View Annotation Guidelines](./Documents/Annotation_Guidelines.png) | [Annotation Examples](https://github.com/EnlightenedAI/CAPP-130)

---

## 🚀 News

* **[2026/04]** APPSI-139 has been accepted to **ACL 2026**.
* **[2025/10]** Core datasets and code have been released and will be continuously maintained.

---

## 🛠️ Quick Start

### Dependencies
```bash
pip install -r requirements.txt

```

### Training

Train the TCSI-pp-V2 model:

```bash
python ./TCSI-pp-V2/[MODEL_NAME]_rewrite_ddp2_model.py

```

### Inference

Run inference on your privacy policy data:

```bash
python ./Infer/main.py --topic_list [TOPIC_LIST] --data [PRIVACY_POLICY_PATH]

```

---

## 💡 In-Context Learning Template

We utilize a structured instruction template to guide model performance:

```text
- Task Description: You are an expert in privacy policies. Analyze the provided privacy policy 
  text sentence by sentence and summarize it in simple, non-expert friendly language.

- Example 1: 
  - Clause: "..." 
  - Summary: "..."
- Example 2: 
  - Clause: "..." 
  - Summary: "..."

- Based on the examples above, summarize the following: "..."
- Summary:

```

---

## 📊 Model & Data

The `mt5_mtl_model` (based on TCSI-pp-V2) is hosted on [Hugging Face](https://huggingface.co/EnlightenedAI/APPSI-139/tree/main). All associated model parameters and checkpoints are available for download.

**Sample Summarization Result:**


---

## 📜 Citation

If you find this work helpful or use it in your research, please cite our paper:

```bibtex
@inproceedings{zhu-etal-2026-appsi,
    title = "{APPSI}-139: A Parallel Corpus of {E}nglish Application Privacy Policy Summarization and Interpretation",
    author = "Zhu, Pengyun  and
      Sun, Qiheng  and
      Wen, Long  and
      Wang, Yanbo  and
      Cao, Yang  and
      Liu, Junxu  and
      Xiong, Deyi  and
      Liu, Jinfei  and
      Wang, Zhibo  and
      Ren, Kui",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.168/",
    doi = "10.18653/v1/2026.acl-long.168",
    pages = "3681--3706",
    ISBN = "979-8-89176-390-6"
}

```

---

## 📌 Maintenance

This repository is under continuous development. Contributions and suggestions are welcome!

```
