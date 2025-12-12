# Multimodal Satire Classifier in Italian


![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1-red)
![License](https://img.shields.io/badge/license-MIT-green)


## Description
This project implements **unimodal** (text-only) and **multimodal** (text + audio) classifiers to detect satirical content.  
It leverages **Italian BERT** for text and **HuBERT** for audio.

The main objective of this project is to **verify that the multimodal component performs better than the unimodal one**, similar to how humans rely not only on words but also on **tone of voice** and prosody to understand satire.

A key contribution of this project is the **creation of a multimodal dataset** containing text-audio pairs labeled for satire detection.  
The dataset is **not publicly available** due to privacy restrictions, but **researchers can request access** by contacting the author.
Note: The notebooks and scripts are ready to run, but they **require access to the dataset**. Without it, the code cannot be executed.

The project is optimized for execution on **Google Colab**, making it easy to reproduce experiments without local setup.

