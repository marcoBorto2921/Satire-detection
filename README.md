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

The project is optimized for execution on **Google Colab**, which allows you to easily reproduce experiments without a local setup.  
Using Colab also enables access to **free GPU resources**, but the notebooks can also be run locally on a PC if a suitable Python environment and hardware are available.



## Project Structure

The repository is organized as follows:

- `Notebooks/` : Jupyter notebooks for training, evaluation, and multimodal fusion experiments.
  Each notebook contains clear instructions and code to reproduce the results. Example notebooks:
    - `text_trainin.ipynb` – Training and evaluation of the text-only model.
    - `audio_training.ipynb` – Training and evaluation of the audio-only model.
    - `multimodal_training_attention.ipynb` – Training of the multimodal model with attention mechanism.
    - `multimodal_training_lateFusion.ipynb` – Multimodal Late Fusion.
    - `results.ipynb` – Cross-validation and external dataset evaluation.

- `Data/` : Folder for dataset files (**not included in the repository** due to privacy restrictions). 
  Researchers who have access should place the dataset here following the expected folder structure.  

- `README.md` : Main documentation with project description, instructions, and results summary.

- `requirements.txt` : Python dependencies needed to run the notebooks.


## How to Run / Usage

This project is optimized to run on **Google Colab**. To reproduce the experiments, follow these steps:

1. **Prepare the dataset**  
   Create a folder named `Data/` (or any folder of your choice) in your Google Drive and place the preprocessed dataset folds inside.  
   Each fold should have its own subfolder (`Fold_1`, `Fold_2`, ..., `Fold_10`) containing the CSV files (`train.csv`, `validation.csv`, `test.csv`) and the corresponding audio files.  
   Additionally, place the CSV file `external_dataset.csv` in the `Data/` folder for evaluation on the external dataset.

3. **Update paths in notebooks**  
   In each notebook, go to the **main section** and replace the placeholder paths (e.g., `insert path here`) with the actual paths to your dataset folds and the folders where you want to save results.  

4. **Run the training notebooks**  
   Execute the notebooks for text-only, audio-only, and multimodal models in the desired order.  
   The notebooks will save model outputs, predictions, and probabilities in the specified folders.

5. **Run the results notebook**  
   Open `results.ipynb` and set the path to the folder containing the saved outputs from the training.  
   This notebook will compute cross-validation metrics and evaluate performance on any external datasets you have, generating a summary folder with all results.

Following these steps ensures that you can reproduce all training, evaluation, and fusion experiments directly in Colab without modifying the code.


### Dataset Description

The multimodal dataset was created for satire detection in Italian and includes **text-audio pairs** labeled as `satire` or `no_satire`.  

- **Source**: Videos collected from satirical and non-satirical YouTube channels and playlists. Metadata (title, ID, duration, URL, channel) were stored in CSV files.  
- **Audio Segmentation**: Full audio tracks were split into 15–30 second clips using acoustic-based silence detection to preserve natural speech and prosody with **pydub**.  
- **Transcription**: Each segment was transcribed with **OpenAI Whisper (medium, multilingual)**, ensuring accurate text representation.  
- **Dataset Size**: 10,000 instances total, 5,000 satirical and 5,000 non-satirical, perfectly balanced.  
- **Cross-Validation**: 10 channel-based folds were created to ensure independence between training and test sets. Each fold includes training, validation, and test splits with stratified sampling to maintain class balance.  
- **External Evaluation**: A separate `external_dataset.csv` is included for testing on unseen sources.  

This structure ensures robust model training and evaluation, minimizing bias from speaker or channel-specific features.

### Access to Dataset

The dataset used in this project is **not publicly available** due to privacy restrictions.  
Researchers interested in accessing the dataset can request it by contacting the author:

- **Marco Bortolotti** – marco03.bortolotti@edu.unife.it

Requests will be considered for **research purposes only** and may require signing a data usage agreement.



## Results / Expected Performance

### Cross-Validation Results

The following table summarizes the performance of the unimodal and multimodal models after cross-validation (outliers removed):

| Model                          | Accuracy | F1-score | Cohen’s Kappa |
|--------------------------------|---------|----------|---------------|
| **text_training**            | 0.825   | **0.824** | 0.649         |
| **aaudio_training**           | 0.799   | **0.798** | 0.598         |
| **Multimodal: Late Fusion**     | 0.849   | **0.849** | 0.698         |
| **Multimodal: Cross Attention**    | 0.842   | **0.842**    | 0.684         |

### External Dataset Results

After retraining the best models from cross-validation on the full dataset, their performance was evaluated on the external dataset:

| Model                          | Accuracy | F1-score | Cohen’s Kappa |
|--------------------------------|---------|----------|---------------|
| BERT fine-tuned                | 0.896   | 0.895    | 0.792         |
| HuBERT fine-tuned              | 0.804   | 0.798    | 0.609         |
| Multimodal: Late Fusion        | 0.917   | 0.917    | 0.834         |
| **Multimodal: Cross Attention** | 0.947   | **0.948** | 0.894         |

**Observations:**

- The multimodal models consistently outperform unimodal models, confirming that integrating audio features improves satire detection.
- Cross Attention fusion achieves the highest performance on the external dataset, demonstrating strong generalization.
- The text-only model (BERT fine-tuned) performs better than the audio-only model (HuBERT fine-tuned), but combining both modalities yields a significant improvement.








