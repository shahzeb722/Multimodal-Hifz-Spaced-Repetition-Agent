# Multimodal Hifz Spaced-Repetition Agent

An advanced, AI-driven spaced-repetition system designed for memorizing the Quran (Hifz). This agent leverages state-of-the-art multimodal machine learning to evaluate student recitations, offering granular scoring and individualized spaced-repetition schedules.

## 🚀 Tech Stack

- **Speech Recognition:** Whisper ASR
- **NLP & Semantics:** AraBERT
- **Machine Learning Models:** XGBoost (with PCA)
- **Reinforcement Learning:** PPO v2 (5D state environment)
- **Quality & Performance:** Tajweed Scoring, Lahn Classifier, Mutashabihat Detector, Vocal Anxiety Profiler
- **Interface:** Gradio UI

## ✨ Features

- **Robust Arabic Preprocessing:** Includes Uthmani-to-Standard normalization, intelligent Tashkeel stripping/preservation, and automatic handling of continuous/unspaced Arabic inputs.
- **Multimodal Perception Pipeline:** Integrates transcription (Whisper), semantic similarity (AraBERT), and acoustic feature extraction (librosa).
- **Vocal Anxiety Detection:** Analyzes F0 instability and RMS energy fade to identify nervousness or anxiety during recitation, factoring this into the memory retention modeling.
- **Tajweed Scoring Module:** Provides an end-to-end evaluation of tilawah (recitation) utilizing classical Tajweed rules. Computes sub-scores for:
  - **Pronunciation (Makhraj/Sifaat):** Character precision scoring
  - **Phoneme:** Semantic verification via AraBERT
  - **Cadence:** Rhythm consistency and energy variance
  - **Madd:** Elongation accuracy detection
  - **Waqf:** Proper pause placement ratio
  - **Clarity:** Spectral centroid and zero-crossing rate calculation
- **Spaced-Repetition Engine:** Models forgotten probabilities based on a modified Ebbinghaus curve that incorporates tajweed complexity, past mistakes, vocal anxiety, and review intervals.

## 📂 Project Structure

- `SP24_BCS_126_UMER_FRAZ_AHMED_Assignment_4.ipynb`: The main notebook containing the full pipeline, from mock data generation to EDA, feature engineering, NLP preprocessing, perception pipeline, and tajweed scoring.
- `hifz_agent.ipynb`: Secondary agent experimentation.

## 📊 Evaluation & EDA

The project features comprehensive exploratory data analysis, plotting distributions of the forgetting class, analyzing Ebbinghaus time-decay, and evaluating feature correlation via custom visualization pipelines (`seaborn` & `matplotlib`).

## 🛠 Usage

1. Clone the repository.
2. Install the necessary dependencies (as stated in the primary notebook):
   ```bash
   pip install stable-baselines3[extra] gymnasium transformers xgboost scikit-learn accelerate librosa soundfile seaborn nltk gradio
   ```
3. Run the cells within the Jupyter Notebook to experience the grading and RL modeling interactively.

## 📜 License

This project was built for educational and research purposes.
