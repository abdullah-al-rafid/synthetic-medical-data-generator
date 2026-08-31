# 🧬 Synthetic Medical Data Generator

A data-driven project for generating, evaluating, comparing, and exploring synthetic heart disease datasets while preserving important statistical and structural characteristics of the original data.

The project explores multiple synthetic data generation approaches, evaluates their performance, and combines selected generative approaches into a hybrid synthetic data generation pipeline.

An interactive **Streamlit web application** is also provided for exploring the generated synthetic datasets.

---

## 🌐 Live Application

🚀 **Live Demo:**  
https://synthetic-medical-data-generator-evj4orlomhe9czfcfqczju.streamlit.app/

The web application allows users to:

- Select different synthetic data generation models
- Preview generated synthetic datasets
- View dataset statistics
- Inspect dataset dimensions
- Check heart disease rate
- Check missing values
- Check duplicate rows
- Explore the final hybrid dataset

---

## 🎯 Project Objective

The main objective of this project is to investigate how synthetic medical data can be generated and evaluated for research and machine learning workflows.

The project focuses on:

- Generating realistic synthetic heart disease records
- Comparing multiple synthetic data generation techniques
- Evaluating statistical similarity between real and synthetic data
- Evaluating machine learning utility
- Analyzing the quality of generated synthetic datasets
- Developing a hybrid synthetic data generation approach
- Exploring privacy-aware synthetic data generation
- Providing an interactive interface for synthetic data exploration

---

## 🧠 Synthetic Data Generation Methods

This project experiments with several synthetic data generation techniques:

- **CTAB-GAN**
- **CTGAN**
- **CopulaGAN**
- **Gaussian Copula**
- **TVAE**
- **medGAN**
- **medGAN + CTAB-GAN Hybrid**

The generated datasets are stored and organized according to their generation method.

---

## 🔬 Methodology

The overall workflow of the project is:

```text
Original Heart Disease Dataset
              ↓
      Data Preprocessing
              ↓
      Data Cleaning &
       Transformation
              ↓
    Feature Engineering
              ↓
 ┌────────────┼────────────┐
 ↓            ↓            ↓
CTGAN      CTAB-GAN       TVAE
 ↓            ↓            ↓
CopulaGAN  Gaussian       medGAN
           Copula           ↓
 └────────────┼────────────┘
              ↓
   Synthetic Dataset
       Evaluation
              ↓
      Model Comparison
              ↓
       Model Ranking
              ↓
     Hybrid Development
              ↓
   Final Hybrid Dataset
              ↓
    Streamlit Application
```

---

# 📊 Dataset

The project uses a processed heart disease dataset.

The processed dataset used during the project contains:

- **16,000 records**
- **25 features**

The generated synthetic datasets contain:

- **10,000 synthetic records**
- **25 features**

The original medical dataset is **not included** in this public repository.

---

## 🧾 Dataset Features

The processed dataset contains the following 25 features:

```text
Age
Blood Pressure
Cholesterol Level
BMI
Sleep Hours
Triglyceride Level
Fasting Blood Sugar
CRP Level
Homocysteine Level
Gender_Male
Exercise Habits_Low
Exercise Habits_Medium
Smoking_Yes
Family Heart Disease_Yes
Diabetes_Yes
High Blood Pressure_Yes
Low HDL Cholesterol_Yes
High LDL Cholesterol_Yes
Alcohol Consumption_Low
Alcohol Consumption_Medium
Stress Level_Low
Stress Level_Medium
Sugar Consumption_Low
Sugar Consumption_Medium
Heart Disease Status_Yes
```

---

## 🧹 Data Preprocessing

The data preparation workflow includes:

```text
Raw Dataset
      ↓
Data Loading
      ↓
Dataset Inspection
      ↓
Missing Value Handling
      ↓
Duplicate Detection / Removal
      ↓
Outlier Analysis
      ↓
Categorical Encoding
      ↓
Feature Transformation
      ↓
Scaling / Normalization
      ↓
Processed Dataset
```

The processed feature representation is subsequently used by the synthetic data generation models.

---

## 🤖 medGAN

The project includes a **medGAN-based synthetic data generation pipeline**.

The implementation uses:

- TensorFlow
- Autoencoder-based representation learning
- Generator network
- Discriminator network
- Random noise
- Batch normalization
- Synthetic sample generation

The generated medGAN output is converted into the required tabular feature representation for further processing and evaluation.

---

## 🧠 CTAB-GAN

The project also includes a **CTAB-GAN-based tabular synthetic data generation pipeline**.

CTAB-GAN is used to generate synthetic tabular records while modeling relationships between different feature types.

The repository contains the associated CTAB-GAN implementation, preprocessing components, synthesizer components, and evaluation code.

---

# 🔗 Hybrid Synthetic Data Generation

A major component of this project is the development of a hybrid synthetic data generation workflow.

The project combines synthetic outputs from **medGAN and CTAB-GAN** to produce hybrid synthetic datasets.

```text
                 medGAN
                    │
                    ▼
           medGAN Synthetic Data
                    │
                    │
                    ▼
                CTAB-GAN
                    │
                    ▼
        Hybrid Synthetic Processing
                    │
                    ▼
          Final Hybrid Dataset
```

The hybrid pipeline produces the following important outputs:

```text
final_model/
├── medgan_baseline.csv
├── medgan_ctabgan.csv
└── final_hybrid_dataset.csv
```

---

## 📈 Model Evaluation

The project evaluates and compares synthetic datasets generated by different models.

The main evaluation outputs include:

```text
Final_Model_Ranking.csv
Model_Comparison.csv
```

These files contain the comparison and ranking results generated during the evaluation workflow.

The evaluation process is used to compare the investigated synthetic data generation approaches based on the implemented evaluation methodology.

---

## 🏆 Model Ranking

The project includes a final model ranking file:

```text
Final_Model_Ranking.csv
```

The ranking represents the results produced by the implemented evaluation workflow.

Synthetic data quality can be considered from multiple perspectives, including:

```text
Statistical Similarity
        +
Machine Learning Utility
        +
Data Quality
        +
Privacy Considerations
```

---

# 🗂️ Project Structure

```text
synthetic-medical-data-generator/
│
├── app/
│   └── app.py
│
├── model/
│   ├── ctabgan.py
│   │
│   ├── eval/
│   │   └── evaluation.py
│   │
│   ├── pipeline/
│   │   └── data_preparation.py
│   │
│   └── synthesizer/
│       ├── ctabgan_synthesizer.py
│       └── transformer.py
│
├── models/
│   └── heart_medgan.npy
│
├── final_model/
│   ├── final_hybrid_dataset.csv
│   ├── medgan_baseline.csv
│   ├── medgan_ctabgan.csv
│   └── scaler.pkl
│
├── results/
│   ├── Final_Model_Ranking.csv
│   └── Model_Comparison.csv
│
├── synthetic_data/
│   ├── CTABGAN_Synthetic_Data.csv
│   ├── CTGAN_Synthetic_Data.csv
│   ├── CopulaGAN_Synthetic_Data.csv
│   ├── GaussianCopula_Synthetic_Data.csv
│   ├── medGAN_Synthetic_Data.csv
│   └── TVAE_Synthetic_Data.csv
│
├── main.ipynb
├── metadata.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🖥️ Streamlit Application

The project includes an interactive web application built with **Streamlit**.

The application is implemented in:

```text
app/app.py
```

### Main application features

#### 🔹 Model Selection

Users can select from available synthetic datasets, including:

```text
medGAN
CTGAN
TVAE
Gaussian Copula
CopulaGAN
CTAB-GAN
medGAN + CTAB-GAN
Current Hybrid
```

#### 🔹 Dataset Statistics

The application provides:

```text
Records
Features
Heart Disease Rate
Missing Values
```

#### 🔹 Synthetic Data Preview

Users can select the number of rows to preview and inspect generated synthetic records interactively.

#### 🔹 Dataset Information

The application also displays:

```text
Dataset Shape
Duplicate Rows
```

---

# 🛠️ Technologies Used

## 💻 Programming

- Python
- Jupyter Notebook

## 📊 Data Processing

- NumPy
- Pandas
- SciPy
- Scikit-learn

## 📈 Visualization

- Matplotlib
- Seaborn

## 🧬 Synthetic Data Generation

- SDV
- CTAB-GAN
- CTGAN
- CopulaGAN
- Gaussian Copula
- TVAE
- medGAN

## 🤖 Deep Learning

- TensorFlow
- PyTorch

## 🌐 Web Application

- Streamlit

## 🔧 Development Tools

- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/abdullah-al-rafid/synthetic-medical-data-generator.git
cd synthetic-medical-data-generator
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate the Virtual Environment

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Run the Jupyter Notebook

Open `main.ipynb` using **Jupyter Notebook** or **JupyterLab**.

```bash
jupyter notebook
```

---

## 6. Run the Streamlit Application

From the project root directory:

```bash
streamlit run app/app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 📁 Results

Model evaluation and comparison results are available in the `results/` directory.

```text
results/
├── Final_Model_Ranking.csv
└── Model_Comparison.csv
```

The synthetic datasets generated by the individual models are organized as:

```text
synthetic_data/
├── CTABGAN_Synthetic_Data.csv
├── CTGAN_Synthetic_Data.csv
├── CopulaGAN_Synthetic_Data.csv
├── GaussianCopula_Synthetic_Data.csv
├── medGAN_Synthetic_Data.csv
└── TVAE_Synthetic_Data.csv
```

The final hybrid model outputs are available in:

```text
final_model/
├── final_hybrid_dataset.csv
├── medgan_baseline.csv
├── medgan_ctabgan.csv
└── scaler.pkl
```

These datasets and evaluation results can be used for further analysis, comparison, and research.

---

# 📋 Metadata

The project includes:

```text
metadata.json
```

This file contains metadata associated with the processed dataset and feature representation.

---

# 🔐 Data & Privacy

The original source medical dataset is **not included** in this public repository.

The repository contains the project implementation, synthetic datasets, model artifacts, and evaluation results intended for public sharing.

Synthetic data should not automatically be considered completely anonymous or privacy-preserving.

Users should evaluate potential privacy risks before using synthetic medical data in sensitive applications.

Users should also verify the **licensing and redistribution terms** of any external dataset, implementation, or third-party resource before using or redistributing it.

---

# ⚠️ Disclaimer

This project is developed for **academic, research, and educational purposes**.

Synthetic medical data generated by this project should **not be considered a replacement for clinically validated real-world medical data**.

The generated datasets should not be used for:

- Medical diagnosis
- Clinical decision-making
- Patient treatment
- Professional medical advice
- Real-world healthcare decisions

---

# 🔮 Future Improvements

Potential future improvements include:

- Improve synthetic data quality and realism
- Add additional synthetic data generation models
- Perform more comprehensive privacy evaluation
- Implement automated model selection
- Develop an advanced synthetic data generation interface
- Improve visualization and reporting
- Explore differential privacy techniques
- Evaluate synthetic data across additional downstream machine learning tasks
- Improve statistical similarity analysis
- Add automated correlation and distribution comparison
- Expand hybrid synthetic data generation techniques
- Support additional medical datasets

---

# 📚 Research Direction

The project can be further extended toward research in:

```text
Synthetic Medical Data
        │
        ├── Data Fidelity
        │
        ├── Data Utility
        │
        ├── Privacy
        │
        ├── Statistical Similarity
        │
        ├── Machine Learning Utility
        │
        └── Generative Modeling
```

A comprehensive synthetic-data evaluation framework can consider the balance between:

```text
Data Fidelity
      +
Data Utility
      +
Privacy
```

rather than relying exclusively on a single evaluation metric.

---

# 👥 Project Contributors

This project was developed collaboratively by:

## 👨‍💻 Abdullah Al Rafid

**Computer Science & Engineering**  
**Daffodil International University**

🔗 **GitHub:**  
https://github.com/abdullah-al-rafid

## 👩‍💻 Jannatul Islam Asa

**Computer Science & Engineering**  
**Daffodil International University**

🔗 **GitHub:**  
https://github.com/jannatul-islam-asa

---

# 👨‍💻 Author

## Abdullah Al Rafid

**Computer Science & Engineering**  
**Daffodil International University**

🔗 **GitHub:**  
https://github.com/abdullah-al-rafid

🔗 **LinkedIn:**  
https://www.linkedin.com/in/abdullah-al-rafid/

🔗 **Project Repository:**  
https://github.com/abdullah-al-rafid/synthetic-medical-data-generator

🔗 **Live Application:**  
https://synthetic-medical-data-generator-evj4orlomhe9czfcfqczju.streamlit.app/

---

# 📄 License

This project is intended for **academic and research purposes**.

Please review the licenses of individual third-party implementations, libraries, and datasets before redistribution or commercial use.

---

# ❤️ Synthetic Medical Data Generator

### Generate • Compare • Evaluate • Explore

```text
Synthetic Medical Data
          ↓
Multiple Generative Models
          ↓
Synthetic Dataset Evaluation
          ↓
Model Comparison
          ↓
Hybrid Generation
          ↓
Final Synthetic Dataset
          ↓
Interactive Web Application
```
