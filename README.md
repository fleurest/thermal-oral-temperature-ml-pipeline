# Infrared Thermography Temperature Prediction & Fever Classification

**About**
End-to-end machine learning pipeline for oral temperature regression using infrared thermography and multimodal features.

**License**
[MIT License](LICENSE)

---

A machine learning project that predicts oral body temperature and detects fever from thermal imaging data using various regression and classification models.

## 📋 Project Overview

This project contemplates two supervised learning tasks using infrared thermography data:

1. **Regression Task**: Predict oral temperature readings (both fast mode and monitor mode) from thermal images and environmental data
2. **Classification Task**: Detect if a person has a fever (temperature ≥ 37.5°C) based on the same features

## 🎯 Problem Statement

Using environmental information and thermal image readings, this project aims to:
- Predict `aveOralF` (oral temperature in fast mode)
- Predict `aveOralM` (oral temperature in monitor mode)
- Classify whether the person has a fever based on both temperature measurements

## 📊 Dataset

**Source**: [UCI Machine Learning Repository - Infrared Thermography Temperature Dataset](https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset)

**Features**: 33 features including:
- Gender, age, ethnicity
- Ambient temperature and humidity
- Distance from thermal camera
- Temperature readings from various facial regions in thermal images

**Samples**: Multiple patient measurements with labeled oral temperature outputs

## 🔬 Models Implemented

### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- **XGBoost Regressor** ⭐ (Best performer)
- K-Nearest Neighbors Regressor
- Decision Tree Regressor

### Classification Models
- **Logistic Regression** ⭐ (Best performer)
- Random Forest Classifier
- XGBoost Classifier
- K-Nearest Neighbors Classifier
- Decision Tree Classifier
- Multi-Layer Perceptron (Neural Network)

## 📈 Key Results

### Regression Performance
- **XGBoost on aveOralM**: R² = 0.78 on test data
- Strong generalization from training to test set

### Classification Performance
- **Logistic Regression on FeverM**:
  - Accuracy: **97.1%**
  - F1 Score: **0.85**
  - Confusion Matrix: 181 TN, 1 FP, 5 FN, 17 TP
  
High recall prioritized to minimize false negatives (critical for fever detection in healthcare settings).

## 🛠️ Technologies Used

- **Python 3.5+**
- **Data Processing**: NumPy, Pandas, SciPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Deep Learning**: TensorFlow, Keras
- **Visualization**: Matplotlib, Seaborn
- **Data Source**: ucimlrepo

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/fleurest/thermal-oral-temperature-ml-pipeline.git
cd thermal-oral-temperature-ml-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook thermal-oral-temperature-pipeline.ipynb
```

4. The notebook will automatically download the dataset from UCI ML Repository

## 📁 Project Structure

```
thermal-oral-temperature-ml-pipeline/
│
├── thermal-oral-temperature-pipeline.ipynb    # Main project notebook
├── data/                                       # Dataset directory
├── models/                                     # Saved models directory
├── README.md                                   # Project documentation
└── requirements.txt                            # Python dependencies
```

## 🔍 Project Highlights

- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Multiple model comparison and evaluation
- ✅ Hyperparameter tuning and optimization
- ✅ Confusion matrices and detailed performance metrics
- ✅ Focus on practical healthcare application
- ✅ Strong model generalization on unseen test data

## 📝 Key Insights

- XGBoost outperformed traditional regression models for temperature prediction
- Logistic Regression achieved excellent classification performance with simple interpretability
- High precision and recall balanced for fever detection critical in healthcare
- Environmental features combined with thermal readings provide robust predictions

## 👤 Author

Fleur Edwards
[LinkedIn](https://linkedin.com/in/fleureedwards) | [GitHub](https://github.com/fleurest)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by UCI Machine Learning Repository
- Project completed as part of Goldsmiths, University of London coursework