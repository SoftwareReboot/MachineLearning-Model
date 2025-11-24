🤖 MachineLearning-Model

A compilation of machine learning models developed as part of our university coursework and research projects.  
This repository showcases predictive and classification models exploring various algorithms and techniques across different datasets and problem domains.

---

📚 Overview

This project demonstrates hands-on implementations of key machine learning concepts learned throughout our studies.  
Each notebook or script represents a distinct project that focuses on:
- Data preprocessing and feature engineering
- Model training and optimization
- Performance evaluation and visualization
- Real-time predictions and applications

The goal is to create an organized and educational collection of ML projects that can serve as reference materials for future learning and development.

---

🧠 Models Included

| Model | File | Description | Techniques |
|--------|------|--------------|-------------|
| Car Purchasing Model | `Car Purchasing Model.ipynb` | Predicts how much a customer is willing to spend on a new car using demographic and behavioral data. | Linear Regression |
| Decision Tree Model | `Decision Tree.ipynb` | Implementation and improvement of a Decision Tree classifier/regressor using datasets such as customer and loan data. | Decision Tree Classifier, Data Splitting, Entropy & Gini |
| XGBoost Classifier| `xgboost_model.ipynb` | Gradient boosting model for classification tasks with feature importance analysis and cross-validation. | XGBoost, Gradient Boosting, Hyperparameter Tuning |
| Gender Classification (SVM) | `gender_classifier.py` | Real-time gender classification using Linear SVM trained on webcam face data with 85%+ accuracy. | Linear SVM, Computer Vision, Face Detection, Real-time Prediction |
| Expression Classification (SVM) | `expression_classifier.py` | Facial expression recognition (Neutral vs Smiling) using Linear SVM with webcam integration. | Linear SVM, Image Processing, Feature Extraction |

---

 🎯 Project Highlights

 🚗 Car Purchasing Model
- Goal: Predict customer spending on new cars
- Algorithm: Linear Regression
- Features: Age, annual salary, credit card debt, net worth
- Output: Predicted purchase amount in dollars

🌳 Decision Tree Model
- Goal: Classification and regression with interpretable tree structure
- Algorithms: Decision Tree Classifier/Regressor
- Features: Entropy, Gini index, pruning techniques
- Datasets: Customer profiles, loan data

 🚀 XGBoost Classifier
- Goal: High-performance gradient boosting for classification
- Features: 
  - Feature importance visualization
  - Learning curves analysis
  - Cross-validation
  - Hyperparameter tuning with GridSearchCV
- Performance: Optimized accuracy with minimal overfitting

 👤 Gender Classification (Linear SVM)
- Goal: Real-time gender detection from webcam feed
- Accuracy: 85-90% with diverse training data
- Features:
  - Multi-person training support (5+ people recommended)
  - Live face detection with Haar Cascades
  - Confidence scores and margin visualization
  - Model persistence (save/load)
  - Color-coded predictions (Blue=Male, Pink=Female)
- Technical Details:
  - 64x64 grayscale images = 4,096 features per face
  - Linear kernel SVM with C=1.0
  - Support Vector analysis included
  - StandardScaler for feature normalization

 😊 Expression Classification (Linear SVM)
- Goal: Detect facial expressions (Neutral vs Smiling)
- Accuracy: 90%+ on training subject, 70-80% on new people
- Features:
  - Webcam-based data collection
  - Real-time expression detection
  - Decision boundary visualization
  - Margin distance display
- Use Cases:
  - Emotion detection systems
  - User engagement monitoring
  - Interactive applications

---

 🧩 Repository Structure

```
MachineLearning-Model/
│
├── Regression Models/
│   ├── Car Purchasing Model.ipynb
│   ├── CPM.py
│   └── Car_Purchasing_Data.csv
│
├── Tree-Based Models/
│   ├── Decision Tree.ipynb
│   ├── xgboost_model.ipynb
│   ├── Customer.csv
│   └── loan.csv
│
├── SVM Models/
│   ├── gender_classifier.py
│   ├── expression_classifier.py
│   └── svm_testing_suite.py
│
├── Visualizations/
│   ├── diabetes.png
│   ├── gender_confusion_matrix.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```

---

🧪 Technologies Used

 Programming & Core Libraries
- Python 3.8+
- NumPy - Numerical computing and array operations
- Pandas- Data manipulation and analysis
- Scikit-learn - Machine learning algorithms (SVM, Decision Trees, preprocessing)

Specialized ML Libraries
- XGBoost - Gradient boosting framework
- OpenCV - Computer vision and webcam integration

 Visualization
- Matplotlib - Basic plotting and visualization
- Seaborn - Statistical data visualization

Development Tools
- Jupyter Notebook- Interactive development and documentation
- Git/GitHub - Version control and collaboration

---

 🎯 Objectives

-  Apply machine learning algorithms to real-world datasets
-  Visualize model performance and interpret results
-  Compare performance between algorithms (SVM, Decision Trees, XGBoost, Regression)
-  Build real-time prediction systems with computer vision
-  Create an accessible and educational ML model collection
-  Demonstrate proper ML workflow: data collection → preprocessing → training → evaluation

---

 🚀 Getting Started

 Installation

1. Clone the repository
```bash
git clone https://github.com/SoftwareReboot/MachineLearning-Model.git
cd MachineLearning-Model
```

2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

Running the Models

 Car Purchasing Model
```bash
jupyter notebook "Car Purchasing Model.ipynb"
```

 Decision Tree Model
```bash
jupyter notebook "Decision Tree.ipynb"
```

#### XGBoost Model
```bash
jupyter notebook "xgboost_model.ipynb"
```

 Gender Classification (SVM)
```bash
python gender_classifier.py
```
Training Tips:
- Get 5-6 friends per gender for best accuracy
- Ensure good lighting and face visibility
- Each person takes only ~2 seconds to collect
- Model saves automatically for later use

Expression Classification (SVM)
```bash
python expression_classifier.py
```
**Note:** Train on multiple people for better generalization across different faces.

---

 📊 Model Performance Comparison

| Model | Accuracy | Training Time | Real-time? | Best Use Case |
|-------|----------|---------------|------------|---------------|
| Linear Regression | R² > 0.85 | Fast | ❌ | Continuous predictions |
| Decision Tree | 80-90% | Fast | ❌ | Interpretable classification |
| XGBoost | 90-95% | Medium | ❌ | High-accuracy classification |
| Gender SVM | 85-90% | Fast | ✅ | Real-time face classification |
| Expression SVM | 70-90% | Fast | ✅ | Emotion detection |

---

🔬 Key Learnings

### Linear SVM Deep Dive
- **Why SVM excels with high dimensions:** 4,096 pixel features → easier to find separating hyperplane
- **Support Vectors:** Only 20-30% of training data actually matters for the decision boundary
- **Optimal Separation:** Maximum margin ensures better generalization
- **Real-world application:** Perfect for computer vision tasks with limited training data

XGBoost Insights
- Gradient Boosting: Builds trees sequentially, each correcting previous errors
- Regularization:** L1 and L2 penalties prevent overfitting
- Feature Importance:** Automatically identifies most predictive features
- Hyperparameter Tuning:** Critical for optimal performance

 Model Selection Guidelines
- Few samples, many features:** Use SVM
- Need interpretability:** Use Decision Trees
- Want highest accuracy:** Use XGBoost
- Need real-time predictions:** Use SVM with optimized preprocessing

---

 🧩 Future Enhancements

 Completed ✅
-  Linear Regression implementation
-  Decision Tree classifier
-  XGBoost model with tuning
-  Linear SVM for gender classification
-  Real-time computer vision integration

 In Progress 🔄
-  Random Forest ensemble methods
-  Neural Network implementations
-  Unified dashboard for model comparison
-  Advanced face recognition with deep learning

 Planned 🔮
-  Multi-class SVM (emotion recognition with 7+ emotions)
-  Convolutional Neural Networks (CNNs) for image classification
-  Natural Language Processing (NLP) models
-  Time series forecasting models
-  Model deployment with Flask/FastAPI
-  Docker containerization
-  CI/CD pipeline for automated testing

---

## 📖 Documentation

Each model includes:
- **Detailed code comments** explaining logic and methodology
- **Markdown documentation** within Jupyter notebooks
- **Visualization** of results and performance metrics
- **Usage examples** and best practices

### SVM Models Additional Resources
- `svm_testing_suite.py` - Test model generalization on new people
- Visualization tools for decision boundaries and support vectors
- Performance analysis scripts

---

🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch(`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Contribution Ideas
- Add new ML algorithms (Random Forest, Neural Networks, etc.)
- Improve existing models with better preprocessing
- Add more datasets and problem domains
- Enhance documentation and tutorials
- Fix bugs or optimize code performance

---

## 👨‍💻 Contributors

| Name | GitHub Handle | Contributions |
|------|----------------|---------------|
| Joshua Miguel Jamisola | [@SoftwareReboot](https://github.com/SoftwareReboot) | Project lead, SVM models, XGBoost implementation |
| McLovin | [@jamardines](https://github.com/jamardines) | Decision Trees, data preprocessing, visualization |

---

📝 Citation

If you use this code in your research or projects, please cite:

```bibtex
@misc{ml_model_collection_2025,
  author = {Jamisola, Joshua Miguel and McLovin},
  title = {MachineLearning-Model: Educational ML Project Collection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/SoftwareReboot/MachineLearning-Model}
}
```

---

 📄 License

This project is open-source and available under the MIT License for educational and research purposes.  
Feel free to fork, learn, and contribute!

See [LICENSE](LICENSE) file for details.

---

 📧 Contact

For questions, suggestions, or collaboration:
- **Issues:** [GitHub Issues](https://github.com/SoftwareReboot/MachineLearning-Model/issues)
- **Discussions:** [GitHub Discussions](https://github.com/SoftwareReboot/MachineLearning-Model/discussions)

---

 🌟 Acknowledgments

- Our university professors for guidance and mentorship
- The open-source ML community for tools and libraries
- Scikit-learn, XGBoost, and OpenCV teams for excellent documentation
- Stack Overflow and ML forums for troubleshooting support

---

 ⭐ Star This Repository!

If you found this project helpful or educational:
- ⭐ **Star** the repository
- 🔀 **Fork** it for your own projects
- 📢 **Share** with classmates and the ML community

Happy Learning! 🚀🤖

---

