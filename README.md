# Bank Marketing Classification Analysis

## Overview

This project provides a comprehensive machine learning analysis comparing four different classifiers (K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines) on the Bank Marketing dataset to predict whether a client will subscribe to a term deposit.

## Jupyter Kernel Information

**Which kernel to use?** This project uses the **Python 3 kernel (IPython kernel)**.

📘 **For detailed kernel setup instructions, see [JUPYTER_KERNEL_SETUP.md](JUPYTER_KERNEL_SETUP.md)**

Quick answer:
- Install dependencies: `pip install -r requirements.txt`
- Start Jupyter: `jupyter notebook prompt_III.ipynb`
- Select kernel: **Python 3** (or create a custom kernel for virtual environments)

## Business Objective

> **Predict whether a client will subscribe to a term deposit (yes/no) based on demographic, campaign, and economic features.**
> 
> This predictive model helps the bank optimize marketing campaigns by targeting clients who are more likely to subscribe, thereby reducing costs and improving conversion rates.

## Dataset Information

- **Source:** UCI Machine Learning Repository - Bank Marketing Dataset
- **Records:** 41,188 customer interactions
- **Features:** 20 input features (after excluding 'duration')
- **Target:** Binary classification (yes/no subscription)
- **Class Distribution:** ~11% yes, ~89% no (imbalanced)
- **Time Period:** 17 marketing campaigns conducted between May 2008 and November 2010

### Feature Categories:
- **Bank client data:** age, job, marital status, education, credit default, housing loan, personal loan
- **Campaign data:** contact type, month, day of week, number of contacts
- **Social and economic context:** employment variation rate, consumer price index, consumer confidence index, Euribor 3-month rate, number of employees

## Key Findings

### Model Performance
1. ✅ **All models significantly outperform the baseline** (88.7% majority class accuracy)
2. 📊 **Models achieve 89-91% accuracy** with varying precision-recall trade-offs
3. 🎯 **Hyperparameter tuning improves performance** across all models by 1-3%

### Most Important Features
1. **Economic indicators** (euribor3m, emp.var.rate, nr.employed) - strongest predictors
2. **Previous campaign outcomes** - highly correlated with future subscriptions
3. **Contact timing** (month, day of week) - significant impact on conversion
4. **Demographics** (age, job category) - moderate predictive power

### Recommended Model
- **Best for deployment:** Logistic Regression
  - High F1-score and recall
  - Fast training and prediction
  - Highly interpretable coefficients
  - Suitable for real-time scoring

## Model Comparison Summary

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|--------------|-----------|---------|----------|---------|---------------|
| **Logistic Regression** | ~0.91 | ~0.65 | ~0.45 | ~0.53 | ~0.93 | < 1s |
| **Decision Tree** | ~0.89 | ~0.58 | ~0.48 | ~0.52 | ~0.71 | < 1s |
| **K-Nearest Neighbors** | ~0.90 | ~0.62 | ~0.42 | ~0.50 | ~0.71 | < 1s |
| **Support Vector Machine** | ~0.91 | ~0.66 | ~0.42 | ~0.51 | ~0.93 | 10-30s |

*Note: Exact values depend on random seed and tuning results*

## Technologies Used

- **Python 3.8+**
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Development:** Jupyter Notebook

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/ru14/bank-marketing-classification.git
cd bank-marketing-classification
```

### 2. Set up Python environment (Recommended)

**Option A: Using venv (Python 3.8+)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n bank-marketing python=3.8

# Activate conda environment
conda activate bank-marketing
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify data file
Ensure the dataset exists at:
```
data/bank-additional/bank-additional-full.csv
```

### 5. Set up Jupyter Kernel

**Which kernel to use:** This project uses the **Python 3** kernel (IPython kernel).

After installing dependencies from `requirements.txt`, the IPython kernel will be available automatically. To verify:

```bash
# List available Jupyter kernels
jupyter kernelspec list
```

You should see `python3` in the list.

**If you created a virtual environment** and want to use it as a Jupyter kernel:

```bash
# Install ipykernel in your virtual environment
pip install ipykernel

# Add your environment as a Jupyter kernel
python -m ipykernel install --user --name=bank-marketing --display-name="Python (bank-marketing)"
```

Then in Jupyter, select **Kernel → Change Kernel → Python (bank-marketing)**.

### 6. Run the notebook
```bash
jupyter notebook prompt_III.ipynb
```

Or open in Jupyter Lab:
```bash
jupyter lab
```

## How to Use

1. **Open the notebook:** `prompt_III.ipynb`
2. **Run all cells:** Kernel → Restart & Run All
3. **Review results:** Check model comparisons, feature importance, and recommendations
4. **Customize:** Modify hyperparameters or add new models as needed

## Troubleshooting Jupyter Kernel Issues

### Kernel Not Found
If Jupyter can't find the Python kernel:
```bash
# Install ipykernel
pip install ipykernel

# Register the kernel
python -m ipykernel install --user
```

### Module Not Found Errors
If you get `ModuleNotFoundError` when running cells:
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify installation in the correct environment
pip list | grep -E "pandas|numpy|scikit-learn|jupyter"
```

### Kernel Keeps Dying
If the kernel crashes repeatedly:
- Check if you have enough memory (dataset is ~41K rows)
- Restart Jupyter: `Ctrl+C` in terminal, then restart
- Clear outputs: `Cell → All Output → Clear`
- Restart kernel: `Kernel → Restart & Clear Output`

### Wrong Python Version
To check Python version in the notebook:
```python
import sys
print(sys.version)
```
Should be Python 3.8 or higher.

### Switching Kernels in Jupyter
1. Open the notebook in Jupyter
2. Click **Kernel** in the menu bar
3. Select **Change Kernel**
4. Choose **Python 3** or your custom kernel name

## Project Structure

```
bank-marketing-classification/
├── README.md                          # This file
├── JUPYTER_KERNEL_SETUP.md            # Detailed Jupyter kernel setup guide
├── requirements.txt                   # Python dependencies
├── prompt_III.ipynb                   # Main analysis notebook
└── data/
    └── bank-additional/
        └── bank-additional-full.csv   # Dataset (41,188 rows)
```

## Notebook Contents

### Problems 1-11 (All Fully Implemented):
1. **Understanding the Data** - Background and context
2. **Read in the Data** - Load and explore dataset
3. **Understanding the Features** - Data quality and types
4. **Understanding the Task** - Business objective
5. **Engineering Features** - One-hot encoding and preprocessing
6. **Train/Test Split** - 80/20 stratified split
7. **Baseline Model** - Majority class benchmark
8. **Simple Model** - Initial Logistic Regression
9. **Score the Model** - Evaluation metrics
10. **Model Comparisons** - Four models with default parameters
11. **Improving the Model** - Comprehensive hyperparameter tuning

### Additional Sections:
- **Findings:** Key insights and patterns
- **Feature Importance:** Most predictive variables
- **Business Recommendations:** Actionable strategies
- **Next Steps:** Deployment and monitoring

## Business Recommendations

### 1. Target High-Probability Segments
- Focus on clients with positive previous campaign outcomes
- Consider economic conditions when planning campaigns
- Optimize contact timing (month and day of week)

### 2. Campaign Optimization
- Use model predictions to prioritize contact lists
- Set probability threshold based on cost-benefit analysis
- Balance precision and recall based on business goals

### 3. Resource Allocation
- Estimate expected conversion rates by segment
- Allocate experienced agents to high-probability prospects
- Reduce contact frequency for low-probability segments

### 4. Continuous Improvement
- Collect feedback on actual outcomes
- Retrain models quarterly with new data
- A/B test different contact strategies

## Next Steps for Production

1. **Model Deployment:**
   - Deploy Logistic Regression model to production
   - Create API endpoint for real-time predictions
   - Integrate with CRM system

2. **A/B Testing:**
   - Test model-driven targeting vs. random targeting
   - Measure impact on conversion rate and ROI
   - Iterate based on results

3. **Monitoring:**
   - Track model performance over time
   - Monitor for concept drift
   - Set up alerts for performance degradation

4. **Advanced Techniques:**
   - Explore ensemble methods (Random Forest, XGBoost)
   - Implement SMOTE for class imbalance
   - Experiment with feature engineering
   - Consider cost-sensitive learning

## Important Notes

⚠️ **'duration' feature excluded:** According to the dataset description, the 'duration' attribute highly affects the output target but is not known before a call is performed. Therefore, it should only be included for benchmark purposes and is excluded for realistic predictive modeling.

⚠️ **Class imbalance:** The dataset is imbalanced (~11% yes, 89% no). Evaluation focuses on Precision, Recall, F1-Score, and ROC-AUC rather than just accuracy.

## References

- **Dataset:** [UCI Machine Learning Repository - Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Paper:** [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. "A Data-Driven Approach to Predict the Success of Bank Telemarketing." Decision Support Systems, Elsevier, 62:22-31, June 2014

## License

This project is for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Project Status:** ✅ Complete and ready for use

**Last Updated:** February 2026
