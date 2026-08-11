# ProCFair: Proxy-guided Counterfactual Fairness under Missing Sensitive Attributes

ProCFair is an advanced framework for achieving counterfactual fairness in machine learning models, specifically designed to be robust under conditions where the **sensitive attribute (S) is partially missing (Missing Not At Random / MAR)**. 

While existing state-of-the-art models (like CFairMD, CLAIRE, and Adversarial CF) struggle or rely on basic imputation when sensitive attributes are unobserved, ProCFair introduces a novel **Proxy-guided Latent Sensitive Representation** combined with **Probability Marginalization**. This allows the model to accurately infer missing sensitive identities from available proxy variables and maintain strict counterfactual fairness without sacrificing prediction accuracy.

## 🚀 Key Features (Novelty)
1. **Proxy-guided Architecture**: Implements a dedicated neural network branch to infer the missing sensitive attribute ($S$) using the remaining non-sensitive covariates ($X$) as proxy variables.
2. **Probability Marginalization**: Uses an augmented Evidence Lower Bound (ELBO) and probability marginalization within the Counterfactual Importance-Weighted AutoEncoder (CIWAE) to seamlessly handle missingness during the generation of counterfactual representations.
3. **Multi-Dataset Generalization**: Extensively evaluated and dynamically adaptable to 3 different real-world tabular datasets:
   - *Online Retail II* (Target: High Spender, Sensitive: Country)
   - *Telco Customer Churn* (Target: Churn, Sensitive: Gender)
   - *Credit Score & Financial Clustering* (Target: Good Credit, Sensitive: Age > 30)
4. **Comprehensive Benchmarking Dashboard**: Includes a fully interactive **Streamlit Dashboard** that compares 4 major fairness paradigms in a single click:
   - **CFairMD** (Baseline CIWAE without fairness penalty on latent space):https://ieeexplore.ieee.org/document/11625989 (Counterfactual Fairness Prediction on Missing Data)
   - **CLAIRE** (KDD '23 - MMD-based latent matching) : https://dl.acm.org/doi/10.1145/3580305.3599408 (J. Ma, R. Guo, A. Zhang, and J. Li, “Learning for counterfactual fairness
from observational data,” in KDD, 2023, pp. 1620–1630)
   - **Adversarial CF** (Grari et al. - Adversarial Minimax Inference) : Adversarial learning for counterfactual fairness (https://link.springer.com/article/10.1007/s10994-022-06206-8)
   - **ProCFair** (Proposed SOTA Novelty)

## 📊 Evaluation Metrics
ProCFair is evaluated on strict, mathematically grounded fairness metrics:
- **Target Accuracy (↑)**: Classification accuracy of the factual predictions.
- **MMD (Maximum Mean Discrepancy) (↓)**: Distribution divergence between factual and counterfactual predictions in the latent space.
- **W1-dist (Wasserstein-1 Distance) (↓)**: Earth Mover's Distance quantifying the disparity of predictions across counterfactual sensitive groups.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ProCFair.git
   cd ProCFair
   ```

2. **Install the dependencies:**
   Make sure you have Python 3.8+ installed. Install the required packages via `pip`.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Core dependencies include `torch`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`)*

3. **Run the Dashboard:**
   Start the interactive Streamlit application locally:
   ```bash
   python3 -m streamlit run app.py
   ```

## 🖥️ Usage
- Open the provided `localhost:8501` link in your web browser.
- **Single Dataset Test**: Select a dataset from the dropdown, adjust the Missing Rate ($\epsilon$), and click "Generate & Train" to visualize the Counterfactual fairness metrics.
- **Full Scale Evaluation (TABLE I)**: Scroll to the right/bottom panel and click **"Generate Full Comparison Table (Run All)"** to execute a massive parallel training loop that evaluates all 4 algorithms across all 3 datasets, automatically generating the `•` / `◦` formatted performance matrix.
- **Advanced Plotting**: Click "Generate Advanced Plots" to recreate the analytical line charts, tradeoff scatter plots, and counterfactual density histograms.

---
*Built for fairness, robust against uncertainty.*
