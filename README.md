# House Prices ML (starter)

**One-line goal:** End-to-end ML pipeline for predicting house prices (Ames dataset).

**Dataset:** Kaggle - House Prices - Advanced Regression Techniques (link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

**Metric:** RMSE

**Success criteria:** Achieve >=15% improvement over baseline linear regression RMSE.

## Repo structure
- notebooks/: Preprocessing the data and training the model
- src/: reusable modules (data.py, features.py, train.py, predict.py)
- app.py: Streamlit demo
- results/: final model and artifacts

## Model
-The trained model is saved as results/final_model.pkl. It is not uploaded to GitHub because of size and reproducibility reasons. You can regenerate it by running the notebook end-to-end.

## How to run locally
1. Create venv and install requirements
2. `jupyter lab`
3. `streamlit run app.py`

## Notes about preprocessing & output
- The notebook preprocesses numeric features with `np.log1p` and also transforms the **target** `SalePrice` with `np.log1p` before training.
- The saved model (`results/final_model.pkl`) therefore predicts `log1p(SalePrice)`. Inference code and the demo app invert predictions using `np.expm1` and return raw sale prices.
- To reproduce the model: run `notebooks/final_analysis.ipynb` end-to-end (it produces results/final_model.pkl and results/final_model_metadata.json).


## License

- The dataset is licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).

## Author
- Khalil Fawaz — link to LinkedIn: www.linkedin.com/in/khalil-fawaz-aa7709314
