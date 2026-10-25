# Notebook status

The former `UPI_Expense_Analytics_Generalized.ipynb` depended on the removed synthetic-data generator and no longer represents the current training pipeline. To avoid accidentally training or reporting results from that obsolete workflow, it has been removed from this package.

Use `backend/train_model.py` with `data/upi_transactions_2024.csv` for the current reproducible structured-model training process. See the project-root `README.md` for details and limitations.
