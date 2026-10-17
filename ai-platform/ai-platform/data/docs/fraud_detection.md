# Fraud Detection Model

The fraud model is a RandomForest classifier trained on synthetic transaction data with five raw features: amount, hour, tx_count_24h, merchant_risk and distance_km.

Feature engineering adds log_amount, an is_night flag for hours between 23:00 and 05:00, and amount_per_tx which divides the amount by the number of recent transactions.

Anomaly detection uses an IsolationForest trained only on legitimate transactions. Transactions with a negative decision score are flagged as anomalies even when the classifier probability is low.

Model evaluation reports ROC AUC, average precision, precision, recall and F1 on a stratified hold-out set, plus five-fold cross validation. The fraud decision threshold is 0.5 by default.
