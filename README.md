# QNNFraudDetection
This repo contains the code and artifacts for a hybrid quantum classical experiment for credit card fraud detection.

The kaggle credit card fraud transactions dataset can be found here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Be sure to pip install the necessary packages: numpy, pandas, scikit-learn, pennylane, tqdm, torch, torchinfo

Then all you need to do is run the .py file in this repo. Use the CONFIG section to experiment with any of the components i.e. data splits, epochs, learning rate, qubits, q layers, and most notably, the number of priority pairs.
