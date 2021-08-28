# ODSC approach

Download 100 batches of 1000 examples from pubtator
```bash
python src/main/download_data.py 100
```

Preprocess the data into a csv for each word from each fragment
```bash
python src/main/preprocess_data.py
```

Train a model
```bash
python src/main/train_model.py
```
