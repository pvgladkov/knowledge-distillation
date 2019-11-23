# Sentiment Analysis on SST-2 Binary classification


Implementation ideas from "**Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**" (2019) https://arxiv.org/abs/1903.12136 

## Run 

Prepare data

```bash
$ PYTHONPATH=/app python3 experiments/sst2/prepare_data.py
```

Train baseline BiLSTM

```bash
$ PYTHONPATH=/app python3 experiments/sst2/train_baseline.py
```

Train BERT

```bash
$ PYTHONPATH=/app python3 experiments/sst2/train_bert.py
```

Train distilled BiLSTM

```bash
$ PYTHONPATH=/app python3 experiments/sst2/distil_bert.py
```


## Results

| |accuracy|# of trainable params|
|---|---|---|
|BERT|0.92|178M|
|BiLSTM|0.75|0.84M|
|Distilled BiLSTM|0.76|0.84M|

