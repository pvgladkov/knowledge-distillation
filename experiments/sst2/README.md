# Sentiment Analysis on SST-2 Binary classification


Implementation of ideas from "**Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**" (2019) https://arxiv.org/abs/1903.12136 

The step-by-step tutorial for this approach is [here](https://towardsdatascience.com/simple-tutorial-for-distilling-bert-99883894e90a).

Dataset is **SST-2** from https://gluebenchmark.com/tasks.

## Run 



Train baseline BiLSTM

```bash
$ cd /app
$ python3 experiments/sst2/train_baseline.py
```

Train BERT

```bash
$ cd /app
$ python3 experiments/sst2/train_bert.py
```

Train distilled BiLSTM

```bash
$ cd /app
$ python3 experiments/sst2/distil_bert.py
```


## Results

### BiLSTM

![lstm results](https://github.com/pvgladkov/knowledge-distillation/blob/master/experiments/sst2/images/lstm.png)


### Distilled version

![distilled_lstm results](https://github.com/pvgladkov/knowledge-distillation/blob/master/experiments/sst2/images/distil_lstm.png)



