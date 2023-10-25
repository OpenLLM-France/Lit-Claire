# Training log details

## Dataset compositions

### v0.0.1

Training set composition: 19132 conversations, 1802093 turns, 50662108 words, 155788 samples (of length 2049), 77992 batches (of 2):
```
* FR/Politics/TRAIN             :  142 convs ( 0.74 %) 1.3M words ( 2.47 %) 5.2k samples ( 3.32 %) 2.6k batches ( 3.32 %) -- weights =  5.44 %
* FR/Meetings/TRAIN             :  375 convs ( 1.96 %) 1.7M words ( 3.27 %) 6.9k samples ( 4.42 %) 3.4k batches ( 4.42 %) -- weights =  7.36 %
* FR/Theatre/TRAIN              : 6.4k convs (33.40 %)  16M words (31.25 %)  36k samples (23.16 %)  18k batches (23.17 %) -- weights = 17.71 %
* FR/Interviews/TRAIN           : 9.3k convs (48.59 %) 4.3M words ( 8.56 %)  38k samples (24.59 %)  19k batches (24.57 %) -- weights = 19.96 %
* FR/AssembleeNationale/TRAIN   : 1.6k convs ( 8.36 %)  22M words (43.71 %)  41k samples (26.45 %)  21k batches (26.46 %) -- weights = 24.26 %
* FR/Conversations/TRAIN        : 1.3k convs ( 6.95 %) 5.4M words (10.74 %)  28k samples (18.07 %)  14k batches (18.06 %) -- weights = 25.27 %
```

Validation set composition: 303 conversations, 24821 turns, 840649 words, 889 samples (of length 2049), 460 batches (of 2):
```
* FR/Theatre/TEST                :   16 convs ( 5.28 %)  48k words ( 5.75 %)   50 samples ( 5.62 %)   28 batches ( 6.09 %)
* FR/Politics/TEST               :   29 convs ( 9.57 %)  80k words ( 9.55 %)   81 samples ( 9.11 %)   44 batches ( 9.57 %)
* FR/Conversations/TEST          :   32 convs (10.56 %) 108k words (12.79 %)  105 samples (11.81 %)   56 batches (12.17 %)
* FR/AssembleeNationale/TEST     :    8 convs ( 2.64 %) 119k words (14.11 %)  110 samples (12.37 %)   56 batches (12.17 %)
* FR/Interviews/TEST             :  143 convs (47.19 %) 114k words (13.52 %)  205 samples (23.06 %)  104 batches (22.61 %)
* FR/Meetings/TEST               :   75 convs (24.75 %) 372k words (44.28 %)  338 samples (38.02 %)  172 batches (37.39 %)
```

### v0.0.2

Training set composition: 38219 conversations, 2933193 turns, 159568634 words, 373553 samples (of length 2049), 31181 batches (of 12):
```
* FR/PresDiscourse/TRAIN        :   17 convs ( 0.04 %)  86k words ( 0.05 %)  479 samples ( 0.13 %)   40 batches ( 0.13 %) -- weights =  0.37 %
* FR/Assistance/TRAIN           :  542 convs ( 1.42 %) 159k words ( 0.10 %) 3.2k samples ( 0.87 %)  270 batches ( 0.87 %) -- weights =  0.75 %
* FR/Debates/TRAIN              :  119 convs ( 0.31 %) 402k words ( 0.25 %) 2.2k samples ( 0.60 %)  186 batches ( 0.60 %) -- weights =  1.72 %
* FR/AssembleeNationale_16/TRAIN:  381 convs ( 1.00 %)  11M words ( 6.79 %)  19k samples ( 5.14 %) 1.6k batches ( 5.14 %) -- weights =  2.89 %
* FR/Meetings/TRAIN             :  243 convs ( 0.64 %) 1.2M words ( 0.74 %) 6.2k samples ( 1.65 %)  515 batches ( 1.65 %) -- weights =  5.16 %
* FR/AssembleeNationale_13/TRAIN: 1.2k convs ( 3.27 %)  36M words (22.26 %)  64k samples (17.04 %) 5.3k batches (17.04 %) -- weights =  9.45 %
* FR/FreeConversations/TRAIN    :  753 convs ( 1.97 %) 2.2M words ( 1.39 %)  13k samples ( 3.58 %) 1.1k batches ( 3.59 %) -- weights =  9.91 %
* FR/AssembleeNationale_14/TRAIN: 1.3k convs ( 3.45 %)  39M words (24.34 %)  69k samples (18.47 %) 5.8k batches (18.47 %) -- weights = 10.33 %
* FR/AssembleeNationale_15/TRAIN: 1.5k convs ( 4.03 %)  48M words (29.88 %)  85k samples (22.62 %) 7.1k batches (22.63 %) -- weights = 12.68 %
* FR/Theatre/TRAIN              :  31k convs (81.09 %)  16M words (10.16 %)  69k samples (18.53 %) 5.8k batches (18.54 %) -- weights = 17.70 %
* FR/Interviews/TRAIN           : 1.1k convs ( 2.79 %) 6.4M words ( 4.03 %)  42k samples (11.36 %) 3.5k batches (11.36 %) -- weights = 29.02 %
```

Validation set composition: 343 conversations, 27365 turns, 834887 words, 893 samples (of length 2049), 79 batches (of 12):
```
* FR/Assistance/TEST            :    8 convs ( 2.33 %) 3.8k words ( 0.45 %)    8 samples ( 0.90 %)    1 batch   ( 1.27 %)
* FR/PresDiscourse/TEST         :    1 convs ( 0.29 %) 7.5k words ( 0.90 %)    8 samples ( 0.90 %)    1 batch   ( 1.27 %)
* FR/Interviews/TEST            :    7 convs ( 2.04 %)  74k words ( 8.92 %)   70 samples ( 7.84 %)    6 batches ( 7.59 %)
* FR/FreeConversations/TEST     :   39 convs (11.37 %)  67k words ( 7.98 %)   76 samples ( 8.51 %)    7 batches ( 8.86 %)
* FR/Debates/TEST               :   29 convs ( 8.45 %)  80k words ( 9.61 %)   81 samples ( 9.07 %)    8 batches (10.13 %)
* FR/AssembleeNationale/TEST    :    9 convs ( 2.62 %) 145k words (17.34 %)  128 samples (14.33 %)   11 batches (13.92 %)
* FR/Theatre/TEST               :  175 convs (51.02 %)  85k words (10.21 %)  185 samples (20.72 %)   16 batches (20.25 %)
* FR/Meetings/TEST              :   75 convs (21.87 %) 372k words (44.58 %)  337 samples (37.74 %)   29 batches (36.71 %)
```

## Hyper-parameters

### v0.0.1

```json
{
  "devices": 1,
  "precision": "bf16-true",
  "batch_size": 4,
  "micro_batch_size": 2,
  "learning_rate": 0.0001,
  "warmup_steps": 0,
  "weight_decay": 0.01,
  "grad_clip": 1.0,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "lora_query": true,
  "lora_key": true,
  "lora_value": true,
  "lora_projection": true,
  "lora_mlp": true,
  "lora_head": true,
  "max_checkpoints": 20,
  "early_stopping": 2,
  "save_interval": 1800,
  "eval_interval": 1800,
  "interval_unit": "time"
}
```

* Text augmentation technics:
    * Anonymization / Desanonymization
    * Remove punctuations & lower case