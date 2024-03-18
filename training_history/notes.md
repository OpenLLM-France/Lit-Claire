# Training log details

## Dataset compositions

### 2023/10/25 -- v0.0 French

Training set composition: 19132 conversations, 1802093 turns, 50662108 words, 155788 samples (of length 2049):
```
* FR/Politics/TRAIN             :  142 convs ( 0.74 %) 1.3M words ( 2.47 %) 5.2k samples ( 3.32 %) -- weights =  5.44 %
* FR/Meetings/TRAIN             :  375 convs ( 1.96 %) 1.7M words ( 3.27 %) 6.9k samples ( 4.42 %) -- weights =  7.36 %
* FR/Theatre/TRAIN              : 6.4k convs (33.40 %)  16M words (31.25 %)  36k samples (23.16 %) -- weights = 17.71 %
* FR/Interviews/TRAIN           : 9.3k convs (48.59 %) 4.3M words ( 8.56 %)  38k samples (24.59 %) -- weights = 19.96 %
* FR/AssembleeNationale/TRAIN   : 1.6k convs ( 8.36 %)  22M words (43.71 %)  41k samples (26.45 %) -- weights = 24.26 %
* FR/Conversations/TRAIN        : 1.3k convs ( 6.95 %) 5.4M words (10.74 %)  28k samples (18.07 %) -- weights = 25.27 %
```

Validation set composition: 303 conversations, 24821 turns, 840649 words, 889 samples (of length 2049):
```
* FR/Theatre/TEST                :   16 convs ( 5.28 %)  48k words ( 5.75 %)   50 samples ( 5.62 %)
* FR/Politics/TEST               :   29 convs ( 9.57 %)  80k words ( 9.55 %)   81 samples ( 9.11 %)
* FR/Conversations/TEST          :   32 convs (10.56 %) 108k words (12.79 %)  105 samples (11.81 %)
* FR/AssembleeNationale/TEST     :    8 convs ( 2.64 %) 119k words (14.11 %)  110 samples (12.37 %)
* FR/Interviews/TEST             :  143 convs (47.19 %) 114k words (13.52 %)  205 samples (23.06 %)
* FR/Meetings/TEST               :   75 convs (24.75 %) 372k words (44.28 %)  338 samples (38.02 %)
```

### 2023/10/26-29 -- v0.1 French (CC-BY-NC-SA 4.0)

* Weight penalty for Assemblée Nationale: 16
* Weight penalty for Théatre: 4

Training set composition: 38219 conversations, 2933193 turns, 159568634 words, 373553 samples (of length 2049):
```
* FR/PresDiscourse/TRAIN        :   17 convs ( 0.04 %)  86k words ( 0.05 %)  479 samples ( 0.13 %) -- weights =  0.37 %
* FR/Assistance/TRAIN           :  542 convs ( 1.42 %) 159k words ( 0.10 %) 3.2k samples ( 0.87 %) -- weights =  0.75 %
* FR/Debates/TRAIN              :  119 convs ( 0.31 %) 402k words ( 0.25 %) 2.2k samples ( 0.60 %) -- weights =  1.72 %
* FR/Meetings/TRAIN             :  243 convs ( 0.64 %) 1.2M words ( 0.74 %) 6.2k samples ( 1.65 %) -- weights =  5.16 %
* FR/FreeConversations/TRAIN    :  753 convs ( 1.97 %) 2.2M words ( 1.39 %)  13k samples ( 3.58 %) -- weights =  9.91 %
* FR/Theatre/TRAIN              :  31k convs (81.09 %)  16M words (10.16 %)  69k samples (18.53 %) -- weights = 17.70 %
* FR/Interviews/TRAIN           : 1.1k convs ( 2.79 %) 6.4M words ( 4.03 %)  42k samples (11.36 %) -- weights = 29.02 %
* FR/AssembleeNationale_16/TRAIN:  381 convs ( 1.00 %)  11M words ( 6.79 %)  19k samples ( 5.14 %) -- weights =  2.89 %
* FR/AssembleeNationale_13/TRAIN: 1.2k convs ( 3.27 %)  36M words (22.26 %)  64k samples (17.04 %) -- weights =  9.45 %
* FR/AssembleeNationale_14/TRAIN: 1.3k convs ( 3.45 %)  39M words (24.34 %)  69k samples (18.47 %) -- weights = 10.33 %
* FR/AssembleeNationale_15/TRAIN: 1.5k convs ( 4.03 %)  48M words (29.88 %)  85k samples (22.62 %) -- weights = 12.68 %
```

Validation set composition: 343 conversations, 27365 turns, 834887 words, 893 samples (of length 2049):
```
* FR/Assistance/TEST            :    8 convs ( 2.33 %) 3.8k words ( 0.45 %)    8 samples ( 0.90 %)
* FR/PresDiscourse/TEST         :    1 convs ( 0.29 %) 7.5k words ( 0.90 %)    8 samples ( 0.90 %)
* FR/Interviews/TEST            :    7 convs ( 2.04 %)  74k words ( 8.92 %)   70 samples ( 7.84 %)
* FR/FreeConversations/TEST     :   39 convs (11.37 %)  67k words ( 7.98 %)   76 samples ( 8.51 %)
* FR/Debates/TEST               :   29 convs ( 8.45 %)  80k words ( 9.61 %)   81 samples ( 9.07 %)
* FR/AssembleeNationale/TEST    :    9 convs ( 2.62 %) 145k words (17.34 %)  128 samples (14.33 %)
* FR/Theatre/TEST               :  175 convs (51.02 %)  85k words (10.21 %)  185 samples (20.72 %)
* FR/Meetings/TEST              :   75 convs (21.87 %) 372k words (44.58 %)  337 samples (37.74 %)
```

### 2023/11/07 -- v0.1 French (Apache)

Same validation set as for the "CC-BY-NC-SA version"

#### with TheatreClassique

* Weight penalty for Assemblée Nationale: 24
* Weight penalty for Théatre: 6

Training composition: 35816 conversations, 2291620 turns, 150475208 words, 313309 samples (of length 2049):
```
* FR/FreeConversations/TRAIN    :    4 convs ( 0.01 %)  26k words ( 0.02 %)  159 samples ( 0.05 %) -- weights =  0.29 %
* FR/PresDiscourse/TRAIN        :    3 convs ( 0.01 %)  31k words ( 0.02 %)  173 samples ( 0.06 %) -- weights =  0.33 %
* FR/Debates/TRAIN              :  115 convs ( 0.32 %) 326k words ( 0.22 %) 1.9k samples ( 0.59 %) -- weights =  3.37 %
* FR/Meetings/TRAIN             :  216 convs ( 0.60 %) 1.0M words ( 0.67 %) 5.5k samples ( 1.76 %) -- weights = 10.49 %
* FR/Theatre/TRAIN              :  31k convs (86.53 %)  16M words (10.77 %)  69k samples (22.10 %) -- weights = 28.53 %
* FR/AssembleeNationale_16/TRAIN:  381 convs ( 1.06 %)  11M words ( 7.20 %)  19k samples ( 6.13 %) -- weights =  4.66 %
* FR/AssembleeNationale_13/TRAIN: 1.2k convs ( 3.48 %)  36M words (23.60 %)  64k samples (20.31 %) -- weights = 15.24 %
* FR/AssembleeNationale_14/TRAIN: 1.3k convs ( 3.69 %)  39M words (25.81 %)  69k samples (22.02 %) -- weights = 16.65 %
* FR/AssembleeNationale_15/TRAIN: 1.5k convs ( 4.30 %)  48M words (31.69 %)  85k samples (26.97 %) -- weights = 20.44 %
```

#### without TheatreClassique

* Weight penalty for Assemblée Nationale: 40
* Weight penalty for Théatre: 2

Dataset composition: 8906 conversations, 1816190 turns, 136960226 words, 254914 samples (of length 2049):
```
* FR/FreeConversations/TRAIN    :    4 convs ( 0.04 %)  26k words ( 0.02 %)  159 samples ( 0.06 %) -- weights =  0.45 %
* FR/PresDiscourse/TRAIN        :    3 convs ( 0.03 %)  31k words ( 0.02 %)  173 samples ( 0.07 %) -- weights =  0.52 %
* FR/Debates/TRAIN              :  115 convs ( 1.29 %) 326k words ( 0.24 %) 1.9k samples ( 0.73 %) -- weights =  5.34 %
* FR/Meetings/TRAIN             :  216 convs ( 2.43 %) 1.0M words ( 0.73 %) 5.5k samples ( 2.17 %) -- weights = 16.61 %
* FR/Theatre/TRAIN              : 4.1k convs (45.81 %) 2.7M words ( 1.96 %)  11k samples ( 4.25 %) -- weights = 22.92 %
* FR/AssembleeNationale_16/TRAIN:  381 convs ( 4.28 %)  11M words ( 7.92 %)  19k samples ( 7.54 %) -- weights =  4.43 %
* FR/AssembleeNationale_13/TRAIN: 1.2k convs (14.01 %)  36M words (25.93 %)  64k samples (24.97 %) -- weights = 14.48 %                        
* FR/AssembleeNationale_14/TRAIN: 1.3k convs (14.82 %)  39M words (28.36 %)  69k samples (27.06 %) -- weights = 15.82 %
* FR/AssembleeNationale_15/TRAIN: 1.5k convs (17.28 %)  48M words (34.82 %)  85k samples (33.15 %) -- weights = 19.42 %
```

### 2024/03/15 -- v0.0 Bilingual

First run with the bilingual dataset.

Training set composition: 980772 conversations, 23359882 turns, 1005427321 words, 5866757 samples (of length 2049), 489304 batches (of 12):
```
* EN/Assistance/TRAIN           : 468k convs (47.67 %)  53M words ( 5.26 %) 935k samples (15.94 %) -- weights =  6.91 %
* EN/FreeChat/TRAIN             :  17k convs ( 1.75 %) 3.6M words ( 0.36 %)  36k samples ( 0.61 %) -- weights =  0.45 %
* EN/Interviews_1/TRAIN         :  50k convs ( 5.12 %)  59M words ( 5.90 %) 412k samples ( 7.01 %) -- weights =  1.78 %
* EN/Interviews_2/TRAIN         :  50k convs ( 5.13 %)  85M words ( 8.49 %) 496k samples ( 8.46 %) -- weights =  2.55 %
* EN/Interviews_3/TRAIN         :  51k convs ( 5.15 %) 104M words (10.39 %) 561k samples ( 9.57 %) -- weights =  3.12 %
* EN/Interviews_4/TRAIN         :  50k convs ( 5.13 %)  80M words ( 7.98 %) 460k samples ( 7.84 %) -- weights =  2.40 %
* EN/Interviews_5/TRAIN         :  50k convs ( 5.15 %)  77M words ( 7.66 %) 446k samples ( 7.60 %) -- weights =  2.30 %
* EN/Interviews_6/TRAIN         :  51k convs ( 5.16 %)  86M words ( 8.57 %) 478k samples ( 8.15 %) -- weights =  2.57 %
* EN/Interviews_7/TRAIN         :  51k convs ( 5.16 %)  94M words ( 9.34 %) 503k samples ( 8.57 %) -- weights =  2.80 %
* EN/Interviews_8/TRAIN         :  50k convs ( 5.13 %)  88M words ( 8.71 %) 477k samples ( 8.13 %) -- weights =  2.61 %
* EN/Interviews/TRAIN           :  45k convs ( 4.60 %)  44M words ( 4.36 %) 361k samples ( 6.16 %) -- weights =  1.32 %
* EN/Meetings/TRAIN             :  188 convs ( 0.02 %) 1.3M words ( 0.13 %) 7.3k samples ( 0.12 %) -- weights =  0.69 %
* EN/Misc/TRAIN                 :  886 convs ( 0.09 %)  10M words ( 1.02 %)  53k samples ( 0.90 %) -- weights =  5.12 %
* EN/ParliamentaryProceedings/TRAIN: 5.1k convs ( 0.52 %)  56M words ( 5.53 %) 237k samples ( 4.04 %) -- weights = 13.10 %
* EN/SpokenDialogue/TRAIN       : 3.3k convs ( 0.33 %) 4.4M words ( 0.44 %)  32k samples ( 0.54 %) -- weights =  2.20 %
* FR/Assistance/TRAIN           :  542 convs ( 0.06 %) 159k words ( 0.02 %) 3.2k samples ( 0.06 %) -- weights =  0.37 %
* FR/Debates/TRAIN              :  119 convs ( 0.01 %) 402k words ( 0.04 %) 2.2k samples ( 0.04 %) -- weights =  0.86 %
* FR/FreeConversations/TRAIN    :  753 convs ( 0.08 %) 2.2M words ( 0.22 %)  13k samples ( 0.23 %) -- weights =  4.97 %
* FR/Interviews/TRAIN           : 1.1k convs ( 0.11 %) 6.4M words ( 0.64 %)  42k samples ( 0.72 %) -- weights = 14.54 %
* FR/Meetings/TRAIN             :  243 convs ( 0.02 %) 1.2M words ( 0.12 %) 6.2k samples ( 0.11 %) -- weights =  2.59 %
* FR/ParliamentaryProceedings_1/TRAIN: 1.2k convs ( 0.13 %)  36M words ( 3.53 %)  64k samples ( 1.08 %) -- weights =  4.74 %
* FR/ParliamentaryProceedings_2/TRAIN: 1.3k convs ( 0.13 %)  39M words ( 3.86 %)  69k samples ( 1.18 %) -- weights =  5.18 %
* FR/ParliamentaryProceedings_3/TRAIN: 1.5k convs ( 0.16 %)  48M words ( 4.74 %)  85k samples ( 1.44 %) -- weights =  6.35 %
* FR/ParliamentaryProceedings_4/TRAIN:  381 convs ( 0.04 %)  11M words ( 1.08 %)  19k samples ( 0.33 %) -- weights =  1.45 %
* FR/PresDiscourse/TRAIN        :   17 convs ( 0.00 %)  86k words ( 0.01 %)  479 samples ( 0.01 %) -- weights =  0.18 %
* FR/Theatre/TRAIN              :  31k convs ( 3.16 %)  16M words ( 1.61 %)  69k samples ( 1.18 %) -- weights =  8.87 %
```

Validation set composition: 12335 conversations, 386892 turns, 17575218 words, 18962 samples (of length 2049), 1593 batches (of 12):
```
* EN/Assistance/TEST            : 1.8k convs (14.73 %) 186k words ( 1.06 %) 1.8k samples ( 9.58 %)
* EN/Interviews_1/TEST          : 1.1k convs ( 9.09 %) 1.3M words ( 7.50 %) 1.5k samples ( 8.12 %)
* EN/Interviews_2/TEST          : 1.1k convs ( 9.04 %) 1.9M words (10.60 %) 1.8k samples ( 9.72 %)
* EN/Interviews_3/TEST          : 1.1k convs ( 9.13 %) 2.4M words (13.71 %) 2.2k samples (11.54 %)
* EN/Interviews_4/TEST          : 1.1k convs ( 8.98 %) 1.8M words (10.43 %) 1.8k samples ( 9.27 %)
* EN/Interviews_5/TEST          : 1.1k convs ( 9.17 %) 1.7M words ( 9.53 %) 1.7k samples ( 8.78 %)
* EN/Interviews_6/TEST          : 1.1k convs ( 8.77 %) 1.8M words (10.46 %) 1.7k samples ( 9.14 %)
* EN/Interviews_7/TEST          : 1.1k convs ( 8.70 %) 2.0M words (11.46 %) 1.8k samples ( 9.75 %)
* EN/Interviews_8/TEST          : 1.1k convs ( 9.26 %) 2.0M words (11.40 %) 1.8k samples ( 9.69 %)
* EN/Interviews/TEST            :  981 convs ( 7.95 %) 966k words ( 5.49 %) 1.2k samples ( 6.25 %)
* EN/Meetings/TEST              :   26 convs ( 0.21 %) 170k words ( 0.97 %)  164 samples ( 0.86 %)
* EN/Misc/TEST                  :   22 convs ( 0.18 %) 142k words ( 0.81 %)  120 samples ( 0.63 %)
* EN/SpokenDialogue/TEST        :  248 convs ( 2.01 %) 323k words ( 1.84 %)  371 samples ( 1.96 %)
* FR/Assistance/TEST            :    8 convs ( 0.06 %) 3.8k words ( 0.02 %)    8 samples ( 0.04 %)
* FR/Debates/TEST               :   29 convs ( 0.24 %)  80k words ( 0.46 %)   81 samples ( 0.43 %)
* FR/FreeConversations/TEST     :   39 convs ( 0.32 %)  67k words ( 0.38 %)   76 samples ( 0.40 %)
* FR/Interviews/TEST            :    7 convs ( 0.06 %)  74k words ( 0.42 %)   70 samples ( 0.37 %)
* FR/Meetings/TEST              :   75 convs ( 0.61 %) 372k words ( 2.12 %)  337 samples ( 1.78 %)
* FR/ParliamentaryProceedings/TEST:    9 convs ( 0.07 %) 145k words ( 0.82 %)  128 samples ( 0.68 %)
* FR/PresDiscourse/TEST         :    1 convs ( 0.01 %) 7.5k words ( 0.04 %)    8 samples ( 0.04 %)
* FR/Theatre/TEST               :  175 convs ( 1.42 %)  85k words ( 0.49 %)  185 samples ( 0.98 %)
```

## Hyper-parameters

### Text data augmentation

* Anonymization / Desanonymization
* Remove punctuations & lower case
* Use dashes (instead of "[Intervenant X]") for dialogs with 2 people / nothing for monologs

### mono GPU, no FSDP

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
  "early_stopping": 2,
  "interval_unit": "time",
  "save_interval": 3540,
  "eval_interval": 3540,
  "max_checkpoints": 20
}
```

### mono GPU, with FSDP

~50H training

```json
{
  "devices": 1,
  "precision": "bf16-true",
  "batch_size": 132,
  "micro_batch_size": 12,
  "...": "...",
  "lora_r": 16,
  "lora_alpha": 32,
  "...": "..."
}
```

### multi GPU

~6H30 training

```json
{
  "devices": 8,
  "precision": "bf16-true",
  "batch_size": 16,
  "micro_batch_size": 8,
  "...": "...",
  "save_interval": 1800,
  "eval_interval": 1800,
  "max_checkpoints": 13,
}
```

