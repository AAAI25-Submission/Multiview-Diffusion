# Label Efficient and Personalized Arrhythmia Detection via Multiview Diffusion Models

```shell
├── Baselines
│  ├── CMSC
│  ├── EffNet
│  └── Models
├── Dataset
│  ├── data_ChapmanShaoxing_segments
│  └── data_LTAF_segments
├── Frequency
├── Lorenz
├── Multiview
├── requirements.txt
└── run.sh
```

The `Baselines` fold inscludes the pre-training, fine-tuning, and testing code of two baselines EfficientNet and CMSC.

The `Multiview` fold includes the fine-tuning and testing code of the proposed multivi diffusion-based mthod.

Pre-trained models are also provided, to conduct test with pre-trained model, use follow command:

```shell
bash ./run.sh <method> <task>
```

`<method>` can be selected in `eff`, `cmsc`, and `multiDiff`.

`<task>` can be selected in `generalization` and `personalization`.

