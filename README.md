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

The `Baselines` fold includes the pre-training, fine-tuning, and testing code of two baselines EfficientNet and CMSC.

The `Multiview` fold includes the fine-tuning and testing code of the proposed multivi diffusion-based method.

You can install the required package using the command `pip install -r requirements.txt`.

Since the size of some files exceeds the limitation of Github, you need to install `git-lfs` to ensure that these files are downloaded correctly. 
You can use the following command to install `git-lfs` and download the repo.
```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
git lfs install
git clone https://github.com/AAAI25-Submission/Multiview-Diffusion.git
cd Multiview-Diffusion
git lfs pull
```

Pre-trained models are also provided, to conduct test with the pre-trained model, use following command:

```shell
bash ./run.sh <method> <task>
```

`<method>` can be selected in `eff`, `cmsc`, and `multiDiff`.

`<task>` can be selected in `generalization` and `personalization`.

