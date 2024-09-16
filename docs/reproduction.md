<h1 align="center">
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/reproduction.md">
    <img width="94%" height="14px" src="image/titleLine4t.svg">
  </a>
  <div>🧬 How to Reproduce the Experiments 🧬<div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/reproduction.md">
    <img width="94%" height="6px" src="image/titleLine4b.svg">
  </a>
</h1>

To reproduce the model described in the paper and for the [VoiceMOS Challenge 2024](https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024) Track 1, follow the instructions below.
For details on training and inference options, please refer to the [training guide](training.md) and the [inference guide](inference.md).

We used a 40GB A100 GPU for training and inference.

<h2 align="left">
  <div>1. 📩 Install Training Dependencies</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/reproduction.md#---install-training-dependencies---------">
    <img width="85%" height="6px" src="image/line4.svg">
  </a>
</h2>

To install the dependencies required for training, run the following command:

```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .[train]
```

> [!NOTE]
> If you are using zsh, make sure to escape the square brackets like this:
> ```zsh
> pip install -e '.[train]'
> ```

<h2 align="left">
  <div>2. 📦 Prepare Training Data</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/reproduction.md#--1--prepare-training-data--------">
    <img width="85%" height="6px" src="image/line4.svg">
  </a>
</h2>

First, collect the BVCC datasets by following [the official instructions](https://www.codabench.org/competitions/2650/). It should be saved in this format in the project root:

```text
data
|-- DATA
|   `-- sets
|       `-- val_list.txt
|-- data-main.zip
|-- data-sets.zip
`-- main
    `-- DATA
        |-- mydata_system.csv
        |-- sets
        |   |-- DEVSET
        |   |-- test_mos_list.txt
        |   |-- TESTSET
        |   |-- train_mos_list.txt
        |   |-- TRAINSET
        |   `-- val_mos_list.txt
        `-- wav
            |-- sys00691-utt00e6ae6.wav
            |-- sys00691-utt04097bc.wav
            |-- sys00691-utt0682e32.wav
            |-- sys00691-utt12c197c.wav
             ...
```

Next, prepare the data other than BVCC. The locations of these data are as follows: [sarulab-data](https://github.com/sarulab-speech/VMC2024-sarulab-data), [Blizzard Challenges](https://www.cstr.ed.ac.uk/projects/blizzard/data.html), [SOMOS](https://innoetics.github.io/publications/somos-dataset/index.html). And then, organize them in the following format:

```text
data2
|-- blizzard2008
|   |-- blizzard2008_mos.csv
|   `-- blizzard2008_wavs
|       |-- A_arctic_news_2008_0002.wav
|       |-- A_arctic_news_2008_0002..wav
|        ...
|-- blizzard2009
|   |-- blizzard2009_mos.csv
|   `-- blizzard2009_wavs
|       |-- A_EH1_conv_2009_0003.wav
|       |-- A_EH1_conv_2009_0007.wav
|        ...
|-- blizzard2010
|   |-- blizzard2010_mos_EH1.csv
|   |-- blizzard2010_mos_EH2.csv
|   |-- blizzard2010_mos_ES1.csv
|   |-- blizzard2010_mos_ES3.csv
|   |-- blizzard2010_wavs_EH1
|   |   |-- A_EH1_news_2010_0010.wav
|   |   |-- A_EH1_news_2010_0013.wav
|   |    ...
|   |-- blizzard2010_wavs_EH2
|   |   |-- A_EH2_news_2010_0006.wav
|   |   |-- A_EH2_news_2010_0011.wav
|   |    ...
|   |-- blizzard2010_wavs_ES1
|   |   |-- A_ES1_news_2010_0028.wav
|   |   |-- A_ES1_news_2010_0034.wav
|   |    ...
|   `-- blizzard2010_wavs_ES3
|       |-- A_ES3_news_2010_0002.wav
|       |-- A_ES3_news_2010_0004.wav
|        ...
| 
|-- blizzard2011
|   |-- blizzard2011_mos.csv
|   `-- blizzard2011_wavs
|       |-- A_news_2011_0005.wav
|       |-- A_news_2011_0006.wav
|        ...
|-- somos
|    |-- audios
|    |   |-- booksent_2012_0005_001.wav
|    |   |-- booksent_2012_0005_023.wav
|    `-- training_files
|        `-- split1
|            `-- clean
|                |-- test_mos_list.txt
|                |-- TESTSET
|                |-- test_system.csv
|                |-- train_mos_list.txt
|                |-- TRAINSET
|                |-- train_system.csv
|                |-- valid_mos_list.txt
|                |-- VALIDSET
|                `-- valid_system.csv
`--sarulab
    `--VMC2024_MOS.csv
```

<h2 align="left">
  <div>3. 🔍 Reproducing the Experiments in the Paper and the Competition: Training</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/reproduction.md#--2--reproducing-the-experiments-in-the-paper-and-the-competition-training--------">
    <img width="85%" height="6px" src="image/line4.svg">
  </a>
</h2>

### a. Paper

The UTMOSv2 system presented in the paper can be trained as follows:

```bash
python train.py --reproduce --config spec_only # For w/o SSL
python train.py --reproduce --config ssl_only_stage1 # For stage 1 of w/o spec.
python train.py --reproduce --config ssl_only_stage2 --weight ssl_only_stage1 # For stage 2 of w/o spec.
python train.py --reproduce --config fusion_stage2 # For stage 2 of UTMOSv2
python train.py --reproduce --config fusion_stage3 --weight fusion_stage2 # For stage 3 of UTMOSv2
```

> [!NOTE]
> Stages must be trained in ascending numerical order, and the fusion model requires both the spec_only and ssl_only models to be trained beforehand.

The experiments for comparing multi-stage learning can be trained as follows:

```bash
python train.py --reproduce --config fusion_wo_stage2 # For w/o stage 2
python train.py --reproduce --config fusion_wo_stage1and2 # For w/o stage 1 & 2
```

The experiment for investigating the dataset can be trained as follows. For example, to train an experiment without BVCC datasets, use the following steps:

```bash
python train.py --reproduce --config spec_only_wo_bvcc
python train.py --reproduce --config ssl_only_stage1_wo_bvcc
python train.py --reproduce --config ssl_only_stage2_wo_bvcc --weight ssl_only_stage1_wo_bvcc
python train.py --reproduce --config fusion_stage2_wo_bvcc
python train.py --reproduce --config fusion_stage3_wo_bvcc --weight fusion_stage2_wo_bvcc
```

For the experiments w/o BC, w/o SOMOS, and w/o sarulab, you can train them by replacing `bvcc` with `bc`, `somos`, and `sarulab` respectively.

### b. Competition

Our system submitted to the competition can be trained as follows:

```bash
python train.py --reproduce --config c_spec_only_stage1
python train.py --reproduce --config c_spec_only_stage2 --weight c_spec_only_stage1
python train.py --reproduce --config c_ssl_only_stage1
python train.py --reproduce --config c_ssl_only_stage2 --weight ssl_only_stage1
python train.py --reproduce --config c_fusion_stage2
python train.py --reproduce --config c_fusion_stage3 --weight c_fusion_stage2
```

<h2 align="left">
  <div>4. 📊 Reproducing the Experiments in the Paper and the Competition: Inference</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/reproduction.md#--3--reproducing-the-experiments-in-the-paper-and-the-competition-inference--------">
    <img width="85%" height="6px" src="image/line4.svg">
  </a>
</h2>

Follow [the official instructions](https://www.codabench.org/competitions/2650/) to get the `eval_list.txt` and place it as follows:

```text
voicemos2024-track1-eval-phase/
`-- DATA
    `-- sets
        `-- eval_list.txt
```

To make predictions using the weights trained as described above, specify the desired configuration file　with `--config_name` and predict as follows:

```bash
python inference.py --reproduce --config fusion_stage3 --input_dir data/main/DATA --val_list_path voicemos2024-track1-eval-phase/DATA/sets/eval_list.txt --fold -1 --num_repetitions 5
```
