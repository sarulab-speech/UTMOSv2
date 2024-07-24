<h1 align="center">
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md">
    <img width="94%" height="14px" src="image/titleLine3t.svg">
  </a>
  <div>📘 Guide to Inference 📘<div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md">
    <img width="94%" height="6px" src="image/titleLine3b.svg">
  </a>
</h1>

Please refer to [this section](https://github.com/sarulab-speech/UTMOSv2?tab=readme-ov-file#---quick-prediction--------) for basic inference methods.

<h2 align="center">
  <div>📌 Data-domain ID for the MOS Prediction 📌</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md#---data-domain-id-for-the-mos-prediction---------">
    <img width="80%" height="6px" src="image/line3.svg">
  </a>
</h2>

By default, the data-domain ID for the MOS prediction is set to sarulab-data. To specify this and make predictions, you can specify the `--predict_dataset` flag with the following options:

- `sarulab` (default)
- `bvcc`
- `blizzard2008`, `blizzard2009`, `blizzard2010-EH1`, `blizzard2010-EH2`, `blizzard2010-ES1`, `blizzard2010-ES3`, `blizzard2011`
- `somos`

For example, to make predictions with the data-domain ID set to somos, use the following command:

```bash
python inference.py --input_dir /path/to/wav/dir/ --out_path /path/to/output/file.csv --predict_dataset somos
```

<h2 align="center">
  <div>✂️ Predicting Only a Subset of Files ✂️</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md#--%EF%B8%8F-predicting-only-a-subset-of-files-%EF%B8%8F--------">
    <img width="80%" height="6px" src="image/line3.svg">
  </a>
</h2>

By default, all `.wav` files in the `--input_dir` are used for prediction. To specify only a subset of these files, use the `--val_list_path` flag:

```bash
python inference.py --input_dir /path/to/wav/dir/ --out_path /path/to/output/file.csv --val_list_path /path/to/your/val/list.txt
```

The list of `.wav` files specified here should contain utt-id separated by new lines, as shown below. The file extension `.wav` is optional and can be included or omitted.

```text
sys00691-utt0682e32
sys00691-utt31fd854
sys00691-utt33a4826
...
```

<h2 align="center">
  <div>📈 Specify the Fold and the Number of Repetitions for More Accurate Predictions 📈</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md#---specify-the-fold-and-the-number-of-repetitions-for-more-accurate-predictions---------">
    <img width="80%" height="6px" src="image/line3.svg">
  </a>
</h2>

In the paper, predictions are made repeatedly for five randomly selected frames of the input speech waveform for all five folds, and the average is used. To specify this for more accurate predictions, do the following:

```bash
python inference.py --input_dir /path/to/wav/dir/ --out_path /path/to/output/file.csv --fold -1 --num_repetitions 5
```

Here, the `--fold` option specifies the fold number to be used. If set to `-1`, all folds will be used. The `--num_repetitions` option specifies the number of repetitions.

<h2 align="center">
  <div>🎯 Specify a Configuration File 🎯</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md#---specify-a-configuration-file---------">
    <img width="80%" height="6px" src="image/line3.svg">
  </a>
</h2>

To specify a configuration file for predictions, do the following:

```bash
python inference.py --config configuration_file_name --input_dir /path/to/wav/dir/ --out_path /path/to/output/file.csv
```

By default, `fusion_stage3`, which is the entire model of UTMOSv2, is used.

<h2 align="center">
  <div>⚖️ Make Predictions Using Your Own Weights ⚖️</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/inference.md#--%EF%B8%8F-make-predictions-using-your-own-weights-%EF%B8%8F--------">
    <img width="80%" height="6px" src="image/line3.svg">
  </a>
</h2>

To make predictions using your own weights, specify the path to the weights with the `--weight` option:

```bash
python inference.py --input_dir /path/to/wav/dir/ --out_path /path/to/output/file.csv --weight /path/to/your/weight.pth
```

The `--weight` option can specify either the configuration file name or the path to the weight `.pth` file. By default, `models/{config_name}/fold{now_fold}_s{seed}_best_model.pth` is used.

The weights must be compatible with the model specified by `--config_name`.

> [!NOTE]
> In this case, the same weights specified will be used for all folds. To use different weights for each fold, you can do the following:
>
> ```bash
> for i in {0..5}; do
>     python inference.py --input_path /path/to/wav/file.wav --out_path /path/to/output/file.csv --weight /path/to/your/weight_fold${i}.pth --fold $i
> done
> ```
