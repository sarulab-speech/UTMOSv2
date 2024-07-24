<h1 align="center">
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/training.md">
    <img width="94%" height="14px" src="image/titleLine2t.svg">
  </a>
  <div>📚 Guide to Training 📚<div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/training.md">
    <img width="94%" height="6px" src="image/titleLine2b.svg">
  </a>
</h1>

To train UTMOSv2 following the methods described in the paper or used in the competition, please refer to [this document](reproduction.md).

<h2 align="center">
  <div>🚀 Train UTMOSv2 Using Your Own Data 🚀</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/training.md#---train-utmosv2-using-your-own-data---------">
    <img width="80%" height="6px" src="image/line2.svg">
  </a>
</h2>

To train UTMOSv2 using your own data, you need to create a JSON file that contains the location and name of your data. Here is an example structure for the JSON file:

```json
{
  "data": [
    {
      "name": "dataset1",
      "dir": "/path/to/your/dataset1",
      "mos_list": "/path/to/your/moslist1.txt"
    },
    {
      "name": "dataset2",
      "dir": "/path/to/your/dataset2",
      "mos_list": "/path/to/your/moslist2.txt"
    }
    // Add more data entries as needed
  ]
}
```

Here, `name` is used to identify the data-domain ID, and `dir` specifies the directory where the corresponding `.wav` files are located. Additionally, mos_list records the MOS values for the .wav files in the directory, in the following format:

```text
sys64e2f-utt491a78a,2.375
sys64e2f-utt8485f83,3.625
sys7ab3c-utt1417b69,4.0
...
```

The file extension `.wav` is optional and can be included or omitted. The common files between those in the dir and those specified in the mos_list will be used.

Specify the name, dir, and mos_list set for each dataset-domain ID you want to train. 

Save this JSON file with an appropriate name, for example, `data_config.json` and run the following command:

```bash
python train.py --config spec_only --data_config data_config.json
```

<h2 align="center">
  <div>🧪 Fine-tuning from Pre-trained Weights 🧪</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/training.md#---fine-tuning-from-pre-trained-weights---------">
    <img width="80%" height="6px" src="image/line2.svg">
  </a>
</h2>

To continue training from existing weights, specify the `--weight` option and train as follows. This is useful when you want to perform additional training using weights learned in a previous stage or when fine-tuning.

```bash
python train.py --config spec_only --data_config data_config.json --weight /path/to/your/weights.pth
```

The `--weight` option can specify either the configuration file name or the path to the weight `.pth` file. If the configuration file name is specified, `models/{config_name}/fold{now_fold}_s{seed}_best_model.pth` is used.

<h2 align="center">
  <div>🔬 Using Weights & Biases (wandb) for Experiment Tracking 🔬</div>
  <a href="https://github.com/sarulab-speech/UTMOSv2/blob/main/docs/training.md#---using-weights--biases-wandb-for-experiment-tracking---------">
    <img width="80%" height="6px" src="image/line2.svg">
  </a>
</h2>

To use Weights & Biases (wandb) for experiment tracking, specify the `--wandb` option. You will also need to set the `WANDB_API_KEY` in your `.env` file or environment variables.

```bash
python train.py --config spec_only --data_config data_config.json --wandb
```