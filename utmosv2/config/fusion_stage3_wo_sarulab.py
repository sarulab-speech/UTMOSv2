from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from torchvision import transforms

from utmosv2.transform import XYMasking

batch_size = 8
num_folds = 5

sr = 16000

preprocess = SimpleNamespace(
    top_db=30, min_seconds=None, save_path=Path("preprocessed_data")
)

split = SimpleNamespace(
    type="sgkf_kind",
    target="mos",
    group="sys_id",
    kind="dataset",
)

external_data: list[str] | str = [
    # "sarulab",
    "blizzard2008",
    "blizzard2009",
    "blizzard2011",
    "blizzard2010-EH1",
    "blizzard2010-EH2",
    "blizzard2010-ES1",
    "blizzard2010-ES3",
    "somos",
]
use_bvcc = True


validation_dataset = "each"

dataset = SimpleNamespace(
    name="ssl_multispec_ext",
    specs=[
        SimpleNamespace(
            mode="melspec",
            n_fft=4096,
            hop_length=32,
            win_length=4096,
            n_mels=512,
            shape=(512, 512),
            norm=80,
        ),
        SimpleNamespace(
            mode="melspec",
            n_fft=4096,
            hop_length=32,
            win_length=2048,
            n_mels=512,
            shape=(512, 512),
            norm=80,
        ),
        SimpleNamespace(
            mode="melspec",
            n_fft=4096,
            hop_length=32,
            win_length=1024,
            n_mels=512,
            shape=(512, 512),
            norm=80,
        ),
        SimpleNamespace(
            mode="melspec",
            n_fft=4096,
            hop_length=32,
            win_length=512,
            n_mels=512,
            shape=(512, 512),
            norm=80,
        ),
    ],
    spec_frames=SimpleNamespace(
        num_frames=2, frame_sec=1.4, mixup_inner=True, mixup_alpha=0.4, extend="tile"
    ),
    ssl=SimpleNamespace(
        duration=3,
    ),
)
transform = dict(
    train=transforms.Compose(
        [
            transforms.Resize((512, 512)),
            XYMasking(
                num_masks_x=(0, 2),
                num_masks_y=(0, 2),
                mask_x_length=(10, 40),
                mask_y_length=(10, 30),
                fill_value=0,
                p=0.5,
            ),
            # transforms.ToTensor(),
        ]
    ),
    valid=transforms.Compose(
        [
            transforms.Resize((512, 512)),
            # transforms.ToTensor()
        ]
    ),
)

loss = [
    (SimpleNamespace(name="pairwize_diff", margin=0.2, norm="l1"), 0.7),
    (SimpleNamespace(name="mse"), 0.2),
]

optimizer = SimpleNamespace(name="adamw", lr=5e-5, weight_decay=1e-4)

scheduler = SimpleNamespace(name="cosine", T_max=None, eta_min=1e-8)

model = SimpleNamespace(
    name="ssl_multispec_ext_v2",
    multi_spec=SimpleNamespace(
        backbone="tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=True,
        num_classes=1,
        pool_type="catavgmax",
        # feature_height=16,
        atten=True,
        # classifier=None,
    ),
    ssl=SimpleNamespace(
        name="facebook/wav2vec2-base",
        attn=1,
        freeze=False,
        num_classes=1,
    ),
    ssl_spec=SimpleNamespace(
        ssl_weight="ssl_only_stage2_wo_sarulab",
        spec_weight="spec_only_wo_sarulab",
        num_classes=1,
        freeze=False,
    ),
)

run = SimpleNamespace(
    mixup=True,
    mixup_alpha=0.4,
    num_epochs=2,
)

main_metric = "sys_srcc"
id_name = None


inference = SimpleNamespace(
    save_path=Path("preds"),
    submit_save_path=Path("submissions"),
    num_tta=5,
    batch_size=8,
    extend="tile",
)
