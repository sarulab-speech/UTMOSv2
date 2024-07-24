from pathlib import Path
from types import SimpleNamespace

batch_size = 32
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

dataset = SimpleNamespace(
    name="sslext",
    ssl=SimpleNamespace(
        duration=3,
    ),
)

external_data = "all"
use_bvcc = True


validation_dataset = "each"

loss = [
    (SimpleNamespace(name="pairwize_diff", margin=0.2, norm="l1"), 0.7),
    (SimpleNamespace(name="mse"), 0.2),
]

optimizer = SimpleNamespace(name="adamw", lr=1e-3, weight_decay=1e-4)

scheduler = SimpleNamespace(name="cosine", T_max=None, eta_min=1e-7)

model_path = "model"
model = SimpleNamespace(
    name="sslext",
    ssl=SimpleNamespace(
        name="facebook/wav2vec2-base",
        attn=1,
        freeze=True,
        num_classes=1,
    ),
)

run = SimpleNamespace(
    mixup=True,
    mixup_alpha=0.4,
    num_epochs=20,
)

main_metric = "sys_srcc"
id_name = None


inference = SimpleNamespace(
    save_path=Path("preds"),
    submit_save_path=Path("submissions"),
    num_tta=5,
    batch_size=8,
    # extend="tile",
)
