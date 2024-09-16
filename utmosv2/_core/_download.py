import subprocess

from utmosv2._core._constants import _UTMOSV2_CHACHE


def download_pretrained_weights_from_github(cfg_name: str) -> None:
    print(f"Downloading pretrained weights for `{cfg_name}`...")
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/sarulab-speech/UTMOSv2.git",
                _UTMOSV2_CHACHE.as_posix(),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "models"],
            cwd=_UTMOSV2_CHACHE,
            check=True,
        )
        subprocess.run(
            ["git", "checkout"],
            cwd=_UTMOSV2_CHACHE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to download pretrained weights: {e}")
    print("Done.")
