# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch

from ultralytics.nn.modules.head import Detect, DetectCAI
from ultralytics.nn.tasks import DetectionModel


def _make_feats(ch=(256, 512, 1024), b=2):
    return [
        torch.randn(b, ch[0], 80, 80),
        torch.randn(b, ch[1], 40, 40),
        torch.randn(b, ch[2], 20, 20),
    ]


def test_detect_cai_infer_compatible_with_detect_shape():
    """DetectCAI should keep inference tensor shape contract identical to Detect."""
    x = _make_feats()
    head_ref = Detect(nc=80, ch=(256, 512, 1024)).eval()
    head_cai = DetectCAI(nc=80, ch=(256, 512, 1024)).eval()

    with torch.no_grad():
        y_ref = head_ref([xi.clone() for xi in x])[0]
        y_cai = head_cai([xi.clone() for xi in x])[0]

    assert y_ref.shape == y_cai.shape


def test_detect_cai_train_eval_differs():
    """DetectCAI should apply CAI in train mode and bypass in eval mode."""
    torch.manual_seed(0)
    x = _make_feats(ch=(128, 256, 512), b=1)
    head = DetectCAI(nc=10, ch=(128, 256, 512))

    head.train()
    preds_train = head([xi.clone() for xi in x])

    head.eval()
    with torch.no_grad():
        preds_eval = head([xi.clone() for xi in x])[1]

    assert isinstance(preds_train, dict)
    assert isinstance(preds_eval, dict)
    assert preds_train["scores"].shape == preds_eval["scores"].shape
    assert not torch.allclose(preds_train["scores"], preds_eval["scores"])


def test_detect_cai_yaml_build_and_forward():
    """Model YAML using DetectCAI should be parsable and runnable."""
    repo_root = Path(__file__).resolve().parents[1]
    yaml_path = repo_root / "yolo11-scelan-lska-tscg-detect-cai.yaml"
    assert yaml_path.exists(), f"Missing YAML file: {yaml_path}"

    model = DetectionModel(cfg=str(yaml_path), ch=3, nc=80)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        preds = model.predict(x)

    assert isinstance(preds, dict)
    assert "boxes" in preds
    assert isinstance(preds["boxes"], torch.Tensor)
    assert preds["boxes"].shape[0] == 1
    assert preds["boxes"].numel() > 0


def test_detect_cai_visdrone_instance_prior_default():
    """DetectCAI with nc=10 should initialize prior from VisDrone instance table."""
    head = DetectCAI(nc=10, ch=(128, 256, 512))
    expected = torch.tensor([21000, 6376, 1302, 28063, 5770, 2659, 530, 599, 2938, 5845], dtype=torch.float32)
    expected = expected / expected.sum()
    assert torch.allclose(head.cai_class_prior, expected)
    assert head.cai_tail_mask[1].item() == 1.0
    assert head.cai_tail_mask[2].item() == 1.0
    assert head.cai_tail_mask[6].item() == 1.0
    assert head.cai_tail_mask[7].item() == 1.0
