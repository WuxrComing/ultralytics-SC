# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import SC_ELAN_LSKA_TSCG
from ultralytics.nn.tasks import DetectionModel


def test_sc_elan_lska_tscg_forward_shape():
    """SC_ELAN_LSKA_TSCG should preserve spatial size and output c2 channels."""
    model = SC_ELAN_LSKA_TSCG(c1=64, c2=128, c3=128, c4=128, c5=1).eval()
    x = torch.randn(2, 64, 48, 48)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 128, 48, 48)


def test_sc_elan_lska_tscg_gate_range_and_effect():
    """TSCG gate should be in [0, 1] and produce non-trivial blending."""
    torch.manual_seed(0)
    model = SC_ELAN_LSKA_TSCG(c1=32, c2=64, c3=64, c4=64, c5=1).eval()
    x = torch.randn(1, 32, 32, 32)

    with torch.no_grad():
        y_split = list(model.cv1(x).chunk(2, 1))
        y_split.extend((m(y_split[-1])) for m in [model.cv2, model.cv3])
        feat = model.cv4(torch.cat(y_split, 1))
        context = model.interaction(feat)
        gate = model.tscg.detail(feat)
        out = model.tscg(feat, context)

    assert gate.min().item() >= 0.0
    assert gate.max().item() <= 1.0
    assert out.shape == feat.shape == context.shape
    assert not torch.allclose(out, feat), "Output should differ from identity feature for random input."


def test_sc_elan_lska_tscg_gradient_flow():
    """Gradients should flow through both LSKA and TSCG branches."""
    torch.manual_seed(1)
    model = SC_ELAN_LSKA_TSCG(c1=64, c2=128, c3=128, c4=128, c5=1).train()
    x = torch.randn(2, 64, 40, 40, requires_grad=True)
    target = torch.randn(2, 128, 40, 40)

    y = model(x)
    loss = nn.MSELoss()(y, target)
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
    assert model.interaction.conv1.weight.grad is not None
    assert model.interaction.conv1.weight.grad.abs().sum().item() > 0
    assert model.tscg.detail[-2].weight.grad is not None
    assert model.tscg.detail[-2].weight.grad.abs().sum().item() > 0


def test_scelan_lska_tscg_yaml_build_and_forward():
    """Model YAML using SC_ELAN_LSKA_TSCG should be parsable and runnable."""
    repo_root = Path(__file__).resolve().parents[1]
    yaml_path = repo_root / "yolo11-scelan-lska-tscg.yaml"
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
