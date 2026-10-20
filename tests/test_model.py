import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(HAS_TORCH, "torch/torchvision not installed")
class TestModels(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.x = torch.randn(2, 3, 64, 64)

    def test_forward_shapes_all_archs(self):
        from btvit.model import build_model

        for arch in ("hybrid", "cnn", "vit"):
            for backbone in ("light", "resnet18"):
                m = build_model(arch, num_classes=4, img_size=64, backbone=backbone, depth=2, heads=4, dim=64)
                self.assertEqual(tuple(m(self.x).shape), (2, 4), (arch, backbone))

    def test_wrong_image_size_raises(self):
        import torch

        from btvit.model import build_model

        m = build_model("hybrid", img_size=64, backbone="light", dim=64, depth=1, heads=4)
        with self.assertRaises(ValueError):
            m(torch.randn(1, 3, 96, 96))

    def test_backward_reaches_cnn_and_transformer(self):
        from btvit.model import build_model

        m = build_model("hybrid", img_size=64, backbone="light", dim=64, depth=2, heads=4)
        m(self.x).sum().backward()
        self.assertIsNotNone(m.cnn.features[0][0].weight.grad)
        self.assertIsNotNone(m.encoder.blocks[0].attn.in_proj_weight.grad)

    def test_gradcam_and_rollout_maps(self):
        from btvit.explain import GradCAM, attention_rollout
        from btvit.model import build_model

        m = build_model("hybrid", img_size=64, backbone="light", dim=64, depth=2, heads=4)
        cam = GradCAM(m)
        heat, cls = cam(self.x[:1])
        cam.remove()
        self.assertEqual(tuple(heat.shape), (64, 64))
        self.assertTrue(0 <= float(heat.min()) and float(heat.max()) <= 1.0 + 1e-6)
        roll = attention_rollout(m, self.x[:1])
        self.assertEqual(tuple(roll.shape), (64, 64))

    def test_can_overfit_tiny_batch(self):
        import torch

        from btvit.model import build_model

        m = build_model("hybrid", img_size=64, backbone="light", dim=64, depth=2, heads=4, dropout=0.0)
        x, y = torch.randn(8, 3, 64, 64), torch.arange(8) % 4
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        first = None
        for _ in range(40):
            loss = torch.nn.functional.cross_entropy(m(x), y)
            first = first if first is not None else loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.5 * first)

    def test_end_to_end_smoke_training(self):
        from btvit.train import main

        with tempfile.TemporaryDirectory() as d:
            m = main(["--fake", "--epochs", "1", "--img-size", "64", "--backbone", "light", "--dim", "64",
                      "--depth", "1", "--heads", "4", "--batch-size", "8", "--device", "cpu", "--out", d])
            self.assertIn("macro_f1", m)
            for f in ("best.pt", "history.csv", "test_metrics.json", "confusion_matrix.png", "explanations.png"):
                self.assertTrue((Path(d) / f).exists(), f)


class TestDedup(unittest.TestCase):
    def test_detects_duplicate_across_splits(self):
        import numpy as np
        from PIL import Image

        from btvit.dedup import find_leaks

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split in ("Training", "Testing"):
                (root / split / "glioma").mkdir(parents=True)
            imgs = [rng.integers(0, 255, (64, 64), dtype=np.uint8) for _ in range(3)]
            for i, a in enumerate(imgs):
                Image.fromarray(a).save(root / "Training" / "glioma" / f"{i}.png")
            Image.fromarray(imgs[1]).save(root / "Testing" / "glioma" / "leak.png")
            Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(root / "Testing" / "glioma" / "new.png")
            leaks, n_tr, n_te = find_leaks(root, max_distance=4)
            self.assertEqual((n_tr, n_te), (3, 2))
            self.assertEqual([p.name for p, _, _ in leaks], ["leak.png"])


if __name__ == "__main__":
    unittest.main()
