import unittest

import torch
from torch import nn

from models.clip_vision import _forward_layer_with_artifacts


class PassThroughAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.Identity()
        self.self_attn = PassThroughAttention()
        self.layer_norm2 = nn.Identity()
        self.mlp = nn.Identity()


class ArtifactFusionTest(unittest.TestCase):
    def test_artifacts_are_appended_with_a_zero_initial_residual(self):
        image_tokens = torch.ones(1, 2, 3)
        artifact_tokens = torch.full((1, 1, 3), 2.0)

        output = _forward_layer_with_artifacts(
            FakeLayer(), image_tokens, artifact_tokens
        )[0]

        self.assertEqual(tuple(output.shape), (1, 3, 3))
        torch.testing.assert_close(output[:, :2], torch.full((1, 2, 3), 4.0))
        torch.testing.assert_close(output[:, 2:], torch.full((1, 1, 3), 4.0))

    def test_mismatched_width_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "width"):
            _forward_layer_with_artifacts(
                FakeLayer(), torch.ones(1, 2, 3), torch.ones(1, 1, 4)
            )


if __name__ == "__main__":
    unittest.main()
