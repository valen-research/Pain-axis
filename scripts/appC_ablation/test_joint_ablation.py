"""CPU regressions: python -m unittest discover -s scripts/appC_ablation -p 'test_*.py'.

Extract the projection helpers and condition block without executing the notebook's
GPU setup, model downloads, authentication, or interactive prompts.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


def load_projection_code(filename):
    tree = ast.parse(Path(__file__).with_name(filename).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and (node.name.startswith("orthogonal") or node.name in
                      {"get_decoder_layers", "residual_write_matrices"})]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    condition = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == "specs")
    condition_code = compile(ast.Module(body=[condition], type_ignores=[]), filename, "exec")
    specs = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "CONDITION_SPECS"
                         for t in node.targets))
    return namespace, condition_code, specs


def fixture(scaled=False, dtype=torch.float32):
    # Rectangular weights exercise both embedding and residual-output orientations.
    weights = torch.tensor([[1., 2., 0.], [0., 1., 3.], [2., 0., 1.],
                            [1., -1., 2.], [0., 2., -1.]], dtype=dtype)
    def parameter(value):
        return torch.nn.Parameter(value.clone(), requires_grad=False)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=parameter(weights))),
        mlp=SimpleNamespace(down_proj=SimpleNamespace(weight=parameter(weights))))
    if scaled:
        scale = torch.tensor([0.5, 1.5, 2., 0.75, 1.25], dtype=dtype)
        layer.post_attention_layernorm = SimpleNamespace(weight=parameter(scale - 1))
        layer.post_feedforward_layernorm = SimpleNamespace(weight=parameter(scale - 1))
    embed = SimpleNamespace(weight=parameter(weights.T))
    return SimpleNamespace(model=SimpleNamespace(layers=[layer]),
                           get_input_embeddings=lambda: embed)


class JointAblationTests(unittest.TestCase):
    scripts = ("01_ablation_small_models.py", "02_ablation_large_models.py")

    def apply(self, code, model, vectors, condition="s1s2", reverse=False):
        namespace, condition_code, conditions = code
        specs = conditions[condition]
        exec(condition_code, dict(namespace, model=model, cond=condition,
             specs=list(reversed(specs)) if reverse else specs,
             single_direction=lambda key, layer: vectors[key],
             layer_of={"s1": 0, "s2": 0}, print=lambda *args, **kwargs: None))

    def test_combined_conditions_remove_every_direction(self):
        eye = torch.eye(5)
        vectors = {"s1_pain_vector": eye[0],
                   "s2_pain_vector": (eye[0] + eye[1]) / 2**0.5,
                   "fear_vector": (eye[0] + eye[2]) / 2**0.5,
                   "negemotion_vector": (eye[1] + eye[2]) / 2**0.5}
        for script in self.scripts:
            code = load_projection_code(script)
            for scaled in (False, True):
                for dtype in (torch.float32, torch.bfloat16):
                    for condition in ("s1s2", "s1s2_fear", "s1s2_negval"):
                        with self.subTest(script=script, scaled=scaled, dtype=dtype, condition=condition):
                            model = fixture(scaled, dtype)
                            original = {name: w.clone() for name, w, _ in
                                        code[0]["residual_write_matrices"](model)}
                            self.apply(code, model, vectors, condition)
                            directions = torch.stack([vectors[key] for key, _ in code[2][condition]])
                            for name, w, scale in code[0]["residual_write_matrices"](model):
                                effective = w.float()
                                if name == "embed":
                                    remaining = effective @ directions.T
                                    untouched = w[:, 3:]
                                    before = original[name][:, 3:]
                                else:
                                    if scale is not None:
                                        effective = scale[:, None] * effective
                                    remaining = directions @ effective
                                    untouched, before = w[3:], original[name][3:]
                                torch.testing.assert_close(remaining, torch.zeros_like(remaining),
                                                           atol=0.03 if dtype == torch.bfloat16 else 1e-5,
                                                           rtol=0)
                                torch.testing.assert_close(untouched, before, atol=1e-6, rtol=0)

    def test_order_independence_and_dependent_directions(self):
        e1, e2 = torch.eye(5)[:2]
        for script in self.scripts:
            code = load_projection_code(script)
            for scaled in (False, True):
                with self.subTest(script=script, scaled=scaled):
                    vectors = {"s1_pain_vector": e1, "s2_pain_vector": (e1 + e2) / 2**0.5}
                    forward, reverse = fixture(scaled), fixture(scaled)
                    self.apply(code, forward, vectors)
                    self.apply(code, reverse, vectors, reverse=True)
                    vectors["s2_pain_vector"] = -e1
                    single, duplicate = fixture(scaled), fixture(scaled)
                    self.apply(code, single, vectors, "s1")
                    self.apply(code, duplicate, vectors)
                    matrices = code[0]["residual_write_matrices"]
                    for first, second in ((forward, reverse), (single, duplicate)):
                        for (_, a, _), (_, b, _) in zip(matrices(first), matrices(second)):
                            torch.testing.assert_close(a, b, atol=2e-6, rtol=0)

    def test_single_direction_preserves_existing_formula(self):
        direction = torch.tensor([1., 2., 3., 4., 5.])
        direction /= direction.norm()
        for script in self.scripts:
            code = load_projection_code(script)
            for scaled in (False, True):
                with self.subTest(script=script, scaled=scaled):
                    model = fixture(scaled)
                    expected = []
                    for name, weight, scale in code[0]["residual_write_matrices"](model):
                        w = weight.clone().float()
                        r = direction.unsqueeze(0)
                        if name == "embed":
                            w -= (w @ r.T) @ r
                        elif scale is None:
                            w -= r.T @ (r @ w)
                        elif script.startswith("01"):
                            s = torch.where(scale.abs() < 1e-4, torch.full_like(scale, 1e-4), scale)
                            w -= (r.T @ (r @ (w * s[:, None]))) / s[:, None]
                        else:
                            v = scale * direction
                            v = (v / v.norm()).unsqueeze(0)
                            w -= v.T @ (v @ w)
                        expected.append(w)
                    self.apply(code, model, {"s1_pain_vector": direction}, "s1")
                    for (_, actual, _), before in zip(code[0]["residual_write_matrices"](model), expected):
                        self.assertTrue(torch.equal(actual, before))


if __name__ == "__main__":
    unittest.main()
