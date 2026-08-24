from types import SimpleNamespace

import pytest
import torch

import main
from model.trainer import build_evaluation_details


def test_evaluation_details_preserve_per_flow_partition_and_cpu_float32_artifacts():
    target = torch.tensor([[[4.0, 10.0], [6.0, 20.0], [8.0, 7.0]]])
    prediction = torch.tensor([[[0.0, 12.0], [7.0, 14.0], [5.0, 10.0]]])
    event = torch.tensor([[[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]])

    details, artifacts = build_evaluation_details(
        prediction, target, event, capture_predictions=True
    )

    assert len(details) == 2
    for detail in details:
        assert {
            "valid_count",
            "event_count",
            "normal_count",
            "valid_abs_error_sum",
            "event_abs_error_sum",
            "normal_abs_error_sum",
            "event_signed_error_sum",
        } <= detail.keys()
        assert detail["valid_count"] == detail["event_count"] + detail["normal_count"]
        assert detail["valid_abs_error_sum"] == pytest.approx(
            detail["event_abs_error_sum"] + detail["normal_abs_error_sum"]
        )

    assert details[0]["valid_count"] == 2
    assert details[0]["event_count"] == 1
    assert details[0]["normal_count"] == 1
    assert details[0]["event_signed_error_sum"] == pytest.approx(-3.0)
    assert details[1]["valid_count"] == 3
    assert details[1]["event_count"] == 2
    assert details[1]["normal_count"] == 1
    assert details[1]["event_signed_error_sum"] == pytest.approx(5.0)

    assert set(artifacts) == {"prediction", "target", "event", "valid"}
    for value in artifacts.values():
        assert value.device.type == "cpu"
        assert value.dtype == torch.float32
    assert torch.equal(artifacts["event"], torch.tensor([[[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]]))
    assert torch.equal(artifacts["valid"], torch.tensor([[[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]))


def test_evaluation_details_reject_nonbinary_events_before_valid_intersection():
    target = torch.tensor([[[4.0], [6.0]]])
    prediction = target.clone()
    event = torch.tensor([[[0.5], [0.0]]])

    with pytest.raises(ValueError, match="bool or exact binary 0/1"):
        build_evaluation_details(prediction, target, event)


def test_evaluation_details_intersect_legacy_binary_events_with_valid_targets():
    target = torch.tensor([[[4.0], [6.0]]])
    prediction = target.clone()
    event = torch.tensor([[[1.0], [0.0]]])

    details, artifacts = build_evaluation_details(
        prediction, target, event, capture_predictions=True
    )

    assert details[0]["valid_count"] == 1
    assert details[0]["event_count"] == 0
    assert details[0]["normal_count"] == 1
    assert torch.equal(artifacts["event"], torch.tensor([[[0.0], [0.0]]]))


def _patch_test_supervisor_dependencies(monkeypatch, trainer_class):
    class FakeModel(torch.nn.Module):
        def __init__(self, args):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def load_state_dict(self, state_dict):
            self.loaded_state_dict = state_dict

    monkeypatch.setattr(main, "init_seed", lambda seed: None)
    monkeypatch.setattr(main, "get_dataloader", lambda **kwargs: {
        "train": [object()], "test": [object()], "scaler": object()
    })
    monkeypatch.setattr(main, "load_graph", lambda graph_file, device: [0, 1])
    monkeypatch.setattr(main, "get_model_params_grouped", lambda model, args: ([model.weight], [], []))
    monkeypatch.setattr(main, "Trainer", trainer_class)
    monkeypatch.setattr(main.torch.optim, "Adam", lambda groups: object())
    monkeypatch.setattr(main.torch, "load", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr("model.models.STSSL", FakeModel)


def test_model_supervisor_uses_explicit_evaluation_phase_and_exposes_artifacts(monkeypatch):
    calls = []

    class FakeTrainer:
        def __init__(self, model, optimizer, dataloader, graph, args):
            self.logger = SimpleNamespace(info=lambda message: None)
            self.args = args

        @staticmethod
        def test(model, dataloader, scaler, graph, logger, args, phase, capture_predictions=False):
            calls.append((phase, capture_predictions))
            model.last_test_details = [{"valid_count": 1}]
            model.last_test_artifacts = {"prediction": torch.ones(1, dtype=torch.float32)}
            return torch.tensor([[1.0, 2.0]]).numpy()

    _patch_test_supervisor_dependencies(monkeypatch, FakeTrainer)
    args = SimpleNamespace(
        seed=1, device="cpu", data_dir="data", dataset="dataset", batch_size=1,
        test_batch_size=1, graph_file="graph", stop_after_phase="bias",
        evaluation_phase="pred_2", ablation_mode="original", lr_init=0.01,
        mode="test", best_path="weights", capture_predictions=True,
    )

    results = main.model_supervisor(args)

    assert calls == [("pred_2", True)]
    assert "test_results" in results
    assert results["test_details"] == [{"valid_count": 1}]
    assert results["test_artifacts"] == {"prediction": torch.ones(1, dtype=torch.float32)}


def test_model_supervisor_falls_back_when_evaluation_phase_is_none(monkeypatch):
    phases = []

    class FakeTrainer:
        def __init__(self, model, optimizer, dataloader, graph, args):
            self.logger = SimpleNamespace(info=lambda message: None)
            self.args = args

        @staticmethod
        def test(model, dataloader, scaler, graph, logger, args, phase, capture_predictions=False):
            phases.append(phase)
            return torch.tensor([[1.0, 2.0]]).numpy()

    _patch_test_supervisor_dependencies(monkeypatch, FakeTrainer)
    args = SimpleNamespace(
        seed=1, device="cpu", data_dir="data", dataset="dataset", batch_size=1,
        test_batch_size=1, graph_file="graph", stop_after_phase="pred_2",
        evaluation_phase=None, ablation_mode="original", lr_init=0.01,
        mode="test", best_path="weights",
    )

    main.model_supervisor(args)

    assert phases == ["pred_2"]


def test_model_supervisor_exposes_training_artifacts_when_capture_is_requested(monkeypatch):
    artifact = {"prediction": torch.ones(1, dtype=torch.float32)}

    class FakeTrainer:
        def __init__(self, model, optimizer, dataloader, graph, args):
            self.logger = SimpleNamespace(info=lambda message: None)
            self.args = args
            self.model = model

        def train(self):
            self.model.last_test_details = [{"valid_count": 1}]
            self.model.last_test_artifacts = artifact
            return {"test_results": torch.tensor([[1.0, 2.0]]).numpy()}

    _patch_test_supervisor_dependencies(monkeypatch, FakeTrainer)
    args = SimpleNamespace(
        seed=1, device="cpu", data_dir="data", dataset="dataset", batch_size=1,
        test_batch_size=1, graph_file="graph", stop_after_phase="bias",
        evaluation_phase="bias", ablation_mode="original", lr_init=0.01,
        mode="train", capture_predictions=True,
    )

    results = main.model_supervisor(args)

    assert results["test_artifacts"] is artifact


def test_selected_train_component_forwards_prediction_capture(monkeypatch):
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.ones(1))

    trainer = main.Trainer.__new__(main.Trainer)
    trainer.model = FakeModel()
    trainer.args = SimpleNamespace(
        epochs=1, load_path=None, start_phase="bias", evaluation_phase="bias",
        stop_after_phase="bias", capture_predictions=True, debug=True,
        early_stop=False, device="cpu",
    )
    trainer.logger = SimpleNamespace(
        info=lambda message: None, warning=lambda message: None
    )
    trainer.optimizer = SimpleNamespace(state_dict=lambda: {})
    trainer.val_loader = [object()]
    trainer.test_loader = [object()]
    trainer.scaler = object()
    trainer.graph = object()
    trainer.load_path_baseline_used = False
    trainer.save_weights = lambda weights: None
    trainer.plot_losses = lambda *args: None
    trainer.train_epoch = lambda epoch, weights, losses, pred_losses, class_losses, phase: (
        1.0, [1.0], [1.0], [0.0], weights
    )
    trainer.val_epoch = lambda epoch, loader, weights, phase: (1.0, 1.0)

    def fake_test(model, dataloader, scaler, graph, logger, args, phase, capture_predictions=False):
        calls.append((phase, capture_predictions))
        return torch.tensor([[1.0, 2.0]]).numpy()

    monkeypatch.setattr(main.Trainer, "test", staticmethod(fake_test))

    main.Trainer.train_component(trainer, [], [], "bias", esp=30)

    assert calls == [("bias", True)]


def test_trainer_test_does_not_run_unused_classifier_inference():
    class FakeModel(torch.nn.Module):
        def forward(self, data, graph):
            return data, data

        def predict(self, repr1, repr1_cls, phase):
            return repr1

        def classify_evs(self, repr1, repr1_cls):
            raise AssertionError("test evaluation must not run unused classifier inference")

    class IdentityScaler:
        @staticmethod
        def inverse_transform(value):
            return value

    target = torch.tensor([[[6.0, 7.0], [8.0, 9.0]]])
    event = torch.zeros_like(target)

    main.Trainer.test(
        FakeModel(), [(target, target, event, None)], IdentityScaler(), object(),
        SimpleNamespace(info=lambda message: None),
        SimpleNamespace(ablation_mode="original"), "pred",
    )


def test_model_supervisor_reraises_when_requested(monkeypatch):
    class FakeTrainer:
        def __init__(self, model, optimizer, dataloader, graph, args):
            self.logger = SimpleNamespace(info=lambda message: None)
            self.args = args

        def train(self):
            raise RuntimeError("training failure")

    _patch_test_supervisor_dependencies(monkeypatch, FakeTrainer)
    args = SimpleNamespace(
        seed=1, device="cpu", data_dir="data", dataset="dataset", batch_size=1,
        test_batch_size=1, graph_file="graph", stop_after_phase="bias",
        ablation_mode="original", lr_init=0.01, mode="train", raise_exceptions=True,
    )

    with pytest.raises(RuntimeError, match="training failure"):
        main.model_supervisor(args)
