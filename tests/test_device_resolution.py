import torch

from main import resolve_device


def test_explicit_cpu_does_not_query_cuda(monkeypatch):
    def fail_if_queried():
        raise AssertionError("explicit CPU must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", fail_if_queried)

    assert resolve_device("cpu") == "cpu"


def test_absent_device_selects_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device(None) == "cuda"


def test_absent_or_unavailable_requested_cuda_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device(None) == "cpu"
    assert resolve_device("cuda:0") == "cpu"
