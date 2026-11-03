from xrd_geometry.factory import _backbone_source


def test_local_backbone_override_forces_offline_loading(monkeypatch):
    model_id = "microsoft/swin-base-patch4-window12-384-in22k"
    monkeypatch.setenv("XRD_BACKBONE_PATH", "/models/swin-local")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    source, local_files_only = _backbone_source(model_id)

    assert source == "/models/swin-local"
    assert local_files_only is True
    assert model_id == "microsoft/swin-base-patch4-window12-384-in22k"


def test_hugging_face_offline_mode_keeps_portable_model_id(monkeypatch):
    model_id = "microsoft/swin-base-patch4-window12-384-in22k"
    monkeypatch.delenv("XRD_BACKBONE_PATH", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    source, local_files_only = _backbone_source(model_id)

    assert source == model_id
    assert local_files_only is True
