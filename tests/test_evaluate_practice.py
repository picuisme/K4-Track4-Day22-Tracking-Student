"""Kiểm tra việc đọc cấu hình chấm của video luyện."""

import json

from evaluate_practice import DEFAULT_EVAL_CONFIG, PRACTICE_VIDEO, _load_eval_config


def test_load_eval_config_doc_file_khi_co(tmp_path):
    video_dir = tmp_path / PRACTICE_VIDEO
    video_dir.mkdir()
    (video_dir / "eval_config.json").write_text(json.dumps({"benchmark": "ABC", "split": "val"}))

    assert _load_eval_config(tmp_path) == {"benchmark": "ABC", "split": "val"}


def test_load_eval_config_mac_dinh_khi_thieu_file(tmp_path):
    (tmp_path / PRACTICE_VIDEO).mkdir()

    config = _load_eval_config(tmp_path)

    assert config == DEFAULT_EVAL_CONFIG
    assert config is not DEFAULT_EVAL_CONFIG
