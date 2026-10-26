import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "python"))
from backend import build_command, run_backend, validate_config, write_csv, write_json


def test_validation():
    assert validate_config({})["window"] == 9
    with pytest.raises(ValueError): validate_config({"window": 4})
    with pytest.raises(ValueError): validate_config({"sigma_range": 0})


def test_command_is_argument_list(tmp_path):
    command = build_command(tmp_path / "backend.exe", "denoise", {"mode": "serial"},
                            output_path=tmp_path / "name with spaces.pgm")
    assert command[0].endswith("backend.exe")
    assert command[command.index("--output") + 1].endswith("name with spaces.pgm")


def test_missing_backend(tmp_path):
    with pytest.raises(FileNotFoundError): run_backend([str(tmp_path / "missing"), "info"])


def test_backend_failure_and_json(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text("import json; print(json.dumps({'status':'ok'}))", encoding="utf-8")
    assert run_backend([sys.executable, str(script)]) == {"status": "ok"}


def test_exports(tmp_path):
    write_json(tmp_path / "result.json", {"value": 1})
    write_csv(tmp_path / "result.csv", [{"mode": "serial", "time": 1.0}])
    assert json.loads((tmp_path / "result.json").read_text())["value"] == 1
    assert "mode,time" in (tmp_path / "result.csv").read_text()


def test_native_backend_escapes_windows_output_path(tmp_path):
    root = Path(__file__).parents[2]
    executable = next((path for path in (root / "build-verified" / "denoise.exe",
                                          root / "build" / "denoise.exe") if path.is_file()), None)
    if executable is None:
        pytest.skip("native backend has not been built")
    output = tmp_path / "output image.pgm"
    command = build_command(executable, "denoise", {"mode": "serial", "width": 8, "height": 8,
                                                     "window": 5}, output_path=output)
    result = run_backend(command)
    assert Path(result["output"]) == output
    assert output.is_file()
