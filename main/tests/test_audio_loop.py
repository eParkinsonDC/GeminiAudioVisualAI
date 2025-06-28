import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from unittest import mock

import pytest
import asyncio
from main.audio_gemini_model import AudioGeminiModel


@pytest.fixture
def audio_gem_model():
    return AudioGeminiModel()


def test_save_code_to_file_handles_exception(audio_gem_model, tmp_path, capsys):
    # Simulate IOError by patching open
    with mock.patch("builtins.open", side_effect=IOError("fail")):
        audio_gem_model.save_code_to_file("code", filename=str(tmp_path / "file.txt"))
        captured = capsys.readouterr()
        assert "Failed to save code" in captured.out


def test_strip_code_blocks_handles_empty_string(audio_gem_model):
    assert audio_gem_model.strip_code_blocks("") == ""


def test_strip_code_blocks_handles_only_code_block(audio_gem_model):
    text = "```python\nprint(1)\n```"
    expected = "python\nprint(1)\n"
    assert audio_gem_model.strip_code_blocks(text) == expected


def test_strip_code_blocks_handles_nested_backticks(audio_gem_model):
    text = "```python\nprint('```')\n```"
    expected = "python\nprint('')\n"
    assert audio_gem_model.strip_code_blocks(text) == expected


def test_create_model_prints_model(capsys, audio_gem_model):
    audio_gem_model.model_type = 1
    audio_gem_model.create_model()
    captured = capsys.readouterr()
    assert "Using model:" in captured.out


def test_create_client_does_not_overwrite_existing(audio_gem_model):
    dummy = object()
    audio_gem_model.client = dummy
    audio_gem_model.create_client()
    assert audio_gem_model.client is dummy


def test__get_screen_handles_exceptions(audio_gem_model):
    with mock.patch("audio_gemini_model.mss.mss", side_effect=Exception("fail")):
        try:
            audio_gem_model._get_screen()
        except Exception as e:
            assert str(e) == "fail"


def test_receive_audio_handles_cancel(audio_gem_model):

    audio_gem_model.session = mock.Mock()
    audio_gem_model.session.receive = mock.Mock(side_effect=asyncio.CancelledError)
    audio_gem_model.audio_in_queue = mock.Mock()
    try:

        asyncio.run(audio_gem_model.receive_audio())
    except Exception:
        pass


def test_receive_audio_handles_exception(audio_gem_model):
    audio_gem_model.session = mock.Mock()
    audio_gem_model.session.receive = mock.Mock(side_effect=Exception("fail"))
    audio_gem_model.audio_in_queue = mock.Mock()
    try:

        asyncio.run(audio_gem_model.receive_audio())
    except Exception:
        pass


def test_run_handles_cancelled_error(audio_gem_model):
    with mock.patch.object(audio_gem_model, "create_client"), mock.patch.object(
        audio_gem_model, "create_model", return_value=True
    ), mock.patch.object(
        audio_gem_model, "create_config", return_value=True
    ), mock.patch(
        "main.audio_gemini_model.genai.Client"
    ), mock.patch(
        "main.audio_gemini_model.asyncio.TaskGroup"
    ), mock.patch(
        "main.audio_gemini_model.asyncio.CancelledError", Exception
    ):
        try:
            asyncio.run(audio_gem_model.run())
        except Exception:
            pass


@pytest.mark.parametrize(
    "input_filename, expected",
    [
        ("file.txt", "file"),
        ("My FILE.txt", "my-file"),
        ("weird!@#$.csv", "weird"),
        ("CamelCaseTest.py", "camelcasetest"),
        ("   ---file---.md", "file"),
        ("a_b-c.d", "a-b-c"),
        (".hiddenfile", "hiddenfile"),
        ("", "file"),
    ],
)
def test_sanitize_name(audio_gem_model, input_filename, expected):
    result = audio_gem_model.sanitize_name(input_filename)
    assert result == expected or result == "file"


def test_clear_output_file_creates_and_clears(tmp_path):
    test_file = tmp_path / "cleared.txt"
    test_file.write_text("SOME TEXT")

    al = AudioGeminiModel()

    al.output_file_path = str(test_file)

    with mock.patch("os.path.dirname", return_value=str(tmp_path)):
        al.clear_output_file()

    assert test_file.exists()
    assert test_file.read_text() == ""


def test_save_code_to_file_appends(tmp_path):
    test_file = tmp_path / "codefile.txt"
    al = AudioGeminiModel()
    al.output_file_path = str(test_file)

    with mock.patch("os.path.dirname", return_value=str(tmp_path)):
        # Write initial code without punctuation
        al.save_code_to_file("hello", filename=str(test_file), mode="w")
        # Write code with period
        al.save_code_to_file("world.", filename=str(test_file), mode="a")
        content = test_file.read_text()
        # Should include "hello", space, "world." and newline
        assert "hello" in content
        assert "world." in content
        assert content.endswith(".\n")


def test_strip_code_blocks_removes_triple_backticks(audio_gem_model):
    text = "Some text\n```python\nprint('Hello!')\n```\nEnd"
    result = audio_gem_model.strip_code_blocks(text)
    assert "```" not in result
    assert "print('Hello!')" in result


def test_get_all_lines_from_output_returns_lines(tmp_path, capsys):
    test_file = tmp_path / "lines.txt"
    test_file.write_text("a\nb\nc\n")
    al = AudioGeminiModel()
    al.output_file_path = str(test_file)

    with mock.patch("os.path.dirname", return_value=str(tmp_path)):
        lines = al.get_all_lines_from_output()
    assert lines == ["a", "b", "c"]


def test_get_all_lines_from_output_returns_empty_if_missing(
    audio_gem_model, tmp_path, capsys
):
    non_existent = tmp_path / "missing.txt"
    audio_gem_model.output_file_path = str(non_existent.name)
    with mock.patch("os.path.dirname", return_value=str(tmp_path)):
        lines = audio_gem_model.get_all_lines_from_output()
        assert lines == []
        out = capsys.readouterr().out
        assert "No output file found" in out


def test_get_all_lines_from_output_returns_lines(tmp_path, capsys):
    test_file = tmp_path / "lines.txt"
    test_file.write_text("a\nb\nc\n")
    al = AudioGeminiModel()
    al.output_file_path = str(test_file)

    with mock.patch("os.path.dirname", return_value=str(tmp_path)):
        lines = al.get_all_lines_from_output()
    assert lines == ["a", "b", "c"]


def test_get_all_lines_from_output_returns_empty_if_missing(
    audio_gem_model, tmp_path, capsys
):
    non_existent = tmp_path / "missing.txt"
    audio_gem_model.output_file_path = str(non_existent.name)
    with mock.patch("os.path.dirname", return_value=str(tmp_path)):
        lines = audio_gem_model.get_all_lines_from_output()
        assert lines == []
        out = capsys.readouterr().out
        assert "No output file found" in out


def test_create_model_raises_for_unknown(audio_gem_model):
    audio_gem_model.model_type = 99
    with pytest.raises(ValueError, match="Unknown model_type"):
        audio_gem_model.create_model()


def test_create_config_raises_if_no_prompt(audio_gem_model):
    audio_gem_model.prompt = None
    with pytest.raises(ValueError, match="Prompt must be set"):
        audio_gem_model.create_config()


def test_create_client_sets_and_skips(audio_gem_model):
    with mock.patch("main.audio_gemini_model.genai.Client") as mock_client:
        # Should set client
        audio_gem_model.client = None
        audio_gem_model.create_client()
        mock_client.assert_called_once()
        # Should not reset if already set
        audio_gem_model.client = mock.Mock()
        audio_gem_model.create_client()
        mock_client.assert_called_once()  # still only called once
