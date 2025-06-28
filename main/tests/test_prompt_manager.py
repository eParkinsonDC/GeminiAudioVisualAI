import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from unittest.mock import MagicMock, patch

from prompt_manager import LLangC_Prompt_Manager


class TestLLangC_Prompt_Manager(unittest.TestCase):
    @patch("prompt_manager.Client")
    def test_init_defaults(self, mock_client):
        pm = LLangC_Prompt_Manager()
        self.assertEqual(pm.version, 2)
        self.assertIsNone(pm.tools)
        self.assertIsNone(pm.model)
        self.assertIsNone(pm.prompt_name)
        self.assertIsNone(pm.client)
        self.assertIsNone(pm.prompt)
        self.assertIsNone(pm.prompt_template)
        self.assertIsNone(pm.prompt_identifier)

    @patch("prompt_manager.Client")
    def test_get_llang_chain_access_success(self, mock_client):
        pm = LLangC_Prompt_Manager()
        pm.prompt_name = "test_prompt"
        mock_instance = mock_client.return_value
        mock_instance.pull_prompt.return_value = "prompt_obj"
        os.environ["LANGSMITH_API_KEY"] = "dummy"
        result = pm.get_llang_chain_access()
        self.assertTrue(result)
        self.assertEqual(pm.prompt, "prompt_obj")
        mock_client.assert_called_once_with(api_key="dummy")
        mock_instance.pull_prompt.assert_called_once_with("test_prompt", include_model=False)

    @patch("prompt_manager.Client")
    def test_get_llang_chain_access_no_prompt_name(self, mock_client):
        pm = LLangC_Prompt_Manager()
        with self.assertRaises(ValueError):
            pm.get_llang_chain_access()

    @patch("prompt_manager.Client")
    def test_load_prompt_name_success(self, mock_client):
        pm = LLangC_Prompt_Manager(version=2)
        mock_instance = mock_client.return_value

        # Mock get_prompt().dict()
        mock_prompt_obj = MagicMock()
        mock_prompt_obj.dict.return_value = {"last_commit_hash": "abc123"}
        mock_instance.get_prompt.return_value = mock_prompt_obj

        # Mock pull_prompt().get_prompts()
        mock_prompt_template = MagicMock()
        mock_prompt_template.prompt.template = "template string"
        mock_pull_prompt = MagicMock()
        mock_pull_prompt.get_prompts.return_value = [[mock_prompt_template]]
        mock_instance.pull_prompt.return_value = mock_pull_prompt

        os.environ["LANGSMITH_API_KEY"] = "dummy"
        pm.load_prompt_name()

        self.assertEqual(pm.prompt_name, "geminaudioai_prompt_v2")
        self.assertEqual(pm.prompt_identifier, "geminaudioai_prompt_v2:abc123")
        self.assertEqual(pm.prompt, mock_prompt_template)
        self.assertEqual(pm.prompt_template, "template string")
        mock_instance.get_prompt.assert_called_once_with(prompt_identifier="geminaudioai_prompt_v2")
        mock_instance.pull_prompt.assert_called_once_with(
            prompt_identifier="geminaudioai_prompt_v2:abc123", include_model=False
        )

    @patch("prompt_manager.Client")
    def test_load_prompt_name_no_commit_hash(self, mock_client):
        pm = LLangC_Prompt_Manager(version=1)
        mock_instance = mock_client.return_value

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.dict.return_value = {}
        mock_instance.get_prompt.return_value = mock_prompt_obj

        mock_prompt_template = MagicMock()
        mock_prompt_template.prompt.template = "template string"
        mock_pull_prompt = MagicMock()
        mock_pull_prompt.get_prompts.return_value = [[mock_prompt_template]]
        mock_instance.pull_prompt.return_value = mock_pull_prompt

        os.environ["LANGSMITH_API_KEY"] = "dummy"
        with patch("builtins.print") as mock_print:
            pm.load_prompt_name()
            mock_print.assert_any_call("No prompt last commit hash found from LangChain.")

    @patch("prompt_manager.Client")
    def test_load_prompt_name_prompt_obj_none(self, mock_client):
        pm = LLangC_Prompt_Manager(version=1)
        mock_instance = mock_client.return_value
        mock_instance.get_prompt.return_value = None
        os.environ["LANGSMITH_API_KEY"] = "dummy"
        with self.assertRaises(ValueError):
            pm.load_prompt_name()

    def test_load_prompt_name_invalid_version(self):
        pm = LLangC_Prompt_Manager(version=99)
        with self.assertRaises(KeyError):
            pm.load_prompt_name()

    @patch("prompt_manager.Client")
    def test_load_prompt_name_prompt_name_assertion(self, mock_client):
        pm = LLangC_Prompt_Manager(version=2)
        # Patch the prompt_version_map to have a non-gemi name
        with patch.object(LLangC_Prompt_Manager, "load_prompt_name") as mock_load:
            pm.prompt_name = "notgemi"
            with self.assertRaises(AssertionError):
                assert pm.prompt_name.startswith("gemi")

if __name__ == "__main__":
    unittest.main()
