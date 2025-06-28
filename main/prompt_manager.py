import os
from langsmith import Client


class LlangChainPromptManager:
    def __init__(self, version=2):
        self.version = version
        self.tools = None
        self.model = None
        self.prompt_name = None
        self.client = None
        self.prompt = None
        self.prompt_template = None
        self.prompt_identifier = None

    def get_llang_chain_access(self):
        # Only init client if not set
        if not self.client:
            api_key = os.environ.get("LANGSMITH_API_KEY")
            if not api_key:
                raise EnvironmentError("LANGSMITH_API_KEY environment variable not set")
            self.client = Client(api_key=api_key)
        if not self.prompt_name:
            raise ValueError("Prompt name must be set before fetching prompt.")
        identifier = self.prompt_identifier or self.prompt_name
        self.prompt = self.client.pull_prompt(identifier, include_model=False)
        return True

    def load_prompt_name(self):
        prompt_version_map = {
            1: "geminiaudioai_prompt_version_1",
            2: "geminaudioai_prompt_v2",
            3: "gemaudioprompt_v3",
            4: "gemini_audioprompt_memoryrecall_v4",
        }
        self.prompt_name = prompt_version_map[self.version]
        assert self.prompt_name.startswith("gem")

        if not self.client:
            api_key = os.environ.get("LANGSMITH_API_KEY")
            if not api_key:
                raise EnvironmentError("LANGSMITH_API_KEY environment variable not set")
            self.client = Client(api_key=api_key)

        prompt_obj = self.client.get_prompt(prompt_identifier=self.prompt_name)
        if prompt_obj is None:
            raise ValueError(f"Prompt object for '{self.prompt_name}' not found.")
        prompt_dict = prompt_obj.dict()
        prompt_last_comm_hash = prompt_dict.get("last_commit_hash", None)
        if prompt_last_comm_hash:
            self.prompt_identifier = f"{self.prompt_name}:{prompt_last_comm_hash}"
            prompt_pulled = self.client.pull_prompt(
                prompt_identifier=self.prompt_identifier, include_model=False
            ).get_prompts()[0][0]
            self.prompt = prompt_pulled
            self.prompt_template = prompt_pulled.prompt.template
            print("Prompt Template loaded with name", self.prompt_name)
            print(f"Prompt template {self.prompt_template}")
        else:
            print("No prompt last commit hash found from LangChain.")
            return None
        return prompt_last_comm_hash
