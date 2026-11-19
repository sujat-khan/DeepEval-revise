from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        self.model = ChatGroq(model=model_name)
        super().__init__(model_name=model_name)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return str(self.model.invoke(prompt).content)

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return str(res.content)

    def get_model_name(self) -> str:
        return self.model_name
