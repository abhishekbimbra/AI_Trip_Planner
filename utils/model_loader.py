import os 
from dotenv import load_dotenv
from typing import Literal , Optional, Any
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field


class ConfigLoader:
     
    def __init__(self):
        print(f"Loading Config........")
        self.config = load_config()

    def __getitem__(self,key):
        return self.config[key]
    


class ModelLoader:
    model_provider: Literal["groq", "gemini_flash"] = "groq"
    config: Optional[ConfigLoader]=Field(default=None,exclude=True)

    def model_post_init(self,_context:Any) -> None:
        self.config=ConfigLoader()

    
    class Config:
        arbitrary_types_allowed=True

    def load_llm(self):
        """
        load and return the llm model 

        """
        print("LLm loading.....")
        print(f"Loading model from provider :{self.model_provider}")

        if self.model_provider =="groq":
            print("Loading llm form Groq.......")
            groq_api_key=os.getenv("GROQ_API_KEY")
            model_name = self.config["llm"]["groq"]["model_name"]

            llm = ChatGroq(model_name="qwen/qwen3-32b", api_key=groq_api_key)

        elif self.model_provider == "gemini_flash":
            print("Loading LLm from Gemini Flash ........ ")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            model_name = self.config["llm"]["gemini_flash"]["model_name"]
            llm = ChatGoogleGenerativeAI(model_name="gemini-2.5-flash",api_key=google_api_key)

        return llm 

    