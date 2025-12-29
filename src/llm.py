import os
import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import ssl

load_dotenv()

# 强制修补 SSL 上下文 强制 Python 全局忽略 SSL 证书验证 防止握手失败导致的断连
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

proxy_url = os.getenv("HTTPS_PROXY") or "http://127.0.0.1:7890"
os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url
os.environ["all_proxy"] = proxy_url
os.environ["CURL_CA_BUNDLE"] = ""

def get_llm(temperature: float = 0, use_smart_model: bool = True):
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("未找到API")

    if use_smart_model:
        # 价格：2 美元 / 12 美元（<20 万个 token）4 美元 / 18 美元（>20 万个 token）
        model_name = "gemini-3-pro-preview" 
        timeout_setting = 500.0
    else:
        # 价格：$0.50 / $3
        model_name = "gemini-3-flash-preview"
        timeout_setting = 60.0

    print(f"[PaperAlchemy] Initializing Gemini: {model_name} (temp={temperature})")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        transport="rest", 
        max_retries=3,
        timeout=timeout_setting,
        # Gemini 的一个特殊设置，防止它因为安全设置拒绝回答某些学术敏感内容
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        },
        convert_system_message_to_human=True # 兼容性设置
    )