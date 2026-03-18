import os
from dotenv import load_dotenv

print("--- 加载前 ---")
print(f"GOOGLE_API_KEY: {os.environ.get('GOOGLE_API_KEY')}")

# 执行加载
load_dotenv()

# 加载再打印
print("\n--- 加载后 ---")
print(f"GOOGLE_API_KEY: {os.environ.get('GOOGLE_API_KEY')}")
print(f"HTTPS_PROXY:    {os.environ.get('HTTPS_PROXY')}")

# 验证代理注入
if os.environ.get('HTTPS_PROXY') == "http://127.0.0.1:7890":
    print("\n✅ 成功！代码已经读到了你的配置。")
else:
    print("\n❌ 失败！没读到配置，请检查 .env 文件名或路径。")