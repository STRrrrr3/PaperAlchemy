import os
from dotenv import load_dotenv

# 1. 没加载前，打印看看（此时应该什么都没有，或者只有你系统的杂乱变量）
print("--- 加载前 ---")
print(f"GOOGLE_API_KEY: {os.environ.get('GOOGLE_API_KEY')}")

# 2. 执行加载魔法
load_dotenv()

# 3. 加载后，再打印看看
print("\n--- 加载后 ---")
print(f"GOOGLE_API_KEY: {os.environ.get('GOOGLE_API_KEY')}")
print(f"HTTPS_PROXY:    {os.environ.get('HTTPS_PROXY')}")

# 4. 验证代理注入是否成功
if os.environ.get('HTTPS_PROXY') == "http://127.0.0.1:7890":
    print("\n✅ 成功！代码已经读到了你的配置。")
else:
    print("\n❌ 失败！没读到配置，请检查 .env 文件名或路径。")