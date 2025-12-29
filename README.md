PaperAlchemy ⚗️
================

基于多智能体协同的学术论文网页自动化构建系统  
(Automated Paper-to-Page Construction based on Multi-Agent Collaboration)

📖 项目简介

PaperAlchemy 是一个致力于将静态的 PDF 学术论文“点石成金”，转化为动态、交互式项目网页的智能系统。它利用多智能体 (Multi-Agent) 架构与大语言模型 (LLM) 技术，实现从多模态解析到代码生成的全自动化流程。

🛠️ 技术栈

Parser Agent: Docling / LayoutLM (PDF 多模态解析)

Planner Agent: LLM (内容重组与规划)

Coder Agent: LLM (HTML/CSS/JS 代码生成)

UI: React/Vue (人机协作界面)

🚀 快速开始

安装依赖: pip install -r requirements.txt

运行解析: python src/parser.py