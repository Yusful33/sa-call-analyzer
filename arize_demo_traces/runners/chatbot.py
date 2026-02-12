"""Backward-compatible re-export. Actual implementation moved to langchain/chatbot.py."""
from .langchain.chatbot import run_chatbot

__all__ = ["run_chatbot"]
