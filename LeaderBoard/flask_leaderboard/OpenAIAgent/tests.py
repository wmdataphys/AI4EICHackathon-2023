from langchain.memory.chat_message_histories import SQLChatMessageHistory

chat_message_history = SQLChatMessageHistory(
    session_id="test_session", connection_string="sqlite:///sqlite.db"
)

chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi")



from datetime import datetime
from typing import Any

from langchain.memory.chat_message_histories.sql import BaseMessageConverter
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import Column, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CustomMessage(Base):
    __tablename__ = "custom_message_store"

    id = Column(Integer, primary_key=True)
    session_id = Column(Text)
    type = Column(Text)
    content = Column(Text)
    created_at = Column(DateTime)
    author_email = Column(Text)


class CustomMessageConverter(BaseMessageConverter):
    def __init__(self, author_email: str):
        self.author_email = author_email

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        if sql_message.type == "human":
            return HumanMessage(
                content=sql_message.content,
            )
        elif sql_message.type == "ai":
            return AIMessage(
                content=sql_message.content,
            )
        elif sql_message.type == "system":
            return SystemMessage(
                content=sql_message.content,
            )
        else:
            raise ValueError(f"Unknown message type: {sql_message.type}")

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        now = datetime.now()
        return CustomMessage(
            session_id=session_id,
            type=message.type,
            content=message.content,
            created_at=now,
            author_email=self.author_email,
        )

    def get_sql_model_class(self) -> Any:
        return CustomMessage


chat_message_history = SQLChatMessageHistory(
    session_id="test_session",
    connection_string="sqlite:///sqlite.db",
    custom_message_converter=CustomMessageConverter(author_email="test@example.com"),
)

chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi")
