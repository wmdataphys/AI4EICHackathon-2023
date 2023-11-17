from langchain.memory.chat_message_histories import SQLChatMessageHistory


from datetime import datetime
from typing import Any

from langchain.memory.chat_message_histories.sql import BaseMessageConverter
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import Column, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CustomMessage(Base):
    '''
    This is a custom message class that is used to store the messages in a database.
    '''
    def __init__(self, user_name, session_id, useful):
        super().__init__(CustomMessage)
        self.user_name = user_name
        self.session_id = session_id
        self.Useful = self.useful
    def _getuname(self):
        return self.user_name
    __tablename__ = _getuname()

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer)
    type = Column(Text)
    content = Column(Text)
    created_at = Column(DateTime)
    Useful = Column(Boolean)


class CustomMessageConverter(BaseMessageConverter):
    '''
    instructions on how to convert them into the schema.
    '''
    def __init__(self, username: str, session_id: int):
        self.username = username
        self.sessoin_id = session_id

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
            user_name = self.username,
            session_id=self.session_id,
            type=message.type,
            content=message.content,
            created_at=now,
            Useful=True if message.type == "ai" else False, # To Do this should come from the user.
        )

    def get_sql_model_class(self) -> Any:
        return CustomMessage