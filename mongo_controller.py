import pymongo
from pymongo import AsyncMongoClient
from beanie import Document, init_beanie

class DiagnosisQuiz(Document):
  question: str
  options: list[str]
  answer: int
  analysis: str
  concept: str
  chapter: str

  class Settings:
    name = "diagnosis_quiz"

async def init_mongo(database_name):
  client = AsyncMongoClient("mongodb://localhost:27017/")
  database = client.get_database(database_name) 
  await init_beanie(database=database, document_models=[DiagnosisQuiz])
  # collection = client["<collection name>"]
  return client
