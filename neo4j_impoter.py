# Based on code from [https://github.com/stair-lab/kg-gen]

from typing import Optional
from neo4j import GraphDatabase, Driver, Result
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class Triple(BaseModel):
    subject: str
    relation: str
    object: str
    source_file: str

class TripleList(BaseModel):
    triples: list[Triple]

class Neo4jImoprter:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
    
    def connect(self) -> bool:
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
        
    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def upload_triples(self, triple_list: TripleList) -> bool:
        try:
            with self.driver.session() as session:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

                data_to_upload = [t.model_dump() for t in triple_list.triples]
                
                cypher_query = """
                UNWIND $batch AS row
                MERGE (s:Entity {name: row.subject})
                MERGE (o:Entity {name: row.object})
                SET s.source_file = apoc.coll.toSet(coalesce(s.source_files, []) + row.source_file)
                SET o.source_file = apoc.coll.toSet(coalesce(o.source_files, []) + row.source_file)
                WITH s, o, row
                CALL apoc.create.relationship(s, row.relation, {}, o) YIELD rel
                RETURN count(rel)
                """
                session.run(cypher_query, batch=data_to_upload)

                logger.info(f"成功上傳 {len(triple_list.triples)} 條關係。")
                return True
        
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False
            
    def query_retrival(self, cypher_query: str) -> Result:
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record["name"] for record in result]
                return records       
        except Exception as e:
                logger.error(f"Fail to query from Neo4j: {e}")
                return False