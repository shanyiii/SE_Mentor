# Based on code from [https://github.com/stair-lab/kg-gen]

from typing import Optional, List
from neo4j import GraphDatabase, Driver, Result
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EntityLabel(str, Enum):
    Actor = "Actor"
    Requirement = "Requirement"
    SystemComponent = "SystemComponent"
    API = "API"
    TestCase = "TestCase"
    Concept = "Concept"
    Technology = "Technology"
    Methodology = "Methodology"
    General = "General"

class KeyValPair(BaseModel):
    key: str = Field(..., description="屬性名稱")
    value: str= Field(..., description="屬性值")

class Entity(BaseModel):
    name: str = Field(..., description="實體的唯一名稱") # name 為必填
    label: EntityLabel
    properties: Optional[List[KeyValPair]] = Field(default_factory=list, description="實體的屬性資訊")

class Relation(BaseModel):
    name: str = Field(..., description="實體之間的邏輯關係")
    description: str

class Triple(BaseModel):
    subject: Entity
    relation: Relation
    object: Entity

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

    def upload_textbook_triples(self, triple_list: TripleList, source_file: str) -> bool:
        try:
            with self.driver.session() as session:
                labels = ["Concept", "Technology", "Methodology"]
                for label in labels:
                    session.run(f"""
                    CREATE CONSTRAINT {label.lower()}_group_name_unique IF NOT EXISTS 
                    FOR (n:{label}) 
                    REQUIRE (n.name, n.group) IS UNIQUE
                    """)
                # session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

                # data_to_upload = [t.model_dump() for t in triple_list.triples]
                data_to_upload = list()
                for t in triple_list.triples:
                    item = t.model_dump()
                    # 強制轉換 Enum 為字串
                    item["subject"]["label"] = str(t.subject.label.value)
                    item["object"]["label"] = str(t.object.label.value)

                    s_kv = item["subject"].get("properties") or []
                    item["subject"]["props_dict"] = {kv["key"]: kv["value"] for kv in s_kv} if s_kv else {}
                    
                    o_kv = item["object"].get("properties") or []
                    item["object"]["props_dict"] = {kv["key"]: kv["value"] for kv in o_kv} if o_kv else {}

                    data_to_upload.append(item)
                
                # print(data_to_upload[0])
                cypher_query = """
                UNWIND $batch AS row
                CALL apoc.merge.node([row.subject.label], {name: row.subject.name}, {}, {}) YIELD node AS sNode
                WITH sNode, row
                SET sNode += row.subject.props_dict
                SET sNode.source_files = apoc.coll.toSet(coalesce(sNode.source_files, []) + $source_file)

                WITH sNode, row
                CALL apoc.merge.node([row.object.label], {name: row.object.name}, {}, {}) YIELD node AS oNode
                WITH sNode, oNode, row
                SET oNode += row.object.props_dict
                SET oNode.source_files = apoc.coll.toSet(coalesce(oNode.source_files, []) + $source_file)

                WITH sNode, oNode, row
                CALL apoc.merge.relationship(sNode, row.relation.name, {}, {}, oNode) YIELD rel
                SET rel.description = row.relation.description
                SET rel.source_files = apoc.coll.toSet(coalesce(rel.source_files, []) + $source_file)
                RETURN count(rel)
                """
                session.run(cypher_query, batch=data_to_upload, source_file=source_file)

                logger.info(f"成功上傳 {len(triple_list.triples)} 條關係。")
                return True
        
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False

    def upload_doc_triples(self, triple_list: TripleList, source_file: str, group: str) -> bool:
        try:
            with self.driver.session() as session:
                labels = ["Requirement", "SystemComponent", "API", "TestCase", "Actor", "General"]
                for label in labels:
                    session.run(f"""
                    CREATE CONSTRAINT {label.lower()}_group_name_unique IF NOT EXISTS 
                    FOR (n:{label}) 
                    REQUIRE (n.name, n.group) IS UNIQUE
                    """)
                
                # data_to_upload = [t.model_dump() for triples in triple_list for t in triples.triples]
                data_to_upload = list()
                for t in triple_list.triples:
                    item = t.model_dump()
                    # 強制轉換 Enum 為字串
                    item["subject"]["label"] = str(t.subject.label.value)
                    item["object"]["label"] = str(t.object.label.value)

                    s_kv = item["subject"].get("properties") or []
                    item["subject"]["props_dict"] = {kv["key"]: kv["value"] for kv in s_kv} if s_kv else {}
                    
                    o_kv = item["object"].get("properties") or []
                    item["object"]["props_dict"] = {kv["key"]: kv["value"] for kv in o_kv} if o_kv else {}

                    data_to_upload.append(item)

                cypher_query = """
                UNWIND $batch AS row
                CALL apoc.merge.node([row.subject.label], {name: row.subject.name, group: $group}, {}, {}) YIELD node AS sNode
                WITH sNode, row
                SET sNode += row.subject.props_dict
                SET sNode.source_files = apoc.coll.toSet(coalesce(sNode.source_files, []) + $source_file)

                WITH sNode, row
                CALL apoc.merge.node([row.object.label], {name: row.object.name, group: $group}, {}, {}) YIELD node AS oNode
                WITH sNode, oNode, row
                SET oNode += row.object.props_dict
                SET oNode.source_files = apoc.coll.toSet(coalesce(oNode.source_files, []) + $source_file)

                WITH sNode, oNode, row
                CALL apoc.create.relationship(sNode, row.relation, {source_file: $source_file}, oNode) YIELD rel
                RETURN count(rel)
                """
                result = session.execute_write(lambda tx: tx.run(cypher_query, batch=data_to_upload, source_file=source_file, group=group).single())

                # print(f"成功上傳 {len(triple_list.triples)} 條關係。")
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