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
    UserStory = "UserStory"
    SystemComponent = "SystemComponent"
    Service = "Service"
    API = "API"
    TestCase = "TestCase"
    Concept = "Concept"
    Technology = "Technology"
    Methodology = "Methodology"

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

class EntityList(BaseModel):
    entities: list[Entity]

class Neo4jImporter:
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

    def upload_doc_triples(self, triple_list: TripleList, source_file: str, doc_type: str, group: str) -> bool:
        try:
            with self.driver.session() as session:
                labels = ["Requirement", "Service", "SystemComponent", "API", "TestCase", "Actor", "General", "Technology", "Methodology", "Concept"]
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

                    reference = item["subject"]["props_dict"].get("req_reference")
                    if reference:
                        reference_list = [r.strip() for r in reference.split(',')]
                        item["subject"]["props_dict"]["req_reference"] = reference_list
                    
                    o_kv = item["object"].get("properties") or []
                    item["object"]["props_dict"] = {kv["key"]: kv["value"] for kv in o_kv} if o_kv else {}

                    reference = item["object"]["props_dict"].get("req_reference")
                    if reference:
                        reference_list = [r.strip() for r in reference.split(',')]
                        item["object"]["props_dict"]["req_reference"] = reference_list

                    data_to_upload.append(item)

                cypher_query = """
                UNWIND $batch AS row
                CALL (row) {
                    WITH row
                    CALL apoc.merge.node([row.subject.label], {name: row.subject.name, group: $group}, {}, {}) YIELD node AS sNode
                    SET sNode += row.subject.props_dict
                    SET sNode.source_files = apoc.coll.toSet(coalesce(sNode.source_files, []) + $source_file)
                    SET sNode.doc_type = apoc.coll.toSet(coalesce(sNode.doc_type, []) + $doc_type)

                    WITH sNode, row
                    CALL apoc.merge.node([row.object.label], {name: row.object.name, group: $group}, {}, {}) YIELD node AS oNode
                    SET oNode += row.object.props_dict
                    SET oNode.source_files = apoc.coll.toSet(coalesce(oNode.source_files, []) + $source_file)
                    SET oNode.doc_type = apoc.coll.toSet(coalesce(oNode.doc_type, []) + $doc_type)

                    WITH sNode, oNode, row
                    CALL apoc.merge.relationship(sNode, row.relation.name, {}, {}, oNode) YIELD rel
                    SET rel.description = row.relation.description
                    SET rel.source_files = apoc.coll.toSet(coalesce(rel.source_files, []) + $source_file)
                    RETURN count(rel) AS relCount
                } IN TRANSACTIONS
                RETURN sum(relCount)
                """
                session.run(cypher_query, batch=data_to_upload, source_file=source_file, group=group, doc_type=doc_type)

                # print(f"成功上傳 {len(triple_list.triples)} 條關係。")
                return True
        
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False

    def upload_entities(self, entity_list: EntityList, source_file: str, doc_type: str, group: str) -> bool:
        try:
            with self.driver.session() as session:
                labels = ["Requirement", "Service", "SystemComponent", "API", "TestCase", "Actor", "General", "Technology", "Methodology", "Concept"]
                for label in labels:
                    session.run(f"""
                    CREATE CONSTRAINT {label.lower()}_group_name_unique IF NOT EXISTS 
                    FOR (n:{label}) 
                    REQUIRE (n.name, n.group) IS UNIQUE
                    """)
                
                # data_to_upload = [t.model_dump() for entities in entity_list for t in entities.entities]
                data_to_upload = list()
                for t in entity_list.entities:
                    item = t.model_dump()
                    # 強制轉換 Enum 為字串
                    item["label"] = str(t.label.value)

                    item["props_dict"] = self.convert_properties_to_dict(item.get("properties") or [])

                    # reference = item["props_dict"].get("req_reference")
                    # if reference:
                    #     reference_list = [r.strip() for r in reference.split(',')]
                    #     item["props_dict"]["req_reference"] = reference_list

                    data_to_upload.append(item)

                cypher_query = """
                UNWIND $batch AS row
                CALL (row) {
                    WITH row
                    CALL apoc.merge.node([row.label], {name: row.name, group: $group}, {}, {}) YIELD node AS node
                    SET node += row.props_dict
                    SET node.source_files = apoc.coll.toSet(coalesce(node.source_files, []) + $source_file)
                    SET node.doc_type = apoc.coll.toSet(coalesce(node.doc_type, []) + $doc_type)

                    RETURN count(node) AS nCount
                } IN TRANSACTIONS
                RETURN nCount
                """
                session.run(cypher_query, batch=data_to_upload, source_file=source_file, group=group, doc_type=doc_type)

                # print(f"成功上傳 {len(entity_list.triples)} 條關係。")
                return True
        
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False
        
    def convert_properties_to_dict(self, properties_list):
        """把 properties 轉成 dict，同 key 多值時變成 list"""
        props_dict = {}
        for kv in properties_list:
            key = kv["key"]
            value = kv["value"]
            
            if key in props_dict:
                # 已存在同 key，轉成 list 或加入 list
                if not isinstance(props_dict[key], list):
                    props_dict[key] = [props_dict[key]]
                props_dict[key].append(value)
            else:
                props_dict[key] = value
        
        # 最後確保 req_reference 一律是 list（即使只有一個值）
        if "req_reference" in props_dict:
            if not isinstance(props_dict["req_reference"], list):
                props_dict["req_reference"] = [props_dict["req_reference"]]
        
        return props_dict

    def link_references_to_requirements(self, label: str, doc_type: str, group: str, rel_type: str) -> bool:
        cypher = f"""
        MATCH (n:{label})
        WHERE n.group = $group AND $doc_type in n.doc_type AND n.req_reference IS NOT NULL
        UNWIND n.req_reference AS req_id
        MATCH (r:Requirement {{req_id: req_id, group: $group}})
        MERGE (n)-[rel:{rel_type}]->(r)
        RETURN count(rel) AS linked
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, doc_type=doc_type, group = group).single()
                logger.info(f"Linked {result['linked']} references")
                return True
        except Exception as e:
            logger.error(f"Fail to upload to Neo4j: {e}")
            return False

    def run_cypher(self, cypher_query: str) -> bool:
        try:
            with self.driver.session() as session:
                session.run(cypher_query)
                return True     
        except Exception as e:
                logger.error(f"Fail to query from Neo4j: {e}")
                return False