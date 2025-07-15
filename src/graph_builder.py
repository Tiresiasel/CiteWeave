# neo4j_graph_operations.py

from neo4j import GraphDatabase
from typing import List, Dict, Optional

class GraphDB:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_paper(self, paper_id: str, title: str, authors: List[str], year: int, stub: bool = False):
        query = """
        MERGE (p:Paper {id: $paper_id})
        SET p.title = $title, p.authors = $authors, p.year = $year, p.stub = $stub
        """
        with self.driver.session() as session:
            session.run(query, paper_id=paper_id, title=title, authors=authors, year=year, stub=stub)

    def create_argument(self, arg_id: str, paper_id: str, text: str, claim_type: str,
                        section: Optional[str] = None, version: str = "v1.0",
                        confidence: Optional[float] = None, custom_tags: Optional[List[str]] = None):
        query = """
        MERGE (a:Argument {id: $arg_id})
        SET a.text = $text, a.claim_type = $claim_type, a.section = $section,
            a.version = $version, a.confidence = $confidence, a.custom_tags = $custom_tags
        WITH a
        MATCH (p:Paper {id: $paper_id})
        MERGE (a)-[:BELONGS_TO]->(p)
        """
        with self.driver.session() as session:
            session.run(query, arg_id=arg_id, paper_id=paper_id, text=text, claim_type=claim_type,
                        section=section, version=version, confidence=confidence, custom_tags=custom_tags)

    def create_relation(self, from_arg: str, to_arg_or_paper: str, relation_type: str,
                        confidence: Optional[float] = None, version: str = "v1.0"):
        query = """
        MATCH (a1:Argument {id: $from_arg})
        OPTIONAL MATCH (a2:Argument {id: $to_arg})
        OPTIONAL MATCH (p:Paper {id: $to_arg})
        WITH a1, coalesce(a2, p) as target
        WHERE target IS NOT NULL
        MERGE (a1)-[r:RELATES {relation_type: $relation_type}]->(target)
        SET r.confidence = $confidence, r.version = $version
        """
        with self.driver.session() as session:
            session.run(query, from_arg=from_arg, to_arg=to_arg_or_paper,
                        relation_type=relation_type, confidence=confidence, version=version)

    def create_stub_relation(self, from_arg: str, to_paper_id: str, relation_type: str,
                             confidence: Optional[float] = None, version: str = "v1.0"):
        """
        If the target paper is not in the graph yet, create a stub Paper node and link to it.
        """
        query = """
        MATCH (a1:Argument {id: $from_arg})
        MERGE (p:Paper {id: $to_paper_id})
        ON CREATE SET p.stub = true
        MERGE (a1)-[r:RELATES {relation_type: $relation_type}]->(p)
        SET r.confidence = $confidence, r.version = $version
        """
        with self.driver.session() as session:
            session.run(query, from_arg=from_arg, to_paper_id=to_paper_id,
                        relation_type=relation_type, confidence=confidence, version=version)

    def create_claim_schema_node(self, claim_id: str, label: str, description: str,
                                 is_deprecated: bool = False, is_default: bool = True):
        query = """
        MERGE (c:ClaimTypeSchema {id: $claim_id})
        SET c.label = $label, c.description = $description,
            c.is_deprecated = $is_deprecated, c.is_default = $is_default
        """
        with self.driver.session() as session:
            session.run(query, claim_id=claim_id, label=label,
                        description=description, is_deprecated=is_deprecated, is_default=is_default)

    def create_relation_schema_node(self, rel_id: str, label: str, description: str,
                                    is_deprecated: bool = False, is_default: bool = True):
        query = """
        MERGE (r:RelationTypeSchema {id: $rel_id})
        SET r.label = $label, r.description = $description,
            r.is_deprecated = $is_deprecated, r.is_default = $is_default
        """
        with self.driver.session() as session:
            session.run(query, rel_id=rel_id, label=label,
                        description=description, is_deprecated=is_deprecated, is_default=is_default)



if __name__ == "__main__":
    db =GraphDB(uri="bolt://localhost:7687", user="neo4j", password="12345678")
    db.create_paper(paper_id="1", title="Paper 1", authors=["Author 1"], year=2021)
    db.create_argument(arg_id="1", paper_id="1", text="Argument 1", claim_type="CLAIM_MAIN")
    db.create_relation(from_arg="1", to_arg_or_paper="2", relation_type="CITES")
    db.create_stub_relation(from_arg="1", to_paper_id="2", relation_type="CITES")
    # query the graph
    query = """
    MATCH (p:Paper)
    RETURN p
    """
    with db.driver.session() as session:
        result = session.run(query)
        for record in result:
            print(record)
    db.close()