// ELVARN knowledge graph schema (Neo4j 5, Community edition). Mirrors ontology.ttl.
// One statement per line, run by KnowledgeBase at start-up. All are idempotent (IF NOT EXISTS).
//
// Graph shape:
//   (:Source)<-[:FROM]-(:Passage)                               every text chunk belongs to a cited source
//   (:Entity:Person|Case|Court|...)                             people, cases, agencies, laws, places
//   (:CaseParty)-[:PERSON]->(:Entity), -[:IN_CASE]->(:Entity)   a person's role/status in a case
//   (:CaseEvent)-[:OF_CASE]->(:Entity)                          FIR, arrest, chargesheet, trial, judgment ...
//   (:Claim)-[:SUBJECT]->(:Entity), -[:OBJECT]->(:Entity)       any other fact
//   every CaseParty / CaseEvent / Claim has exactly one -[:SUPPORTED_BY]->(:Passage)
//   (:Candidate)-[:FROM_PASSAGE]->(:Passage)                    proposed facts waiting for human review
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE
CREATE CONSTRAINT source_sha IF NOT EXISTS FOR (n:Source) REQUIRE n.sha256 IS UNIQUE
CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (n:Passage) REQUIRE n.id IS UNIQUE
CREATE CONSTRAINT passage_position IF NOT EXISTS FOR (n:Passage) REQUIRE (n.source_id, n.position) IS UNIQUE
CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.key IS UNIQUE
CREATE CONSTRAINT party_id IF NOT EXISTS FOR (n:CaseParty) REQUIRE n.id IS UNIQUE
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:CaseEvent) REQUIRE n.id IS UNIQUE
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE
CREATE CONSTRAINT candidate_uid IF NOT EXISTS FOR (n:Candidate) REQUIRE n.uid IS UNIQUE
CREATE FULLTEXT INDEX passage_text IF NOT EXISTS FOR (n:Passage) ON EACH [n.text]
CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)
CREATE INDEX candidate_state IF NOT EXISTS FOR (n:Candidate) ON (n.state)
// Processing pipeline (additive): Source nodes are also :Document; (:Source)-[:HAS_PASSAGE]->(:Passage); (:Passage)-[:MENTIONS]->(:Entity);
// (:Entity)-[:RELATION {type, method, confidence, doc_id, passage_id, assertion_level, start, end}]->(:Entity)
CREATE CONSTRAINT document_doc_id IF NOT EXISTS FOR (n:Document) REQUIRE n.doc_id IS UNIQUE
CREATE INDEX passage_doc IF NOT EXISTS FOR (n:Passage) ON (n.doc_id)
CREATE INDEX entity_person_key IF NOT EXISTS FOR (n:Entity) ON (n.person_key)
CREATE INDEX entity_wikidata IF NOT EXISTS FOR (n:Entity) ON (n.wikidata_id)
CREATE INDEX relation_key IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.rel_key)
CREATE INDEX source_url IF NOT EXISTS FOR (n:Source) ON (n.url)
