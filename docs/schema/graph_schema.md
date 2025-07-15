# Graph Schema for Argument Graph Project

This document describes the node and edge (relationship) field specifications for the Neo4j database used in the argument graph project.

## Node Types

### Claim
- `id`: Unique identifier for the claim (string)
- `text`: The content of the claim (string)
- `type`: Claim type (Fact, Value, Policy, etc.)
- `source`: Source document or reference (string)
- `metadata`: Additional metadata (JSON or string)

### Citation
- `id`: Unique identifier for the citation (string)
- `text`: The cited text or reference (string)
- `source`: Source document (string)
- `metadata`: Additional metadata (JSON or string)

## Relationship Types

### Supports
- `from`: Source claim id
- `to`: Target claim id
- `confidence`: Confidence score (float, optional)

### Attacks
- `from`: Source claim id
- `to`: Target claim id
- `confidence`: Confidence score (float, optional)

### Cites
- `from`: Claim or citation id
- `to`: Cited claim or citation id
- `context`: Context of the citation (string, optional)

> **Note:** Extend these fields as needed for your use case. 