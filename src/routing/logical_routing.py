# src/routing/logical_routing.py

class LogicalRouter:
    def __init__(self):
        pass

    def route(self, question):
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ["database", "sql", "table", "records"]):
            return "RelationalDB"
        elif any(keyword in question_lower for keyword in ["graph", "relationship", "node", "edge", "neo4j"]):
            return "GraphDB"
        else:
            return "VectorDB"
