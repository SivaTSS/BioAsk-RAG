# src/routing/logical_routing.py

class LogicalRouter:
    def __init__(self):
        # Define keyword sets for each DB type
        self.relational_keywords = {"database", "sql", "table", "record", "schema", "query"}
        self.graph_keywords = {"graph", "relationship", "node", "edge", "neo4j", "cypher"}

    def route(self, question):
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in self.relational_keywords):
            return "RelationalDB"
        elif any(keyword in question_lower for keyword in self.graph_keywords):
            return "GraphDB"
        else:
            return "VectorDB"
