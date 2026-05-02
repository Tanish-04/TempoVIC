from typing import List

class JoernNode:
    
    def __init__(self, id_: str, type_: str, code_: str, lineNum_: int):

        self.node_id = id_
        self.type = type_
        self.code = code_
        self.line_num = lineNum_
        self.edges = []

    def __str__(self) -> str:
        return f"({self.node_id},{self.type},{self.code},{self.line_num})"

