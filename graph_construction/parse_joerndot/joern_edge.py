class JoernEdge:
    
    def __init__(self, type_: str, content_: str, src_: str, dst_: str):
        # EXACT Java field names  
        self.edge_type = type_     
        self.content = content_  
        self.src = src_        
        self.dst = dst_        

    def __str__(self) -> str:
        return f"({self.type},{self.content},{self.src},{self.dst})"

