
from typing import List

class GetPos:
    def __init__(self, fcontent: str):
        self.line_begs: List[int] = []
        lines = fcontent.split("\n")
        p1 = p2 = 0
        for line in lines:
            p1 = p2
            self.line_begs.append(p1)
            p2 = p1 + len(line) + 1 

    @staticmethod
    def upperbound(vector: List[int], val: int) -> int:
        l, u = 0, len(vector) - 1
        while l < u:
            midp = l + (u - l) // 2
            mid = vector[midp]
            if val < mid:
                u = midp
            else:
                l = midp + 1
        return l if val < vector[l] else len(vector)

    def get_line_num(self, offset: int) -> int:
        return self.upperbound(self.line_begs, offset)

