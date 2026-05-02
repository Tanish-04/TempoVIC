from typing import List
from clang import cindex

class CParserVisitor:
    def __init__(self, line_reader, tree_context):
        self.context = tree_context
        self.stack: List = []
        self.reader = line_reader

    def build_tree(self, translation_unit: cindex.TranslationUnit):
        root = self._push("TRANSLATION_UNIT", "")
        for child in translation_unit.cursor.get_children():
            if self._is_interesting(child):
                self._visit_preorder(child)
        self._pop()
        self.context.set_root(root)
        return self.context

    def _visit_preorder(self, cursor: cindex.Cursor):
        self._process(cursor)
        children = [c for c in cursor.get_children() if self._is_interesting(c)]
        for ch in children:
            self._visit_preorder(ch)
        self._pop()

    def _process(self, cursor: cindex.Cursor):
        type_name = cursor.kind.name
        label = self._label_for(cursor)
        start, length = self._span_for(cursor)
        t = self._push(type_name, label)
        if start is not None and length is not None:
            t.set_pos(start)
            t.set_length(length)

    def _push(self, type_name: str, label: str):
        t = self.context.create_tree(type_name, label)
        if not self.stack:
            self.context.set_root(t)
        else:
            self.stack[-1].add_child(t)
        self.stack.append(t)
        return t

    def _pop(self):
        if self.stack:
            self.stack.pop()

    def _is_interesting(self, cur: cindex.Cursor) -> bool:
        if cur is None or cur.location is None or cur.location.file is None:
            return False
        return True

    def _label_for(self, cur: cindex.Cursor) -> str:
        if cur.spelling:
            return cur.spelling
        if cur.displayname:
            return cur.displayname
        try:
            toks = list(cur.get_tokens())
            if toks:
                return "".join(t.spelling for t in toks)
        except Exception:
            pass
        return ""

    def _span_for(self, cur: cindex.Cursor):
        try:
            extent = cur.extent
            start, end = extent.start, extent.end
            if not start.file or not end.file:
                return None, None
            s_off = self.reader.position_for(start.line, start.column)
            e_off = self.reader.position_for(end.line, end.column) - 1
            if e_off < s_off:
                e_off = s_off
            return s_off, e_off - s_off + 1
        except Exception:
            return None, None

