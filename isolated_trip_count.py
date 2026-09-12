import tree_sitter_c
from tree_sitter import Language, Parser
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from code_parser import _parse_for_loop

code_test1 = b"""
void test_non_unit_step() {
    double A[1024];
    for (int i = 0; i < 1024; i += 4) {
        A[i] = A[i] * 2.0;
    }
}
"""

code_test2 = b"""
void test_decrementing() {
    double A[1024];
    for (int i = 1023; i >= 0; i--) {
        A[i] = A[i] * 2.0;
    }
}
"""

def parse_and_estimate(source):
    lang = Language(tree_sitter_c.language())
    parser = Parser(lang)
    tree = parser.parse(source)
    
    def find_for_stmt(node):
        if node.type == 'for_statement':
            return node
        for child in node.children:
            res = find_for_stmt(child)
            if res:
                return res
        return None
        
    for_node = find_for_stmt(tree.root_node)
    info = _parse_for_loop(for_node, source, 0, None)
    print(f"Isolated trip count: {info.trip_count} (fallback applies if None or <= 0)")

print("Test 1 (i += 4):")
parse_and_estimate(code_test1)
print("Test 2 (i--):")
parse_and_estimate(code_test2)
