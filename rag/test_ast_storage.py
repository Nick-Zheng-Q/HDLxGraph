import pyverilog
from pyverilog.vparser.parser import parse
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator

# Function to parse Verilog file
def parse_verilog(file_path):
    print(f"Try parsing {file_path}")
    ast, directives = parse([file_path])
    return ast

def find_blocks_and_signals(ast):
    # 存储所有代码块
    all_blocks = {}  # key: block_id, value: (block_type, code)
    # 存储信号与块的关系
    signal_blocks = {}  # key: signal_name, value: [(block_type, block_id), ...]
    codegen = ASTCodeGenerator()

    def traverse(node, current_block):
        if hasattr(node, 'children'):
            for child in node.children():
                traverse(child, current_block)
        if hasattr(node, 'name'):
            if node.name not in signal_blocks:
                signal_blocks[node.name] = []
            # 只存储类型和ID的关系
            block_ref = (current_block[0], current_block[1])  # type和id
            if block_ref not in signal_blocks[node.name]:
                signal_blocks[node.name].append(block_ref)

    def traverse_blocks(node):
        if hasattr(node, 'children'):
            for child in node.children():
                if isinstance(child, (pyverilog.vparser.ast.Assign,
                                   pyverilog.vparser.ast.Always,
                                   pyverilog.vparser.ast.Initial,
                                   pyverilog.vparser.ast.Instance,
                                   pyverilog.vparser.ast.Function,
                                   pyverilog.vparser.ast.Task,
                                   pyverilog.vparser.ast.Decl,
                                   pyverilog.vparser.ast.Paramlist,
                                   pyverilog.vparser.ast.Portlist)):
                    block_id = id(child)
                    block_type = child.__class__.__name__
                    block_code = codegen.visit(child)

                    # 存储块信息
                    all_blocks[block_id] = (block_type, block_code)

                    # 传递block信息用于信号关联
                    block_info = (block_type, block_id, block_code)
                    traverse(child, block_info)
                else:
                    traverse_blocks(child)

    traverse_blocks(ast)
    return all_blocks, signal_blocks

# Main function
def blocks_and_signals(file_path):
    ast = parse_verilog(file_path)
    all_blocks, signal_blocks = find_blocks_and_signals(ast)

    # 打印所有代码块
    print("=== All Blocks ===")
    for block_id, (block_type, code) in all_blocks.items():
        print(f"\nBlock ID: {block_id}")
        print(f"Type: {block_type}")
        print("Code:")
        print(f"    {code.replace(chr(10), chr(10)+'    ')}")

    # 打印信号关系
    print("\n=== Signal Relationships ===")
    for signal, blocks in signal_blocks.items():
        print(f"\nSignal: {signal}")
        for block_type, block_id in blocks:
            print(f"  In Block: {block_type} (ID: {block_id})")

    return all_blocks, signal_blocks

