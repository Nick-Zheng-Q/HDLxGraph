import py2neo
from py2neo import Graph, Node, Relationship
import os  # 添加到文件顶部的导入部分
import sys
import pygraphviz as pgv
import pyverilog
from pyverilog.vparser.parser import parse
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pyverilog.dataflow.graphgen import VerilogGraphGenerator

# Neo4j连接配置
uri = "bolt://localhost:7687"
user = "neo4j"
password = "neo4j"

def parse_verilog(file_path):
    ast, directives = parse([file_path])
    return ast

def store_verilog_graph(file_path):
    # 连接Neo4j
    graph = Graph(uri, auth=(user, password))
    graph.delete_all()
    
    # 解析Verilog文件
    ast = parse_verilog(file_path)
    all_blocks, signal_blocks = find_blocks_and_signals(ast)
    dataflow_info = analyze_with_dataflow(file_path, signal_blocks)
    
    # 读取整个文件内容
    with open(file_path, 'r') as f:
        module_code = f.read()
    
    # 创建模块节点
    top_module = get_top_module_name(ast)
    module_node = Node("Module",
                      name=top_module,
                      code=module_code,
                      file_path=file_path)
    graph.create(module_node)
    
    # 存储代码块节点
    block_nodes = {}
    for block_id, (block_type, block_code) in all_blocks.items():
        if block_code:  # 确保代码块不为空
            block_node = Node("Block",
                            id=str(block_id),
                            type=block_type,
                            code=block_code)
            graph.create(block_node)
            block_nodes[block_id] = block_node
            # 创建模块与代码块的关系
            graph.create(Relationship(module_node, "CONTAINS", block_node))
    
    # 存储信号节点和关系
    signal_nodes = {}
    # 存储所有出现的信号（包括中间变量）
    all_signals = set()
    
    # 首先收集所有信号名称
    for signal_name, info in dataflow_info.items():
        if signal_name and isinstance(signal_name, str):
            all_signals.add(signal_name)
            # 收集数据流中的所有相关信号
            for flow in info.get('dataflow', []):
                if 'all_signals' in flow:
                    for src_signal in flow['all_signals']:
                        src_name = str(src_signal).split('.')[-1]
                        if src_name:
                            all_signals.add(src_name)
    
    # 创建所有信号节点
    for signal_name in all_signals:
        signal_node = Node("Signal", name=signal_name)
        graph.create(signal_node)
        signal_nodes[signal_name] = signal_node
    
    # 创建信号与代码块的关系
    for signal_name, info in dataflow_info.items():
        if signal_name in signal_nodes:
            # 创建与代码块的包含关系
            for block_type, block_id in info.get('blocks', []):
                if block_id in block_nodes:
                    graph.create(Relationship(block_nodes[block_id], "CONTAINS", signal_nodes[signal_name]))

            # 处理数据流关系
            for flow in info.get('dataflow', []):
                if 'all_signals' in flow:
                    for src_signal in flow['all_signals']:
                        src_name = str(src_signal).split('.')[-1]
                        if src_name in signal_nodes and src_name != signal_name:
                            # 创建数据流关系
                            flow_type = flow.get('always_type', 'combinational')
                            rel_props = {
                                'type': flow_type,
                                'assignment': flow.get('assignment_type', 'unknown')
                            }
                            
                            # 添加时序信息
                            if flow_type == 'clockedge':
                                rel_props.update({
                                    'clock_edge': flow.get('clock_edge'),
                                    'clock_name': flow.get('clock_name')
                                })
                            
                            # 创建关系
                            graph.create(Relationship(
                                signal_nodes[src_name], 
                                "FLOWS_TO", 
                                signal_nodes[signal_name],
                                **rel_props
                            ))

    # 获取数据流分析的基本信息
    ast = parse_verilog(file_path)
    top_module = get_top_module_name(ast)
    analyzer = VerilogDataflowAnalyzer(
        [file_path],
        top_module,
        [],
        [],
        file_path
    )
    analyzer.generate()
    terms = analyzer.getTerms()  # 获取terms
    binddict = analyzer.getBinddict()

    # 存储 DFG 节点和边
    dfg = analyze_dataflow(file_path, "top")
    dfg_nodes = {}  # 用于存储节点引用
    
    # 先创建所有 DFG 节点
    for node in dfg.nodes():
        node_name = str(node)  # 确保节点名称是字符串
        
        # 使用 find_matching_term 查找匹配的信号
        matching_term = find_matching_term(node_name, terms)
        if matching_term:
            term_name = str(matching_term).split('.')[-1]
            if term_name in signal_nodes:
                print(f"找到匹配的信号节点: {node_name} -> {term_name}")
                dfg_nodes[node_name] = signal_nodes[term_name]
                continue
        
        # 如果没有找到匹配的信号节点，检查是否是操作符或中间结果
        is_operator = any(op in node_name for op in ['Plus', 'Minus', 'And', 'Or', 'Xor', 'Eq'])
        if is_operator:
            print(f"创建操作符节点: {node_name}")
            dfg_node = Node("Operator", name=node_name, type='operator')
        else:
            print(f"创建中间结果节点: {node_name}")
            dfg_node = Node("DFGNode", name=node_name, type='intermediate')
        
        graph.create(dfg_node)
        dfg_nodes[node_name] = dfg_node
    
    # 创建边时使用存储的节点引用
    for edge in dfg.edges():
        source_name = str(edge[0])
        target_name = str(edge[1])
        # edge_type = str(edge.attr)
        if source_name in dfg_nodes and target_name in dfg_nodes:
            source_node = dfg_nodes[source_name]
            target_node = dfg_nodes[target_name]
            
            # 根据节点类型创建不同的关系
            if 'Signal' in source_node.labels and 'Signal' in target_node.labels:
                # 信号节点间的数据流关系
                graph.create(Relationship(source_node, "FLOWS_TO", target_node))
            elif 'Operator' in source_node.labels:
                # 操作符到目标的计算关系
                graph.create(Relationship(source_node, "COMPUTES", target_node))
            else:
                # 其他情况使用一般连接关系
                graph.create(Relationship(source_node, "CONNECTS", target_node))

    print("数据已成功存入Neo4j数据库")

    # 验证数据
    print("\n验证结果:")
    print("Modules数量:", graph.run("MATCH (m:Module) RETURN count(m)").evaluate())
    print("Signals数量:", graph.run("MATCH (s:Signal) RETURN count(s)").evaluate())
    print("Blocks数量:", graph.run("MATCH (b:Block) RETURN count(b)").evaluate())
    print("DFGNodes数量:", graph.run("MATCH (d:DFGNode) RETURN count(d)").evaluate())
    print("CONTAINS关系数量:", graph.run("MATCH ()-[r:CONTAINS]->() RETURN count(r)").evaluate())
    print("FLOWS_TO关系数量:", graph.run("MATCH ()-[r:FLOWS_TO]->() RETURN count(r)").evaluate())
    print("CONNECTS关系数量:", graph.run("MATCH ()-[r:CONNECTS]->() RETURN count(r)").evaluate())

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

def create_signal_map(ast):
    """创建信号名称映射表"""
    signal_map = {}  # key: dataflow_name, value: original_name
    
    def traverse_signals(node):
        if hasattr(node, 'name'):
            # dataflow中的名称通常是 "module.signal" 格式
            if hasattr(node, 'scope'):
                dataflow_name = f"{node.scope}_{node.name}"
                signal_map[dataflow_name] = node.name
            else:
                signal_map[node.name] = node.name
        if hasattr(node, 'children'):
            for child in node.children():
                traverse_signals(child)
    
    traverse_signals(ast)
    return signal_map

def get_top_module_name(node):
    """获取顶层模块名称"""
    if isinstance(node, pyverilog.vparser.ast.Source):
        for child in node.children():
            if isinstance(child, pyverilog.vparser.ast.Description):
                for item in child.definitions:
                    if isinstance(item, pyverilog.vparser.ast.ModuleDef):
                        return item.name
    return None

def analyze_dataflow_tree(tree, flow_info):
    """递归分析表达式树"""
    try:
        # 添加树的基本信息
        flow_info['tree_type'] = tree.__class__.__name__

        # 对于PartSelect类型的节点
        if isinstance(tree, pyverilog.dataflow.dataflow.DFPartselect):
            flow_info['var'] = str(tree.var)
            flow_info['msb'] = str(tree.msb) if hasattr(tree, 'msb') else None
            flow_info['lsb'] = str(tree.lsb) if hasattr(tree, 'lsb') else None

        # 对于Branch类型的节点
        elif isinstance(tree, pyverilog.dataflow.dataflow.DFBranch):
            if hasattr(tree, 'condnode'):
                flow_info['condition'] = str(tree.condnode)
            if hasattr(tree, 'truenode'):
                flow_info['true_value'] = str(tree.truenode)
                # 递归分析true分支
                true_info = {}
                analyze_dataflow_tree(tree.truenode, true_info)
                flow_info['true_branch'] = true_info
            if hasattr(tree, 'falsenode'):
                flow_info['false_value'] = str(tree.falsenode)
                # 递归分析false分支
                false_info = {}
                analyze_dataflow_tree(tree.falsenode, false_info)
                flow_info['false_branch'] = false_info

        # 收集源信号
        srcs = []
        if hasattr(tree, 'children'):
            for child in tree.children():
                if hasattr(child, 'var'):
                    srcs.append(str(child.var))
                child_info = {}
                analyze_dataflow_tree(child, child_info)
                if 'src_signals' in child_info:
                    srcs.extend(child_info['src_signals'])
        flow_info['src_signals'] = list(set(srcs))  # 去重

        return flow_info
    except Exception as e:
        print(f"分析表达式树时出错: {str(e)}")
        return flow_info

def find_matching_term(signal_name, terms):
    """查找匹配的Term，支持多种命名格式"""
    base_name = signal_name.split('.')[-1]  # 获取基本名称
    if '_graphrename_' in base_name:
        base_name = base_name.split('_graphrename_')[0]
    variants = [
        signal_name,                    # 原始名称
        base_name,                      # 去除前缀
        f"top.{signal_name}",          # 添加top前缀
        f"top.{base_name}",            # 添加top前缀并去除前缀
        signal_name.split('.')[-1],    # 去除前缀
        signal_name.split('_')[-1],    # 去除前缀
        f"top_{signal_name}",          # 替换前缀分隔符
        f"top_{base_name}",            # 替换前缀分隔符并去除前缀
        f"top_{signal_name.split('.')[-1]}",  # 替换前缀分隔符并去除前缀
    ]
    
    # 打印调试信息
    print(f"  尝试匹配信号: {signal_name}")
    print(f"  检查变体: {variants}")
    print(f"  可用terms: {[str(k) for k in terms.keys()]}")  # 转换为字符串
    
    # 检查完整匹配
    for term_name in terms:
        term_str = str(term_name)  # 转换为字符串
        for variant in variants:
            if term_str == variant:
                print(f"  找到完整匹配: {term_str}")
                return term_name  # 返回原始的ScopeChain对象
            elif term_str.endswith('.' + variant):
                print(f"  找到部分匹配: {term_str} -> {variant}")
                return term_name  # 返回原始的ScopeChain对象
    
    print(f"  未找到匹配")
    return None

def analyze_with_dataflow(file_path, signal_blocks):
    try:
        ast = parse_verilog(file_path)
        
        # 获取顶层模块名
        top_module = get_top_module_name(ast)
        if not top_module:
            print("警告: 未找到顶层模块名，尝试从文件名获取...")
            top_module = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"使用顶层模块名: {top_module}")
        
        # 使用绝对路径
        abs_file_path = os.path.abspath(file_path)
        print(f"使用文件路径: {abs_file_path}")
        
        # 初始化数据流分析器
        analyzer = VerilogDataflowAnalyzer(
            [abs_file_path],
            top_module,
            [],
            [],
            abs_file_path
        )
        
        print("\n=== 开始数据流分析 ===")
        analyzer.generate()
        terms = analyzer.getTerms()
        binddict = analyzer.getBinddict()
        
        print("\n=== Terms信息 ===")
        print(f"Terms数量: {len(terms)}")
        for termname, term in terms.items():
            print(f"\nTerm名称: {termname}")
            print(f"Term类型: {term.__class__.__name__}")
            print(f"Term属性: {dir(term)}")
            
        print("\n=== Binddict信息 ===")
        print(f"Binddict数量: {len(binddict)}")
        for bindname, bind in binddict.items():
            print(f"\nBind名称: {bindname}")
            print(f"Bind类型: {bind.__class__.__name__}")
            print(f"Bind属性: {dir(bind)}")
            if hasattr(bind, 'tree'):
                print(f"Bind表达式: {bind.tree}")
            if hasattr(bind, 'getSrcSignals'):
                try:
                    srcs = bind.getSrcSignals()
                    print(f"源信号: {[str(s) for s in srcs]}")
                except Exception as e:
                    print(f"获取源信号出错: {str(e)}")
        
        print("\n=== Binddict详细信息 ===")
        for bindname, bind in binddict.items():
            print(f"\nBind名称: {bindname}")
            if isinstance(bind, list):
                print(f"绑定列表长度: {len(bind)}")
                for i, b in enumerate(bind):
                    print(f"\n  绑定项 {i}:")
                    print(f"  类型: {b.__class__.__name__}")
                    print(f"  属性: {dir(b)}")  # 添加属性列表
                    if hasattr(b, 'tree'):
                        print(f"  表达式树: {b.tree}")
                        print(f"  表达式树类型: {type(b.tree)}")  # 添加表达式树类型
                        print(f"  表达式树属性: {dir(b.tree)}")  # 添加表达式树属性
                    if hasattr(b, 'condition'):
                        print(f"  条件: {b.condition}")
                        print(f"  条件类型: {type(b.condition)}")  # 添加条件类型
                    if hasattr(b, 'right'):
                        print(f"  右值: {b.right}")
                        print(f"  右值类型: {type(b.right)}")  # 添加右值类型
                    if hasattr(b, 'left'):  # 添加左值检查
                        print(f"  左值: {b.left}")
                        print(f"  左值类型: {type(b.left)}")
                    if hasattr(b, 'dest'):  # 添加目标检查
                        print(f"  目标: {b.dest}")
                        print(f"  目标类型: {type(b.dest)}")
                    if hasattr(b, 'getSrcSignals'):
                        try:
                            srcs = b.getSrcSignals()
                            print(f"  源信号: {[str(s) for s in srcs]}")
                            print(f"  源信号类型: {[type(s) for s in srcs]}")  # 添加源信号类型
                        except Exception as e:
                            print(f"  获取源信号出错: {str(e)}")
        
        print("\n=== 信号匹配过程 ===")
        combined_info = {}
        if signal_blocks:
            # 首先收集所有数据流信息
            all_dataflows = {}
            
            def process_tree(tree, flow_info):
                """递归处理表达式树，收集所有相关信号"""
                signals = set()
                
                print(f"处理节点: {tree.__class__.__name__}")  # 调试输出
                
                # 处理None类型
                if tree is None:
                    return signals
                
                # 处理Terminal节点 (变量)
                if isinstance(tree, pyverilog.dataflow.dataflow.DFTerminal):
                    signal_name = str(tree.name) if hasattr(tree, 'name') else str(tree)
                    signals.add(signal_name)
                    print(f"找到变量: {signal_name}")  # 调试输出
                
                # 处理操作符
                elif isinstance(tree, pyverilog.dataflow.dataflow.DFOperator):
                    print(f"处理操作符: {tree.operator}")  # 调试输出
                    
                    # 处理所有操作数节点
                    if hasattr(tree, 'nextnodes'):
                        for node in tree.nextnodes:
                            if isinstance(node, pyverilog.dataflow.dataflow.DFTerminal):
                                signal_name = str(node)
                                signals.add(signal_name)
                                print(f"操作符信号: {signal_name}")
                            # 递归处理节点
                            node_signals = process_tree(node, flow_info)
                            signals.update(node_signals)
                            if node_signals:
                                print(f"子节点信号: {node_signals}")
                
                # 处理条件分支
                elif isinstance(tree, pyverilog.dataflow.dataflow.DFBranch):
                    print("处理条件分支")  # 调试输出
                    # 处理条件表达式
                    if hasattr(tree, 'condnode'):
                        cond_signals = process_tree(tree.condnode, flow_info)
                        signals.update(cond_signals)
                        if cond_signals:
                            print(f"条件信号: {cond_signals}")
                    
                    # 处理true分支
                    if hasattr(tree, 'truenode'):
                        true_signals = process_tree(tree.truenode, flow_info)
                        signals.update(true_signals)
                        if true_signals:
                            print(f"true分支信号: {true_signals}")
                    
                    # 处理false分支
                    if hasattr(tree, 'falsenode'):
                        false_signals = process_tree(tree.falsenode, flow_info)
                        signals.update(false_signals)
                        if false_signals:
                            print(f"false分支信号: {false_signals}")
                
                # 处理Part Select
                elif isinstance(tree, pyverilog.dataflow.dataflow.DFPartselect):
                    print("处理部分选择")  # 调试输出
                    # 处理变量部分
                    if hasattr(tree, 'var'):
                        var_signals = process_tree(tree.var, flow_info)
                        signals.update(var_signals)
                        print(f"部分选择变量: {var_signals}")
                
                # 处理其他类型的节点
                elif hasattr(tree, 'children'):
                    for child in tree.children():
                        child_signals = process_tree(child, flow_info)
                        signals.update(child_signals)
                
                print(f"当前节点收集的信号: {signals}")  # 调试输出
                return signals

            # 处理所有binddict条目
            for bind_name, bind in binddict.items():
                if isinstance(bind, list):
                    for b in bind:
                        try:
                            # 创建基本流信息
                            flow_info = {
                                'termname': str(bind_name),
                                'bind_type': b.__class__.__name__,
                                'assignment_type': 'blocking' if hasattr(b, '_assign') else 'non_blocking'
                            }
                            
                            # 处理表达式树
                            if hasattr(b, 'tree'):
                                tree = b.tree
                                flow_info['tree_type'] = tree.__class__.__name__
                                if hasattr(tree, 'tostr'):
                                    flow_info['tree_str'] = tree.tostr()
                                
                                # 收集所有相关信号
                                all_signals = process_tree(tree, flow_info)
                                flow_info['all_signals'] = list(all_signals)
                                
                                # 为所有相关信号添加数据流信息
                                for signal in all_signals:
                                    print(f"bind有: {signal}")
                                    signal_name = str(signal).split('.')[-1]
                                    if signal_name not in all_dataflows:
                                        all_dataflows[signal_name] = []
                                    all_dataflows[signal_name].append(flow_info)
                                
                                # 也为目标信号添加数据流信息
                                dest_signal = str(bind_name).split('.')[-1]
                                if dest_signal not in all_dataflows:
                                    all_dataflows[dest_signal] = []
                                if flow_info not in all_dataflows[dest_signal]:
                                    all_dataflows[dest_signal].append(flow_info)
                            
                            # 处理时序信息
                            if hasattr(b, 'alwaysinfo') and b.alwaysinfo:
                                always = b.alwaysinfo
                                flow_info['always_type'] = 'clockedge' if b.isClockEdge() else 'combination'
                                if b.isClockEdge():
                                    flow_info['clock_edge'] = 'posedge' if b.getClockEdge() else 'negedge'
                                    flow_info['clock_name'] = str(b.getClockName())
                                    
                        except Exception as e:
                            print(f"处理绑定项时出错: {str(e)}")
                            import traceback
                            traceback.print_exc()
            
            # 为每个信号分配数据流信息
            for signal_name, blocks in signal_blocks.items():
                print(f"\n处理信号: {signal_name}")
                combined_info[signal_name] = {
                    'blocks': blocks,
                    'dataflow': all_dataflows.get(signal_name, [])
                }
                print(f"数据流条目数: {len(combined_info[signal_name]['dataflow'])}")
                
                # 打印详细信息
                if combined_info[signal_name]['dataflow']:
                    print("数据流详情:")
                    for flow in combined_info[signal_name]['dataflow']:
                        print(f"  目标: {flow['termname']}")
                        print(f"  涉及信号: {flow.get('all_signals', [])}")
                        print(f"  表达式: {flow.get('tree_str', 'N/A')}")
        
        return combined_info
        
    except Exception as e:
        import traceback
        print(f"\n=== 错误信息 ===")
        print(f"数据流分析错误: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return {}

def analyze_dataflow(file_path, topmodule):
    analyzer = VerilogDataflowAnalyzer(file_path, topmodule,
                                       noreorder=False,
                                       nobind=False,
                                       preprocess_include=[],
                                       preprocess_define=[])
    analyzer.generate()

    directives = analyzer.get_directives()
    terms = analyzer.getTerms()
    binddict = analyzer.getBinddict()

    optimizer = VerilogDataflowOptimizer(terms, binddict)

    optimizer.resolveConstant()
    resolved_terms = optimizer.getResolvedTerms()
    resolved_binddict = optimizer.getResolvedBinddict()
    constlist = optimizer.getConstlist()

    graphgen = VerilogGraphGenerator(topmodule, terms, binddict,
                                     resolved_terms, resolved_binddict, constlist, "out.png")

    for name, bindlist in binddict.items():
        for bind in bindlist:
            graphgen.generate(str(bind.dest), walk=False, identical=False,
                        step=1, do_reorder=False, delay=False, alwaysinfo=bind.alwaysinfo, withcolor=True)    
    return graphgen.graph

def main():
    """主函数：分析Verilog文件并输出信号分析结果"""
    file_path = './test.v'
    try:
        # 解析Verilog文件
        print("=== 开始解析Verilog文件 ===")
        ast = parse_verilog(file_path)
        print("文件解析完成\n")
        
        # 查找代码块和信号
        print("=== 开始分析代码块和信号 ===")
        all_blocks, signal_blocks = find_blocks_and_signals(ast)
        print(f"找到 {len(all_blocks)} 个代码块")
        print(f"找到 {len(signal_blocks)} 个信号\n")
        
        # 数据流分析
        print("=== 开始数据流分析 ===")
        combined_info = analyze_with_dataflow(file_path, signal_blocks)
        
        # 格式化输出结果
        if combined_info:
            print("\n=== 综合信号分析结果 ===")
            for signal_name, info in combined_info.items():
                print(f"\n信号: {signal_name}")
                print("=" * 40)
                
                # 代码块信息
                print("\n代码块:")
                print("-" * 20)
                for block_type, block_id in info['blocks']:
                    print(f"类型: {block_type}")
                    if block_id in all_blocks:
                        code = all_blocks[block_id][1]
                        formatted_code = "\n    ".join(code.split("\n"))
                        print(f"代码:\n    {formatted_code}\n")
                
                # 数据流信息
                print("数据流:")
                print("-" * 20)
                if info['dataflow']:
                    for flow_info in info['dataflow']:
                        print(f"绑定类型: {flow_info['bind_type']}")
                        if 'tree_type' in flow_info:
                            print(f"表达式类型: {flow_info['tree_type']}")
                            
                        # 显示时序信息
                        if 'clock_edge' in flow_info:
                            print(f"时钟边沿: {flow_info['clock_edge']}")
                            print(f"时钟信号: {flow_info['clock_name']}")
                        if 'reset_edge' in flow_info:
                            print(f"复位边沿: {flow_info['reset_edge']}")
                            print(f"复位信号: {flow_info['reset_name']}")
                            
                        # 显示条件和赋值信息
                        if 'condition' in flow_info:
                            print(f"条件: {flow_info['condition']}")
                        if 'true_value' in flow_info:
                            print(f"True分支: {flow_info['true_value']}")
                        if 'false_value' in flow_info:
                            print(f"False分支: {flow_info['false_value']}")
                            
                        # 显示依赖信号
                        if 'src_signals' in flow_info:
                            deps = flow_info['src_signals']
                            if deps:
                                print("依赖信号:")
                                for dep in deps:
                                    print(f"  - {dep}")
                        print()
                else:
                    print("无数据流信息\n")
                print("=" * 40)
        else:
            print("\n未能获取有效的分析结果")
            
    except Exception as e:
        print(f"\n分析过程中出错: {str(e)}")
        print("\n详细错误信息:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试用的Verilog文件路径
    verilog_file = "./test.v"
    store_verilog_graph(verilog_file)
