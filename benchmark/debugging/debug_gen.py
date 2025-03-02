import autogen
import sys
sys.path.insert(0, sys.path[0]+"/../../")
import generation
from unzip import get_bug_info

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json"
)

llm_config = {'config_list': config_list}

def debug_agent(code, bug):
    debugger = autogen.AssistantAgent(
        name="debugger",
        human_input_mode="NEVER",
        llm_config=llm_config,
        system_message="""
        You are a Verilog code debugger.
        You should only give out the correct code.
        """,
    )
    debugging_message = debugger.generate_reply(messages=[{
        "role":"user",
        "content":f"""
        Please debug the following code according to that error message:
        code: {code}
        error message: {bug}
        """
    }])
    return debugging_message

def debug_agent(model, RAG)
