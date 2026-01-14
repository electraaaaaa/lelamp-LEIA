import inspect
import json
class Tool:
    tool_registry = dict()
    tools_schema = list()
    @classmethod
    def register_tool(cls, tool):
        func_name = tool.__name__
        func_doc = tool.__doc__
        sig = inspect.signature(tool)
        parameters = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            if param_name == "self":continue
            param_type = "string"
            if param.annotation == int: param_type = "integer"
            elif param.annotation == bool: param_type = "boolean"
            elif param.annotation == float: param_type = "number"

            parameters["properties"][param_name] = {
                "type": param_type,
                "description": f"Arg: {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        tool_def = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func_doc.strip(),
                "parameters": parameters
            }
        }
        cls.tool_registry[func_name] = tool
        cls.tools_schema.append(tool_def)
        return tool
    @classmethod
    async def execute(cls, name, args_json, instance):
        """执行工具并返回字符串结果"""
        if name not in cls.tool_registry:
            return f"Error: Tool {name} not found"
        try:
            func = cls.tool_registry[name]
            args = json.loads(args_json)
            if hasattr(instance, name):
                bound_method = getattr(instance, name)
                # 检查是否是协程 (async)
                if inspect.iscoroutinefunction(bound_method):
                    result = await bound_method(**args)
                else:
                    result = bound_method(**args)
            else:
                # 静态函数或普通函数的情况
                if inspect.iscoroutinefunction(func):
                    result = await func(**args)
                else:
                    result = func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
