class ToolEntry:

    __slots__ = ("name","toolset","schema",
                 "handler","check_fn","is_async")

    def __init__(self,name,toolset,schema,handler
                 ,check_fn,is_async):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.is_async = is_async
