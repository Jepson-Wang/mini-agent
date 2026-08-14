"""gRPC 服务端：把 Agent 作为一个服务暴露给 Go 后端。

叫 grpcserver 而不是 backend —— 仓库外面那个 Go 项目才叫 backend，
两边同名的话，"改一下 backend 里的取消逻辑"这句话没人能听懂指的是哪边。
"""
