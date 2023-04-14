import contextlib
import paddle
from .opcode_translator import ConvertGuard, eval_frame_callback
from .symbolic.symbolic_context import SymbolicTraceContext
from .proxy_tensor import ProxyTensorContext, ProxyTensor
from .convert_functions import convert_function

def symbolic_trace(func):
    def symbolic_traced_func(*args, **kw):
        ProxyTensorContext().reset()
        with SymbolicTraceContext() as ctx:
<<<<<<< HEAD
            paddle.fluid.core.set_eval_frame(eval_frame_callback)
            returns = func(*args, **kw)
            paddle.fluid.core.set_eval_frame(None)

        # TODO( output analysis, we can get out symbols here. )
        if returns is None:
            return None

        return SymbolicTraceContext().start_compile(ProxyTensorContext().get_runtime())
    return wrapped
=======
            with ConvertGuard(convert_function) as ctx:
                paddle.fluid.core.set_eval_frame(eval_frame_callback)
                try:
                    returns = func(*args, **kw)
                except Exception as e:
                    raise e
                finally: 
                    paddle.fluid.core.set_eval_frame(None)
        ret = SymbolicTraceContext().start_compile(
            ProxyTensorContext(),
            output=returns)
        return ret
    return symbolic_traced_func
>>>>>>> 6b38ea48494f4f34e8ecc0f0f94dbcd7310ac0f5
