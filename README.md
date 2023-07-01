# chat_with_rwkv

RWKV 是一种开源的大语言模型。本项目是通过 rwkv.cpp、fastapi 为 RWKV 实现一个接口。


## 执行

推理使用了 rwkv.cpp，支持 avx2 指令集加速。因此使用纯 cpu 也可以进行推理（虽然有点慢）。

```sh
cd <repo_root>
pip install -t _vendor -r requirements.txt
python .
```

## 接口

TBD.

## LICENSE

RWKV 模型的协议，请见其授权声明；rwkv 目录源自 rwkv.app 项目，请见其授权声明；其他文件如无声明均采用 MIT 协议。
