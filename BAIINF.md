# BaiInf fork 开发约定

长期开发分支为 `dev`。本轮起点是 b11946，提交 `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`。

已加入四项本机验收改动：F32图捕获复制修正、转置拼接、专家加权汇总调度、四warp专家索引。三项性能分支在Linux HIP默认开启，仍严格匹配gfx1151与各自张量形状；设置对应 `BAIINF_PARITY_CONCAT`、`BAIINF_PARITY_REDUCTION`、`BAIINF_PARITY_IDS` 为0可关闭。其他平台默认关闭，设置1仅供独立验收。复制修正限Linux HIP图捕获。没有加入整数诊断kernel或GPU读回。

产品版本由BaiInf父仓库管理，从0.1.0开始；上游build number与ABI版本独立保留。官方更新通过父仓库 `uv run fork` 在独立接收与合并分支处理，保留双方提交祖先关系，验收后快进dev。不要向dev强推或以官方分支覆盖自研提交。

数值和性能证据位于BaiInf父仓库的 `docs/qwen36-35b-a3b-parity-results-20261102.md`。既有结果只适用于固定本机、模型及843/64负载；默认开关迁移后的正式包须重新验证。
