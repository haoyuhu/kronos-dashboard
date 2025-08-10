# Kronos Dashboard

本项目演示 Kronos 金融时间序列概率预测系统的端到端流程：数据获取、模型推理、图表生成与网页展示。
Dashboard 链接：[Kronos Dashboard](https://haoyuhu.github.io/kronos-dashboard/)

## 项目结构

```
kronos-dashboard/
├── config/
│   └── config.ini           # 运行配置（不再包含 mock）
├── core/
│   ├── config.py            # 配置解析
│   ├── data_loader.py       # 数据读取工具
│   ├── data_source.py       # 数据源适配器（Binance / AkShare）
│   ├── engine.py            # 预测引擎（拼装流程）
│   ├── model_manager.py     # 模型加载与推理器封装
│   ├── pipeline.py          # 预测与绘图
│   └── web_generator.py     # 网页生成器
├── main.py                  # 入口脚本（使用 --mock 控制演示模式）
├── model/
│   ├── __init__.py
│   ├── kronos.py
│   └── module.py
├── templates/
│   └── index.jinja          # 页面模板
├── docs/
│   ├── index.html           # 生成的页面
│   └── static/
│       ├── chart/           # Mock 模式下预置的 PNG 图表
│       ├── img/
│       └── style.css
└── requirements.txt
```

## 快速开始

- 安装依赖：
```
pip install -r requirements.txt
```

- 运行（真实推理）：
```
python main.py
```

- 运行（Mock 演示，仅使用 web/static/chart 下已有 PNG 图片）：
```
python main.py --mock
```

- 定时运行（结合任一模式）：
```
python main.py --schedule [--mock]
```

## 配置说明（config/config.ini）

当前配置示例（已移除 mock）：
```
[common]
pred_horizon = 24
n_predictions = 30
hist_points = 200
vol_window = 180
update_interval_minutes = 60

[binance]
symbols = BTCUSDT,ETHUSDT
interval = 1h

[akshare]
symbols = 159985
interval = 1d

[model]
# 可选模型尺寸：mini, small, base
# mini: 4.1M 参数，2048 上下文，速度最快
# small: 24.7M 参数，512 上下文，均衡（默认）
# base: 102.3M 参数，512 上下文，精度更好
model_size = small
# 设备：cpu 或 cuda（如环境支持 CUDA）
device = cpu
```

说明：
- mock 模式完全通过命令行 `--mock` 控制，配置文件中不再有 mock 字段。

### 模型配置（[model]）
- model_size：选择推理所用的 Kronos 模型尺寸，支持 `mini` / `small` / `base`，默认 `small`。
- device：设置推理设备，缺省为 `cpu`；若本机安装了 CUDA，可改为 `cuda`。
- 上下文长度（max_context）会根据模型尺寸自动设置：`mini` 为 2048，其它（`small` / `base`）为 512。

## Mock 模式
- 仅从 `web/static/chart` 目录读取 `.png` 文件作为占位图（不再支持 `.svg`）。
- 页面生成后输出至 `web/index.html`，资源路径相对于 `web/`。
- 文件命名建议与真实推理的输出保持一致（例如：`BTCUSDT.png`）。

## 常见问题
- 页面空白或无图：请确认 `web/static/chart` 目录存在且包含至少一个 `.png` 文件，或使用真实推理模式生成图表。
- 静态资源 404：请确保 `web/index.html` 与 `web/static/` 目录结构未被改动。

更多细节请查阅源码注释。