## 2026-04-16 模型缓存链路校验与模块化测试落地

- 完成时间：2026-04-16
- 完成内容：
  - 校验了 `self.model(...)` 的第二个返回值作为新 `past_key_values` 向上回传的可行性，确认该数据流语义正确。
  - 在 `test/` 目录下建立了首批 4 个独立测试脚本：
    - `test/test_model_cache_flow.py`
    - `test/test_attention_mask_flow.py`
    - `test/test_generation_smoke.py`
    - `test/test_pretrain_dataset.py`
  - 这批脚本覆盖了 KV cache 回传、attention mask 对齐、`generate()` 冒烟和预训练数据集输入输出约束。
- 说明：
  - 当时的测试体系采用“可直接运行的独立脚本 + assert”方案，为后续按模块扩展测试打基础。

## 2026-04-16 测试目录分层整理与规范沉淀

- 完成时间：2026-04-16
- 完成内容：
  - 将 `test/` 目录重整为 `test/model/`、`test/data/`、`test/inference/` 三层结构。
  - 迁移并规范化了现有 4 个测试脚本，补齐了文件头测试说明和中文注释。
  - 新增第二批 6 个测试脚本，覆盖 `Attention`、`RMSNorm`、`FeedForward`、`MiniMindBlock`、`MiniMindForCausalLM.forward` 和 `eval_llm` 路径解析。
  - 新增 `test/_shared.py`，统一维护项目根目录注入、最小模型构造和通用断言辅助。
  - 在 `CLAUDE.md` 中新增“测试规范”章节，明确测试目录分层、命名规则、文件头说明要求和维护约定。
- 说明：
  - 当前测试体系继续采用“可直接运行的独立脚本 + assert”方案，不引入完整 `pytest` 迁移。
