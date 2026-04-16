## 2026-04-16 模型缓存链路校验与模块化测试落地

- 完成时间：2026-04-16
- 完成内容：
  - 校验了 `self.model(...)` 第二个返回值作为新 `past_key_values` 向上回传的可行性，确认这条数据流语义正确。
  - 在 `test/` 目录下建立了首批 4 个独立测试脚本：
    - `test/test_model_cache_flow.py`
    - `test/test_attention_mask_flow.py`
    - `test/test_generation_smoke.py`
    - `test/test_pretrain_dataset.py`
  - 这些脚本覆盖了 KV cache 回传、attention mask 对齐、`generate()` 冒烟和预训练数据集输入输出约束。
- 说明：
  - 当前测试体系采用“可直接 `python test/xxx.py` 运行的独立脚本 + assert”方案，便于后续逐模块补充。
