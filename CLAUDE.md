# CLAUDE.md

本项目的目标是：通过手写代码复现 [minimind](https://github.com/jingyaogong/minimind/tree/master) 模型的训练和推理，从而学习入门大模型。

## 需要 AI 协助我完成的事

- 当前需要 AI 协助我完成的事记录在 @TODO.md 中。
- 当事情完成后：
  - 需要将记录从 @TODO.md 中移动到 @todo_history.md 中，并自拟事件标题、添加完成时间、如果有需要则可适当添加描述说明。
  - 需要将记录从 @TODO.md 中删除。

## 测试规范

- `test/` 目录采用分层结构：
  - `test/model/`：模型结构、缓存链路、前向与生成相关测试。
  - `test/data/`：数据集与数据处理相关测试。
  - `test/inference/`：推理入口与路径解析相关测试。
- 测试脚本统一命名为 `test_<module>.py`，并保持可以直接通过 `python test/.../test_xxx.py` 单独运行。
- 每个测试脚本文件开头都要补充测试说明注释，至少包括：测试目标、预期结果、测试步骤。
- 测试脚本内部要为关键函数、关键输入和关键断言补充中文注释，说明验证意图。
- 允许在 `test/` 下维护轻量共享工具文件，例如项目根目录注入、最小模型构造、通用断言辅助；但当前不引入完整 `pytest` 框架迁移。
- 新增测试时优先顺序如下：
  - `Attention`
  - `RMSNorm`
  - `FeedForward`
  - `MiniMindBlock`
  - `MiniMindForCausalLM.forward`
  - `eval_llm` 本地路径解析
- 完成一批测试规范化工作后，需要同步：
  - 将对应事项从 `TODO.md` 移到 `todo_history.md`
  - 记录完成时间与简要说明
