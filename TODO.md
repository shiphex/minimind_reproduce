待执行命令：
1. 关于`tset`路径下现有的测试脚本：
   - 为现有测试脚本添加测试说明，包括测试目的、预期结果和测试步骤，测试说明以注释的形式编写在每个测试脚本的开头。
   - 为现有测试脚本代码添加中文注释，说明每个函数的作用和参数。

2. 第二批测试按模块逐步补齐，建议顺序：
   - test/test_attention.py
   - test/test_rmsnorm.py
   - test/test_ffn.py
   - test/test_block.py
   - test/test_model_forward.py
   - test/test_eval_llm_path.py

3. 为`tset`路径设置合理的文件结构，包括测试脚本的组织和命名规范。
4. 将测试规范化，将规范要求保存到 @AGENTS.md 的子项目中。