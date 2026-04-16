"""
测试目标：
- 验证 `eval_llm.py` 中本地权重路径解析同时支持相对目录、绝对目录和 `.pth` 文件路径。
- 验证主权重与 LoRA 权重在文件路径模式下不会互相污染。

预期结果：
- 相对目录会解析到项目 `out/` 下的目标文件。
- 绝对目录会直接拼接目标文件名。
- `.pth` 文件路径在主权重模式下直接返回，在 LoRA 模式下回退到父目录拼接。

测试步骤：
1. 分别构造相对目录、绝对目录和绝对文件路径。
2. 调用 `resolve_weight_path(...)`。
3. 检查解析结果是否符合预期。
"""
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_llm import resolve_weight_path
from trainer.trainer_utils import DEFAULT_OUT_DIR


def main():
    """执行 eval_llm 本地权重路径解析测试。"""
    relative_path = resolve_weight_path("out", "pretrain_768.pth")
    assert relative_path == DEFAULT_OUT_DIR / "pretrain_768.pth"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        absolute_dir_path = resolve_weight_path(str(tmp_path), "pretrain_768.pth")
        assert absolute_dir_path == tmp_path / "pretrain_768.pth"

        checkpoint_file = tmp_path / "pretrain_768.pth"
        checkpoint_file.touch()

        main_weight_path = resolve_weight_path(
            str(checkpoint_file),
            "full_sft_768.pth",
            allow_file_path=True,
        )
        lora_weight_path = resolve_weight_path(
            str(checkpoint_file),
            "lora_identity_768.pth",
            allow_file_path=False,
        )

        assert main_weight_path == checkpoint_file
        assert lora_weight_path == tmp_path / "lora_identity_768.pth"

    print("test_eval_llm_path: PASS")


if __name__ == "__main__":
    main()
