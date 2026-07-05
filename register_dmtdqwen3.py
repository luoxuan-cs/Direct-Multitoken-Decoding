# Register DMTD Qwen3 for ms-swift (import via --external_plugins).
from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.model_arch import ModelArch
from swift.template.constant import TemplateType

register_model(
    ModelMeta(
        'dmtdqwen3',
        [
            ModelGroup(
                [
                    Model(model_path='.'),
                ],
                TemplateType.qwen3,
            ),
        ],
        template=TemplateType.qwen3,
        architectures=['DMTDQwen3ForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformer==5.6.2'],
    ),
    exist_ok=True,
)
