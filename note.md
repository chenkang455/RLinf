需要手动注释以下内容
```
# Force enable gradient checkpointing if we're in training mode and the model supports it
# if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
#     if not self.gemma_expert.model.gradient_checkpointing:
#         print("Forcing gradient checkpointing to be enabled for Gemma expert model")
#         self.gemma_expert.model.gradient_checkpointing = True
#     use_gradient_checkpointing = True
```