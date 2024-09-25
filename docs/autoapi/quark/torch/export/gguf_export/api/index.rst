:orphan:

:py:mod:`quark.torch.export.gguf_export.api`
============================================

.. py:module:: quark.torch.export.gguf_export.api


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.export.gguf_export.api.convert_exported_model_to_gguf



.. py:function:: convert_exported_model_to_gguf(model_name: str, json_path: Union[str, pathlib.Path], safetensor_path: Union[str, pathlib.Path], tokenizer_dir: Union[str, pathlib.Path], output_file_path: Union[str, pathlib.Path]) -> None

   This function is used to convert quark exported model to gguf model.

   Args:
       model_name (str): name of this model which will be written to gguf field `general.name`
       json_path (Union[str, Path]): Quark exported model consists of a `.json` file and a `.safetensors` file.
           This arguments indicates the path of `.json` file
       safetensor_path (Union[str, Path]): Path of `.safetensors` file.
       tokenizer_dir (Union[str, Path]): Tokenizer needs to be encoded into gguf model.
           This argument specifies the directory path of tokenizer which contains tokenizer.json, tokenizer_config.json and/or tokenizer.model
       output_file_path (str): The path of generated gguf model.


