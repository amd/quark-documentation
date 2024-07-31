:orphan:

:py:mod:`quark.torch.export.gguf_export.gguf_model_writer`
==========================================================

.. py:module:: quark.torch.export.gguf_export.gguf_model_writer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.export.gguf_export.gguf_model_writer.SentencePieceTokenTypes
   quark.torch.export.gguf_export.gguf_model_writer.ModelWriter
   quark.torch.export.gguf_export.gguf_model_writer.LlamaModelWriter




.. py:class:: SentencePieceTokenTypes




   Enum where members are also (and must be) ints


.. py:class:: ModelWriter(model_name: str, json_path: pathlib.Path, safetensor_path: pathlib.Path, tokenizer_dir: pathlib.Path, fname_out: pathlib.Path, is_big_endian: bool = False, use_temp_file: bool = False)




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: LlamaModelWriter(model_name: str, json_path: pathlib.Path, safetensor_path: pathlib.Path, tokenizer_dir: pathlib.Path, fname_out: pathlib.Path, is_big_endian: bool = False, use_temp_file: bool = False)




   Helper class that provides a standard way to create an ABC using
   inheritance.


