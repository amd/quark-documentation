:orphan:

:py:mod:`quark.torch.algorithm.gptq.gptq`
=========================================

.. py:module:: quark.torch.algorithm.gptq.gptq


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.gptq.gptq.GptqProcessor




.. py:class:: GptqProcessor(gptq_model: quark.torch.algorithm.models.base.BaseQuantModelForCausalLM, model: transformers.PreTrainedModel, quant_algo_config: quark.torch.quantization.config.config.GPTQConfig, data_loader: List[Dict[str, torch.Tensor]])




   Helper class that provides a standard way to create an ABC using
   inheritance.


