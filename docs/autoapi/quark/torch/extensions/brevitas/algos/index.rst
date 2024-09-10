:py:mod:`quark.torch.extensions.brevitas.algos`
===============================================

.. py:module:: quark.torch.extensions.brevitas.algos

.. autoapi-nested-parse::

   Pre-quantization optimization and post quantization algorithms for Brevitas API.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.extensions.brevitas.algos.Preprocess
   quark.torch.extensions.brevitas.algos.ActivationEqualization
   quark.torch.extensions.brevitas.algos.GPFQ
   quark.torch.extensions.brevitas.algos.GPFA2Q
   quark.torch.extensions.brevitas.algos.GPTQ
   quark.torch.extensions.brevitas.algos.BiasCorrection




.. py:class:: Preprocess(trace_model: bool = True, equalize_iterations: int = 20, equalize_merge_bias: bool = True, merge_batch_norm: bool = True, channel_splitting_ratio: float = 0.0, channel_splitting_split_input: bool = False)




   Preprocesses the model to make it easier to quantize.


.. py:class:: ActivationEqualization(is_layerwise: bool = True, alpha: float = 0.5)




   Activation Equalization from the paper "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" by Nagel et al.

   - `is_layerwise`: Whether the model having ActivationEqualization applied to it is using Backend.layerwise for its quantization or not.


.. py:class:: GPFQ(act_order: bool = False, percentage_of_processed_inputs: float = 1.0)




   GPFQ or Greedy Path Following Quantization from the papers
   - "Post-training Quantization for Neural Networks with Provable Guarantees" by Zhang et al. and
   - "A Greedy Algorithm for Quantizing Neural Networks" by Lybrand et al.


.. py:class:: GPFA2Q(act_order: bool = False, percentage_of_processed_inputs: float = 1.0, accumulator_bit_width: int = 16)




   Extension of GPFQ using A2Q or Accumulator-Aware Quantization from the paper "A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance" by Colbert et al.


.. py:class:: GPTQ(act_order: bool = False)




   GPTQ or Generative Pre-Trained Transformers Quantization from the paper "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" by Frantar et al.


.. py:class:: BiasCorrection




   Bias correction from the paper "Data-Free Quantization Through Weight Equalization and Bias Correction" by Nagel et al.


