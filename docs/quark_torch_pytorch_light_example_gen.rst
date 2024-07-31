Introduction
============

This script demonstrates the integration of
**APL**\ (**``AMD Pytorch-light``,internal project name**) into
Quark.APL is a lightweight model. **APL** is a lightweight model
optimization library based on PyTorch, designed primarily for
developers. **APL** is AMD’s internal quantization framework; external
users need to request access. Ensure **APL** is installed before running
this example.

**APL** supports a variety of quantization methods and advanced
quantization data types. ``Quark`` provides a user-friendly interface,
allowing users to easily leverage these quantization techniques. This
example combines the strengths of both frameworks, enabling users to
invoke **APL** through Quark’s interface. We have prepared three
examples that demonstrate the use of the BFP16, INTK, and BRECQ
quantization schemes of **APL** via Quark’s interface.

Example 1
=========

In this example, we use the llama2 model and call **APL** to perform the
int_k model quantization. - model : ``llama2 7b`` - calib method :
``minmax`` - quant dtype : ``int8``

**replace ops**: - replace ``nn.Linear`` =>
``pytorchlight.nn.linear_layer.LLinear``

**run script**

for inner test
``model_path=/group/ossmodelzoo/quark_torch/huggingface_pretrained_models/meta-llama/Llama-2-7b``
and
``dataset_path= /group/ossdphi_algo_scratch_06/meng/pytorch-light/examples/calib/hf_llm/``

::

   model_path={your `llama-2-7b` model path}
   dataset_path={your data path}

   python quantize_quark.py \
       --model llama-7b \
       --model_path  ${model_path}\
       --seqlen 4096 \
       --dataset_path ${dataset_path} \
       --eval \

Example 2
=========

In this example, we use the opt-125m model and call **APL** to perform
bfp16 model quantization. We support the quantization of nn.Linear,
nn.LayerNorm, and nn.Softmax through **APL**. - model : ``opt-125m`` -
calib method : ``minmax`` - quant dtype : ``bfp16``

**replace ops**: - ``nn.Linear`` =>
``pytorchlight.nn.linear_layer.LLinear`` - ``nn.LayerNorm`` =>
``pytorchlight.nn.normalization_layer.LLayerNorm`` - ``nn.Softmax`` =>
``pytorchlight.nn.activation_layer.LSoftmax``

**run script**

for inner test
``model_path=/group/modelzoo/sequence_learning/weights/nlp-pretrained-model/opt_125m_pretrained_pytorch``
and
``dataset_path= /group/ossdphi_algo_scratch_06/meng/pytorch-light/examples/calib/hf_llm/``

::

   model_path={your `opt-125m` model path}
   dataset_path={your data path}

   python quantize_quark.py \
       --model opt-125m \
       --model_path ${model_path} \
       --seqlen 4096 \
       --qconfig 0 \
       --eval \
       --qscale_type fp32 \
       --dataset_path ${dataset_path} \
       --example bfp16

Example 3
=========

This example demonstrates how to use the ``brecq`` algorithm through
``quark`` call **APL**

-  model : ``opt-125m``
-  calib method : ``minmax``
-  quant dtype : ``int8``

for inner test
``model_path=/group/modelzoo/sequence_learning/weights/nlp-pretrained-model/opt_125m_pretrained_pytorch``\ and
``dataset_path= /group/ossdphi_algo_scratch_06/meng/pytorch-light/examples/calib/hf_llm/``

::

   model_path={your `opt-125m` model path}
   dataset_path={your data path}

   export CUDA_VISIBLE_DEVICES=0,5,6;
   python quantize_quark.py \
       --model opt-125m \
       --model_path ${model_path} \
       --seqlen 1024 \
       --eval \
       --example brecq \
       --dataset_path ${dataset_path}
