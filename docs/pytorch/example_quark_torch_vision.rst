Vision Model Quantization using Quark FX Graph Mode
===================================================


In this example, we present a vision model quantization workflow. The
user specified a ``nn.Module`` and transformed the model to
``torch.fx.GraphModule`` format by using PyTorch API. During the
quantization process, after annotation and insertion quantizers, this
modified ``fx.GraphModule`` can be used to perform PTQ (Post Training
Quantization), or/and QAT (Quantization Aware Training). We supply a
demonstration code and show how users assign ``quant config``, more
information can be found in User Guide.

Get example code and script
~~~~~~~~~~~~~~~~~~~~~~~~~~~
After unzip ``quark.zip`` (referring to :doc:`Installation Guide <install>`).
The example folder is in quark.zip. In folder ``/examples/torch/vision``, user can get the detailed explaination of
image classification and object detection quantization demonstration code.


PTQ
~~~

In PTQ, after the ``FakeQuantize`` is inserted, during the calibration,
the ``observer`` is activated for recoding the tensor's distribution the
values such as min and max will be recorded to calculate quant
parameters, while not performing fake quantizing. This means all the
calculations are under FP32 precision. After the calibration, we will
activate the fake quantizer to perform quantization and evaluation.

QAT
~~~

Same as PTQ, after the model is prepared. During the training process,
both ``observer`` and ``fake_quant`` are effective, ``observer`` is used
for recording the tensor's distribution such as min and max value to
calculate quantization parameters, and the tensor will be quantized by
``fake_quant``.

TQT
~~~

A method for uniform symmetric quantizers using standard backpropagation
and gradient descent. Different with QAT, TQT add scale-factors gradient.
And different with LSQ that trains the scale-factors directly, which leads
to stability issues, TQT constrains scale-factors to power-of-2 and uses a
gradient formulation to train log-thresholds instead. So theoretically TQT is
better than LSQ and LSQ is better than QAT. For efficient fixed-point implementations,
TQT constrains quantization scheme to use: Symmetric、Per-tensor scaling、Power-of-2 scaling.
Currently, only signed data are supported for tqt. More experimental results are on the way.

Quick Start
-----------

Perform PTQ to get the quantized model and export to ONNX

::

   python3 quantize.py --data_dir [Train and Test Data floder] \
                       --model_name [mobilenetv2 or resnet18] \
                       --pretrained [Pre-trained model file address] \
                       --model_export onnx
                       --export_dir [dir to save exported model]

Users can also select to perform QAT to further improve classification
accuracy. Typically, that there are some training parameters that need
to be modified for higher accuracy.

::

   python3 quantize.py --data_dir [Train and Test Data floder] \
                       --model_name [mobilenetv2 or resnet18] \
                       --pretrained [Pre-trained model file address] \
                       --model_export onnx
                       --export_dir [dir to save exported model] \
                       --qat True

LSQ and TQT are optimized methods for QAT which can improve accuracy theoretically. The params ``--tqt True`` ``--lsq True`` are provided for users to try. Model export is not supported now.

**Fine-Grained User Guide**
---------------------------

**Step1:Prepare float point model, dataset, loss function**

.. code:: python

   from torchvision.models import resnet18
   float_model = resnet18(pretrained=False)
   float_model.load_state_dict(torch.load(pretrained))
   calib_loader = prepare_calib_dataset(args.data_dir, device, calib_length=args.train_batch_size * 10)
   train_loader, val_loader = prepare_data_loaders(args.data_dir)
   criterion = nn.CrossEntropyLoss().to(device)

**Step 2: transformer the ``torch.nn.Module`` to
``torch.fx.GraphModule``.**

::

   from torch._export import capture_pre_autograd_graph
   example_inputs = (torch.rand(args.train_batch_size, 3, 224, 224).to(device), )
   graph_model = capture_pre_autograd_graph(float_model, example_inputs)

**Step3: Init the quantizer and quantization configuration**

::

   from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config
   from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
   from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
   INT8_PER_TENSER_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           symmetric=True,
                                           scale_type=ScaleType.float,
                                           round_method=RoundType.half_even,
                                           is_dynamic=False)
   quant_config = QuantizationConfig(input_tensors=INT8_PER_TENSER_SPEC,
                                         output_tensors=INT8_PER_TENSER_SPEC,
                                         weight=INT8_PER_TENSER_SPEC,
                                         bias=INT8_PER_TENSER_SPEC)
   quant_config = Config(global_quant_config=quant_config,
                   quant_mode=QuantizationMode.fx_graph_mode)
   quantizer = ModelQuantizer(quant_config)

**Step4: Generate the quantized graph model by performing calibration**

::

   quantized_model = quantizer.quantize_model(graph_model, calib_loader)

**Step5 (Optional): QAT for more high accuracy**

::

   train(quantized_model, train_loader, val_loader, criterion, device_ids)

**Step6: Validate model performance and export**

::

   acc1_quant = validate(val_loader, quantized_model, criterion, device)
   freezed_model = quantizer.freeze(prepared_model)
   acc1_freeze = validate(val_loader, freezed_model, criterion, device)
   # check whether acc1_quant == acc1_freeze

   # ==============export to ONNX ==================
   from quark.torch import ModelExporter
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   config = ExporterConfig(json_export_config=JsonExporterConfig())
   exporter = ModelExporter(config=config, export_dir=args.export_dir)
   example_inputs = (torch.rand(batch_size, 3, 224, 224).to(device),)
   exporter.export_onnx_model(freezed_model, example_inputs[0])

   # ==========export using torch.export============
   example_inputs = (next(iter(val_loader))[0].to(device),)
   model_file_path = os.path.join(args.export_dir, args.model_name + ".pth")
   exported_model = torch.export.export(freezeded_model, example_inputs)
   torch.export.save(exported_model, model_file_path)

Experiment Result
-----------------


1. Image classification Task PTQ/QAT Result.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We conducted PTQ and QAT on both ResNet-18 and MobileNet-V2. In these
model, all weight, bias, and activation are quantized. All kinds of
Tensors are quantized in INT8, per-tensor, symmetric(zero point is 0).
The scale factor is in float format. The following table shows the
validation accuracy in the ImageNet dataset produced by the above
script.

============ =============== ===============
Method       ResNet-18       MobileNetV2
============ =============== ===============
Float Model  69.764 / 89.085 71.881 / 90.301
PTQ  (INT8)  69.084 / 88.648 65.291 / 86.254
QAT (INT8)   69.469 / 88.872 68.562 /88.484
============ =============== ===============

2. Object Detection Task PTQ/QAT Result.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We conducted PTQ and QAT on YOLO-NAS. In this model quantization, we partly quantized this model by assigned the configuration.

==============  ===============  ===============  ===============
Metric          FP32 model       INT 8 PTQ         INT 8 QAT
==============  ===============  ===============  ===============
mAP@0.50        0.6466             0.6236           0.6239
mAP@0.50:0.95   0.4759             0.4537           0.4532
==============  ===============  ===============  ===============

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
