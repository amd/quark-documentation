Language Model Evaluation in Quark
==================================

This document provides examples of evaluating large language models using Quark evaluation API. Models are evaluated either by calculating perplexity (PPL) on WikiText2 or by using the third-party benchmark `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`__.
Quark supports two types of perplexity calculations: one with a sequence length of 2048 and another for KV cache, while lm-evaluation-harness also includes a separate perplexity metric. By default, Quark uses the 2048-sequence-length perplexity as its primary perplexity metric.
The evaluated models can be either pre-trained or quantized, and evaluation can be performed after the model is exported from Quark or during the quantization pipeline.



How to get the example code and script
-----------------------

Users can get the example code after downloading and unzipping ``quark.zip`` (referring to :doc:`Installation Guide <install>`).
The example folder is in quark.zip.

   Directory Structure of the ZIP File:

   ::

         + quark.zip
            + examples
               + torch
                  + language_modeling
                     + llm_ptq
                        + README.md                       <--- Scripts
                        + quantize_quark.py               <--- Main function of example
                        + configuration_preparation.py
                     + llm_eval
                        + README.md                       <--- Scripts
                        + requirements.txt
                        + evaluation.py                   <--- Main function of evaluation
                     + utils
                        + data_preparation.py
                        + model_preparation.py

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
