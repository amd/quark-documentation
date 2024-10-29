Language Model QAT Using Quark
===========================================================

This document provides examples of Quantization-Aware Training (QAT) for language models using Quark.


Table of Contents
=================

.. contents::
  :local:
  :depth: 1

Supported Models
----------------

+-----------------------------------------+-------------------------------+
| Model Name                              | WEIGHT-ONLY (INT4.g128)       |
+=========================================+===============================+
| microsoft/Phi-3-mini-4k-instruct        | ✓                             |
+-----------------------------------------+-------------------------------+
| THUDM/chatglm3-6b                       | ✓                             |
+-----------------------------------------+-------------------------------+



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
                     + llm_qat
                        + main.py               <--- Main function of example
                     + llm_utils
                        + data_preparation.py
                        + model_preparation.py
                        + evaluation.py

.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->

