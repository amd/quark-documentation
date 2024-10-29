Language Model Pruning Using Quark
==================================

This document provides examples of pruning the language models(OPT, Llama…) using Quark.


Table of Contents
=================

.. contents::
  :local:
  :depth: 1

Supported Models
----------------


+------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| Model Name                         | Model Size | Pruning Rate | Pruned Model Size | Before Pruning PPL On Wiki2 | After Pruning PPL On Wiki2 |
+====================================+============+==============+=================================================+============================+
| Qwen/Qwen2.5-14B-Instruct          | 14.8B      | 7.0284%      | 13.7B             | 5.6986                      | 7.5994                     |
+------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| CohereForAI/c4ai-command-r-08-2024 | 32.3B      | 7.4025%      | 29.9B             | 4.5081                      | 6.3794                     |
+------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| facebook/opt-6.7b                  | 6.7B       | 7.5651%      | 6.2B              | 10.8602                     | 11.8958                    |
+------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| meta-llama/Llama-2-7b-hf           | 6.7B       | 6.7224%      | 6.2B              | 5.4721                      | 6.2462                     |
+------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+



.. note::
   - Experiments for all models using 128 samples of pile val dataset as a calibration dataset.


How to get the example code and script
--------------------------------------

Users can get the example code after downloading and unzipping ``quark.zip`` (referring to :doc:`Installation Guide <install>`).
The example folder is in quark.zip.

   Directory Structure of the ZIP File:

   ::

         + quark.zip
            + examples
               + torch
                  + language_modeling
                     + llm_pruning
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
