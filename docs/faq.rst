Frequently Asked Questions (FAQ)
================================

Environment Issues
------------------

Issue 1
~~~~~~~

Windows CPU mode does not support fp16.

**Solution**:

Because of torch
`issue <https://github.com/pytorch/pytorch/issues/52291>`__\ ， Windows
CPU mode cannot perfectly support fp16.

C++ Compilation Issues
----------------------

.. _issue-1-1:

Issue 1
~~~~~~~

Stuck in the compilation phase for a long time (over ten minutes), the
terminal shows like:

.. code:: shell

   [QUARK-INFO]: Configuration checking start. 

   [QUARK-INFO]: C++ kernel build directory [cache folder path]/torch_extensions/py39...

**Solution**:

delete the cache folder ``[cache folder path]/torch_extensions`` and run
quark again.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
