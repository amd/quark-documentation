Quark Extension for Brevitas Integration
========================================

   **Note:** This feature is experimental and under active development.
   As such, it does not guarantee backward compatibility for the APIs.
   Users should exercise caution and expect changes in future releases.

Overview
--------

This documentation provides an outline for the new extension feature
under development within Quark. The extension feature enables the
integration of external quantization libraries into Quark, enhancing its
capabilities and broadening its applicability. A key example of this
integration is with Brevitas, another popular quantization library.

Extension Feature Design
------------------------

The Quark extension feature is designed to facilitate integration
without code leakage into the Quark core. This is achieved by exposing a
well-defined interface that external libraries can implement. The main
components of this interface include the ``QuantizationConfig``
and ``ModelQuantizer`` classes.

QuantizationConfig
~~~~~~~~~~~~~~~~~~

This class is responsible for holding all configuration parameters
necessary for quantization processes.

ModelQuantizer
~~~~~~~~~~~~~~

The ModelQuantizer class serves as the primary interface for model
quantization. Implementations of this class should encapsulate the logic
required to parse the QuantizationConfig and apply quantization steps
accordingly from the external library.

Brevitas Integration
--------------------

Brevitas is one of the first libraries that we integrate with quark
using the new extension feature.

Step-by-Step Integration
~~~~~~~~~~~~~~~~~~~~~~~~

1. Implement QuantizationConfig: Begin by creating a subclass of
   QuantizationConfig specific to Brevitas settings. This subclass
   should define all Brevitas-specific parameters such as bit-widths,
   quantization modes, and other relevant settings.

2. Develop ModelQuantizer: Create a subclass of ModelQuantizer that
   leverages Brevitas for the quantization process. This subclass should
   implement methods to quantize models using Brevitas APIs, ensuring
   that it complies with the interface expected by Quark.

3. Register with Quark: Once the classes are implemented, register them
   with Quark's extension system. This involves adding the Brevitas
   quantizer as an available option within Quark's configuration files
   or through an API call, depending on Quark's system design. Note:
   This feature is currently under developement. For the time being,
   users can manually instantiate and use the ``BrevitasQuantizer``
   directly as in the examples provided within this directory.

Conclusion
----------

The extension feature in Quark aims to provide a flexible and powerful
mechanism for integrating various quantization libraries, enhancing both
Quark's and the external libraries' utility and effectiveness. Through
this interface, users can leverage advanced features from other
libraries like Brevitas while maintaining a streamlined quantization
process within Quark's ecosystem.

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
