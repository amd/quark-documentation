:orphan:

:py:mod:`quark.shares.utils.testing_utils`
==========================================

.. py:module:: quark.shares.utils.testing_utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.shares.utils.testing_utils.require_torch_gpu
   quark.shares.utils.testing_utils.require_accelerate
   quark.shares.utils.testing_utils.delete_directory_content
   quark.shares.utils.testing_utils.retry_flaky_test



.. py:function:: require_torch_gpu(test_case: Any) -> Any

   Decorator marking a test that requires CUDA and PyTorch.


.. py:function:: require_accelerate(test_case: Any) -> Any

   Decorator marking a test that requires Accelerate library.


.. py:function:: delete_directory_content(directory: str) -> None

   Deletes all content within a directory

   Args:
       directory (str): The path to the directory whose content should be deleted.


.. py:function:: retry_flaky_test(max_attempts: int = 5)

   Allows to retry flaky tests multiple times.


