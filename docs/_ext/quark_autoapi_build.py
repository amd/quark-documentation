#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from sphinx.util import logging

logger = logging.getLogger(__name__)

def update_autoapi_toc_placeholder(app, docname, source):
    """
    Replace `@quark_autoapi_toc_placeholder@` with actual autoapi toc when QUARK_SPHINX_BUILD_SKIP_AUTOAPI is not set on env var
    """

    autoapi_index_rst = 'autoapi_index.rst_'
    if "READTHEDOCS" not in os.environ:
        # TODO: Start using github.com/amd/quark on readthedocs.com project which has same folder structure
        autoapi_index_rst = os.path.join('source', autoapi_index_rst)
    autoapi_toc_placeholder = '@quark_autoapi_toc_placeholder@'
    with open(autoapi_index_rst, 'r') as f:
        quark_autoapi_toc = f.read().strip()

    generate_autoapi_docs = "QUARK_SPHINX_BUILD_SKIP_AUTOAPI" not in os.environ or os.environ["QUARK_SPHINX_BUILD_SKIP_AUTOAPI"].lower() in ("0", "false", "off")
    if not generate_autoapi_docs:
        quark_autoapi_toc = ""

    if autoapi_toc_placeholder in source[0]:  # Only process documents containing `autoapi_toc_placeholder`
        source[0] = source[0].replace(autoapi_toc_placeholder, quark_autoapi_toc)
        if len(quark_autoapi_toc) > 0:
            logger.info(f'Replaced {autoapi_toc_placeholder} with AutoAPI table of Content defined at {autoapi_index_rst} and index by {docname}.rst')
        else:
            logger.info(f'Removed {autoapi_toc_placeholder} from {docname}.rst')

def setup(app):
    """
    Setup Sphinx extension
    """
    app.connect('source-read', update_autoapi_toc_placeholder)
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
