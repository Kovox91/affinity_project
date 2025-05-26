#!/bin/bash
if [[ "`uname`" == Darwin ]]; then
    PYMOL_APP="$PREFIX/PyMOL.app"
    rm -rf "$PYMOL_APP"
fi
