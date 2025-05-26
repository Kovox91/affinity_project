#!/bin/bash
if [[ "`uname`" == Darwin ]]; then
    PYMOL_APP="$PREFIX/PyMOL.app"

    rm -rf "$PYMOL_APP"

    # allow suffix (e.g. version number) in app name
    # allow leading underscore on "Contents"
    # e.g.: foo/PyMOL2.app/_Contents
    if [[ "$PREFIX" != */PyMOL*.app/*Contents ]]; then
        mv "$PREFIX/PyMOL_app" "$PYMOL_APP"

        cd "$PYMOL_APP/Contents/MacOS"

        rel_prefix=../..
    else
        cd "$PREFIX"

        # beware of case-sensitive file systems
        mv resources Resources

        rsync -avuP PyMOL_app/Contents/* .
        rm -rf PyMOL_app

        cd MacOS

        rel_prefix=.
    fi

    # python symlink
    ln -sf ../$rel_prefix/bin/python .

    # backwards compatibility
    ln -sf ../$rel_prefix/bin/pymol MacPyMOL

fi
