#!/bin/bash
##
## Make relative (portable) python shebangs
##

PREFIX="`dirname "$0"`"

echo_usage() {
    echo "Usage: $0 [<directory>] [<python>]"
}

case "$1" in
    -h|--help)
        echo_usage
        echo ""
        echo "Makes Python scripts under <directory> portable by replacing"
        echo 'the absolute #!$PREFIX/bin/python shebang with something relative.'
        echo ""
        echo "Defaults:"
        echo '  directory: $(dirname $0)/bin'
        echo '  python:    $(dirname $0)/bin/python'
        exit 1
        ;;
    "")
        bindirs=("$PREFIX/bin" "$PREFIX/condabin")
        ;;
    *)
        bindirs=("$1")
        ;;
esac

if [ -z "$2" ]; then
    pythonarg="$PREFIX/bin/python"
else
    pythonarg="$2"
fi

if [ ! -f "$pythonarg" ]; then
    echo "Error: no such file: $pythonarg"
    echo_usage
    exit 2
fi

relpath() {
    python -Sc 'import os, sys;print(os.path.relpath(*sys.argv[1:]))' "$@"
}

patch_python_shebang() {
    filename="$1"
    if head -n 1 "$filename" | grep -q '^#!.*\<python'; then
        echo "patching shebang for $filename"
        python="$(relpath "$pythonarg" "`dirname "$filename"`")"
        temp="`mktemp /tmp/tmp_XXXXXXXX`"
        cat << EOF > "$temp"
#!/bin/sh
"unset" "PYTHONHOME" "QT_PLUGIN_PATH"
"exec" """\`python -Sc "print(__import__('os').path.realpath(r'\$0' '/..'))" || dirname "\$0"\`/$python""" "\$0" "\$@"
EOF
        tail -n +2 "$filename" >> "$temp"
        cat "$temp" > "$filename"
        rm -f "$temp"
    fi
}

find "${bindirs[@]}" -type f | while read f; do
    patch_python_shebang "$f"
done

# if called with arguments, stop here
if [ -n "$1" ]; then
    exit 0
fi

# fix conda 4.4 activate script
for f in \
    bin/activate \
    bin/deactivate \
    etc/profile.d/conda.sh \
    ; do
    up=`echo $f|sed 's#[^/]##g;s#/#/..#g'`
    f="$PREFIX/$f"
    if [ -f "$f" ]; then
        sed -i.bak \
            -e 's#^_CONDA_ROOT=.*#_CONDA_ROOT="$(cd "`dirname "$BASH_SOURCE"`" \&\& pwd)'$up'"#' \
            -e 's#CONDA_EXE=.*#CONDA_EXE="$(cd "`dirname "$BASH_SOURCE"`" \&\& pwd)'$up'/bin/conda"#' \
            -e 's#(PS1=".PS1"#(_PS1="$PS1"#' \
            "$f"
        rm -f "$f.bak"
    fi
done
for f in "$PREFIX"/lib/python*/site-packages/conda/activate.py; do
    if [ -f "$f" ]; then
        sed -i.bak \
            -e 's#\(= self.environ.get(.PS1., \)..)#\1self.environ.get("_PS1", ""))#' \
            "$f"
        rm -f "$f.bak"
    fi
done

# fix Qt (if called without arguments)
if [ -z "$1" -a -n "`ls $PREFIX/lib/libQt*`" ]; then
    echo -e "[Paths]\nPrefix = ..\nHeaders = include/qt" > \
        "$PREFIX/bin/qt.conf"

    echo -e '#!/bin/sh\nexec "`dirname "$0"`/python" -m PyQt4.uic.pyuic ${1+"$@"}' > \
        "$PREFIX/bin/pyuic4"

    echo -e '#!/bin/sh\nexec "`dirname "$0"`/python" -m PyQt5.uic.pyuic ${1+"$@"}' > \
        "$PREFIX/bin/pyuic5"

    find "$PREFIX" -name "*.app" | while read app; do
        Contents="$app/Contents"
        test -d "$Contents" || continue
        mkdir -p "$Contents/Resources"
        p="$(relpath "$PREFIX" "$Contents")"
        echo "$Contents/Resources/qt.conf Prefix = $p"
        echo -e "[Paths]\nPrefix = $p\nHeaders = include/qt" > \
            "$Contents/Resources/qt.conf"
    done

    for f in "$PREFIX"/lib/python*/site-packages/PyQt*/__init__.py; do
        if [ "`uname`" = Linux -a -f "$f" ]; then
            cat >> "$f" <<EOF
import os, sys
os.environ['QT_XKB_CONFIG_ROOT'] = sys.prefix + '/lib'
os.environ['FONTCONFIG_FILE'] = sys.prefix + '/etc/fonts/fonts.conf'
os.environ['FONTCONFIG_PATH'] = sys.prefix + '/etc/fonts'
del os, sys
EOF
        fi
    done
fi

# ssl
for f in "$PREFIX"/lib/python*/ssl.py; do
    if [ -f "$PREFIX/ssl/cert.pem" -a -f "$f" ]; then
        cat >> "$f" <<EOF
os.environ['SSL_CERT_FILE'] = sys.prefix + '/ssl/cert.pem'
EOF
    fi
done
