#!/usr/bin/env python3
# vim: set et sw=4 sts=4 fileencoding=utf-8:
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)


import argparse
import sys
import os

from mako.template import Template, exceptions
from mako.lookup import TemplateLookup


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='the template file to render')
    parser.add_argument('-o', '--output', help='the output file')
    parser.add_argument('--boilerplate-dir', help='where to look for additional template definitions for common functions', required=True)
    parser.add_argument('--target-os', help='the target operating system to generate kernels for', required=True)

    args, extras = parser.parse_known_args(argv[1:])

    with open(args.path, 'r') as f:
        # look for templates in the locations given in the "path" and "boilerplate-dir" arguments
        template_lookup = TemplateLookup(directories=[os.path.dirname(args.path), args.boilerplate_dir])
        template = Template(f.read(), strict_undefined=True, lookup=template_lookup)

    try:
        output = template.render(target_os=args.target_os, args=extras)
    except:
        print(exceptions.text_error_template().render())
        return 1

    if (args.output):
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
