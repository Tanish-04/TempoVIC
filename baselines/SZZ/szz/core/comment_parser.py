"""
Comment parser for C/C++ source files.

Uses srcML to identify comment line ranges, which are then excluded
from blame results by the SZZ implementations.
"""
import logging as log
import os
import re
import subprocess
from collections import namedtuple
import tempfile

CommentRange = namedtuple('CommentRange', 'start end')
srcml_file_ext = ['.c', '.h', '.hh', '.hpp', '.hxx', '.cxx', '.cpp', '.cc']


def parse_comments(file_str: str, file_name: str, temp_dir: str = tempfile.gettempdir()):
    """
    Parse comment ranges from a C/C++ source file using srcML.

    :param file_str: content of the source file
    :param file_name: name of the source file (used for extension detection)
    :param temp_dir: temporary directory for srcML processing
    :returns: list of CommentRange(start, end) namedtuples
    """
    return parse_comments_srcml(file_str, file_name, temp_dir)


def parse_comments_srcml(file_str: str, file_name: str, temp_folder: str = tempfile.gettempdir()):
    line_comment_ranges = list()

    if any(file_name.endswith(e) for e in srcml_file_ext):
        if not os.path.isdir(temp_folder):
            os.makedirs(temp_folder)

        temp_file_path = os.path.join(temp_folder, 'temp_' + file_name)
        with open(temp_file_path, 'w', encoding='utf-8', errors='ignore') as temp_file:
            temp_file.write(file_str)

        process_out = list()
        p = subprocess.Popen(
            f'srcml --position {temp_file_path}',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        for line in p.stdout.readlines():
            process_out.append(line.decode('utf-8').strip())
        status = p.wait()

        if status == 0:
            for line in process_out:
                if line.strip().startswith("<comment"):
                    start_match = re.search(r'pos:start="(\d+):', line)
                    end_match = re.search(r'pos:end="(\d+):', line)
                    if start_match and end_match:
                        line_comment_ranges.append(
                            CommentRange(
                                start=int(start_match.group(1)),
                                end=int(end_match.group(1))
                            )
                        )
        else:
            log.error(process_out)

        if os.path.isfile(temp_file_path):
            os.remove(temp_file_path)
    else:
        log.error(f"file not supported by srcML: {file_name}")

    return line_comment_ranges
