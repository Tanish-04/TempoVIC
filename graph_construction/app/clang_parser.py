
import os
from typing import List, Optional, Tuple
import clang.cindex
from .unit import CUnit
import json
from config_loader import (
    get_libclang_path,
    get_c_standard,
    get_system_include_paths,
    get_project_root,
    get_project_compile_db,
)
from .compile_commands_supports import (
    ensure_compile_db_loaded,
    compile_db_by_sanitized,
    compile_db_by_path,
    generate_kernel_path_candidates,
    extract_compile_arguments,
)
from .clang_gen_units import ClangGenUnits


clang.cindex.conf.set_library_file(get_libclang_path())

class ClangParser:
    
    def __init__(self, file_path: str, line_window: Optional[Tuple[int, int, int]] = None):
        """
        Initialize parser and parse file.
        
        Args:
            file_path: Path to C source file
        """
        self.file_path = file_path
        self.file_content = []
        self.ast = None
        self.units = []
        self.line_window = line_window
        
        # Simply call the parse method
        self.parse_source()
    
    
    def parse_source(self):
        try:
            print(f"Parsing C file with Clang: {self.file_path}")
            
            # Step 1. Read file content
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.file_content = content.split("\n")

            compile_db_path = get_project_compile_db()

            ensure_compile_db_loaded(compile_db_path)

            # Step 3. Resolve normalized path via compile database mappings
            normalized_path = os.path.normpath(os.path.abspath(self.file_path))
            fname = os.path.basename(self.file_path)
            compile_entry = None

            sanitized_matches = compile_db_by_sanitized.get(fname)
            if sanitized_matches:
                compile_entry = sanitized_matches[0]
                normalized_path = compile_entry["resolved_source_path"]
                print(f"[PATH-MAP] {self.file_path} → {normalized_path} (sanitized mapping)")
            else:
                candidate_paths = [normalized_path]
                if "_" in fname:
                    parts = fname.split("_", 1)
                    base_dir = parts[0]
                    rest = parts[1]
                    for candidate in generate_kernel_path_candidates(base_dir, rest):
                        candidate_paths.append(candidate)

                tried_paths = []
                seen_paths = set()
                for candidate in candidate_paths:
                    norm_candidate = os.path.normpath(candidate)
                    if norm_candidate in seen_paths:
                        continue
                    seen_paths.add(norm_candidate)
                    tried_paths.append(norm_candidate)
                    entries = compile_db_by_path.get(norm_candidate)
                    if entries:
                        compile_entry = entries[0]
                        normalized_path = norm_candidate
                        if norm_candidate == os.path.normpath(os.path.abspath(self.file_path)):
                            print(f"[PATH-MAP] {self.file_path} → {normalized_path} (direct match)")
                        else:
                            print(f"[PATH-MAP] {self.file_path} → {normalized_path} (heuristic match)")
                        break

                if compile_entry is None:
                    raise ValueError(f"No compile entry found for file: {self.file_path}")

            # Step 4. Load compile args from selected entry
            compile_args = extract_compile_arguments(compile_entry)
            if not compile_args:
                raise ValueError(f"Compile entry missing arguments for file: {normalized_path}")

            # Step 4a. Filter GCC-incompatible flags before passing to libclang
            import re

            # Matches -Wfoo=5, -Wimplicit-fallthrough=3, etc. (GCC numeric warning levels)
            _W_NUMERIC = re.compile(r'^-W[^=]+=\d+$')

            # GCC-only flag prefixes (startswith check)
            _GCC_PREFIXES = (
                "-Werror",                        # turns warnings → errors; kills libclang parse
                "-Wno-",                          # GCC-specific suppressions clang may not know
                "--param=",                       # GCC tuning params
                "--param",                        # --param key=val (two-token form)
                "-fplugin=",                      # GCC plugins
                "-fplugin-arg-",
                "-mpreferred-stack-boundary",
                "-mindirect-branch",
                "-mfunction-return",
                "-mrecord-mcount",
                "-mfentry",
                "-mtraceback",
                "-fpatchable-function-entry",
                "-fno-allow-store-data-races",
                "-fconserve-stack",
                "-fno-code-hoisting",
                "-fno-var-tracking-assignments",
                "-femit-struct-debug-baseonly",
                "-fno-tree-loop-distribute-patterns",
                "-fno-reorder-blocks-and-partition",
                "-fno-ipa-cp-clone",
                "-fno-partial-inlining",
                "-fno-gcse",
                "-fno-devirtualize-speculatively",
                "-fno-fat-lto-objects",
                "-fuse-ld=",
                "-fno-stack-clash-protection",
                "-fno-jump-tables",
                "-Wa,",                           # assembler passthrough flags
                "-Wbad-function-cast",
                "-Wdeclaration-after-statement",
            )

            # GCC-only exact flags
            _GCC_EXACT = {
                "-mindirect-branch-register",
                "-mindirect-branch-cs-prefix",
                "-mrecord-mcount",
            }

            filtered_args = []
            removed_args = []
            skip_next = False

            for i, arg in enumerate(compile_args):
                if skip_next:
                    skip_next = False
                    continue

                # Two-token --param key=val
                if arg == "--param":
                    skip_next = True
                    removed_args.append(arg)
                    continue

                # GCC numeric warning level  e.g. -Wimplicit-fallthrough=5
                if _W_NUMERIC.match(arg):
                    # Strip the =N and keep the base warning (clang understands it)
                    base = arg[:arg.rindex("=")]
                    filtered_args.append(base)
                    removed_args.append(f"{arg} → {base}")
                    continue

                # Exact match
                if arg in _GCC_EXACT:
                    removed_args.append(arg)
                    continue

                # Prefix match
                if any(arg.startswith(pfx) for pfx in _GCC_PREFIXES):
                    removed_args.append(arg)
                    continue

                filtered_args.append(arg)

            if removed_args:
                print(f"[FLAG-FILTER] Removed/adjusted {len(removed_args)} GCC-incompatible flags")

            compile_args = filtered_args

            # Step 5. Convert -I and -include paths to absolute
            kernel_root = get_project_root()
            abs_args = []
            i = 0
            while i < len(compile_args):
                a = compile_args[i]
                if a.startswith("-I") and len(a) > 2 and not os.path.isabs(a[2:]):
                    abs_args.append(f"-I{os.path.join(kernel_root, a[2:])}")
                    i += 1
                elif a == "-include" and (i + 1) < len(compile_args):
                    rel = compile_args[i + 1]
                    abs_args.append(a)
                    abs_args.append(rel if os.path.isabs(rel) else os.path.join(kernel_root, rel))
                    i += 2
                else:
                    abs_args.append(a)
                    i += 1
            compile_args = abs_args

            # Step 6. Build final args: keep only -I / -D / -include, then add essentials
            essential_args = []
            for a in compile_args:
                if a.startswith("-I") or a.startswith("-D") or a.startswith("-include"):
                    essential_args.append(a)

            essential_args.extend([
                "-w",                          # silence all remaining warnings
                f"-std={get_c_standard()}",
                "-fno-builtin",
            ])
            # Append system include paths
            essential_args.extend([f"-I{p}" for p in get_system_include_paths()])

            compile_args = essential_args
            
            import clang.cindex
            index = clang.cindex.Index.create()
            options = clang.cindex.TranslationUnit.PARSE_INCOMPLETE

            # old_cwd = os.getcwd()
            # os.chdir(kernel_root)
            # try:
            #     self.ast = index.parse(
            #         self.file_path,
            #         args=compile_args,
            #         options=options,
            #         unsaved_files=[(self.file_path, content)],
            #     )
            # finally:
            #     os.chdir(old_cwd)
            # Use absolute path so libclang can match unsaved_files correctly
            # (chdir is no longer needed since all -I/-include paths are absolute)
            abs_file_path = os.path.abspath(self.file_path)
            self.ast = index.parse(
                abs_file_path,
                args=compile_args,
                options=options,
                unsaved_files=[(abs_file_path, content)],
            )
                
            if self.ast is None:
                raise RuntimeError("Clang returned null AST (index.parse returned None)")
            if self.ast.cursor:
                clang_gen_units = ClangGenUnits(self.file_content, self.units, self.ast, line_window=self.line_window)
                self.units = clang_gen_units.get_all_units()
                
        except Exception as e:
            print(f"\n❌ [ERROR] Failed to parse {self.file_path}: {e}")
            self.ast = None
            self.units = []
            try:
                from ParsingTracker import get_tracker
                get_tracker().record_parse_failure(self.file_path, str(e))
            except:
                pass

    def get_all_units(self) -> List[CUnit]:
        return self.units
