import os
import re
import ast
import chardet
import subprocess
from argparse import ArgumentTypeError
from git import Repo
from pathlib import Path
from tempfile import TemporaryDirectory

import traceback
import networkx as nx
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from lib2to3.pgen2.parse import ParseError

# This is to handle python2 code
fixers = get_fixers_from_package('lib2to3.fixes')
refactorer = RefactoringTool(fixers)


DIFF_PATTERN = re.compile(r"^diff(?:.*)")
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),(\d+)\s+\+(\d+),(\d+)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)


def get_first_idx(charlist):
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)


def get_last_idx(charlist):
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1


def strip_content(hunk):
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split("\n")))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split("\n")[first_idx:last_idx]))
    new_hunk = "\n" + "\n".join(new_lines) + "\n"
    return new_hunk, first_idx - 1


def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
    stats = {"context": 0, "added": 0, "subtracted": 0}
    hunk = hunk.split("\n", 1)[-1].strip("\n")
    for line in hunk.split("\n"):
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    pre_len = context + subtracted
    post_start = pre_start + total_delta
    post_len = context + added
    total_delta = total_delta + (post_len - pre_len)
    return pre_start, pre_len, post_start, post_len, total_delta


def repair_patch(model_patch):
    if model_patch is None:
        return None
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        diff_header = DIFF_PATTERN.findall(patch)
        if diff_header:
            new_patch += diff_header[0] + "\n"
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                *list(map(lambda x: int(x) if x.isnumeric() else x, hunk)), total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


def extract_minimal_patch(model_patch):
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        diff_header = DIFF_PATTERN.findall(patch)
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]


def is_test(name, test_phrases=None):
    if test_phrases is None:
        test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", name.lower()))
    return any(word in words for word in test_phrases)


class ContextManager:
    def __init__(self, repo_path, base_commit, verbose=False):
        self.repo_path = Path(repo_path).resolve().as_posix()
        self.old_dir = os.getcwd()
        self.base_commit = base_commit
        self.verbose = verbose

    def __enter__(self):
        os.chdir(self.repo_path)
        cmd = f"git reset --hard {self.base_commit} && git clean -fdxq"
        if self.verbose:
            subprocess.run(cmd, shell=True, check=True)
        else:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return self

    def get_environment(self):
        raise NotImplementedError()  # TODO: activate conda environment and return the environment file

    def get_readme_files(self):
        files = os.listdir(self.repo_path)
        files = list(filter(lambda x: os.path.isfile(x), files))
        files = list(filter(lambda x: x.lower().startswith("readme"), files))
        return files

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.old_dir)


class AutoContextManager(ContextManager):
    """Automatically clones the repo if it doesn't exist"""

    def __init__(self, instance, root_dir=None, verbose=False, token=None):
        if token is None:
            token = os.environ.get("GITHUB_TOKEN", "git")
        self.tempdir = None
        if root_dir is None:
            self.tempdir = TemporaryDirectory()
            root_dir = self.tempdir.name
        self.root_dir = root_dir
        repo_dir = os.path.join(self.root_dir, instance["repo"].replace("/", "__"))
        if not os.path.exists(repo_dir):
            repo_url = (
                f"https://{token}@github.com/swe-bench-repos/"
                + instance["repo"].replace("/", "__")
                + ".git"
            )
            if verbose:
                print(f"Cloning {instance['repo']} to {root_dir}")
            Repo.clone_from(repo_url, repo_dir)
        super().__init__(repo_dir, instance["base_commit"], verbose=verbose)
        self.instance = instance

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tempdir is not None:
            self.tempdir.cleanup()
        return super().__exit__(exc_type, exc_val, exc_tb)


def get_imported_modules(filename):
    with open(filename) as file:
        tree = ast.parse(file.read(), filename)
    return [
        node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def resolve_module_to_file(module, level, root_dir):
    components = module.split(".")
    if level > 0:
        components = components[:-level]
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath.endswith(os.sep.join(components)):
            return [
                os.path.join(dirpath, filename)
                for filename in filenames
                if filename.endswith(".py")
            ]
    return []


def ingest_file_directory_contents(target_file, root_dir):
    imported_files = []
    files_to_check = [target_file]
    while files_to_check:
        current_file = files_to_check.pop()
        imported_files.append(current_file)
        imports = get_imported_modules(current_file)
        for node in imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    files = resolve_module_to_file(alias.name, 0, root_dir)
                    for file in files:
                        if file not in imported_files and file not in files_to_check:
                            files_to_check.append(file)
            elif isinstance(node, ast.ImportFrom):
                files = resolve_module_to_file(node.module, node.level, root_dir)
                for file in files:
                    if file not in imported_files and file not in files_to_check:
                        files_to_check.append(file)
    return imported_files


def detect_encoding(filename):
    """
    Detect the encoding of a file
    """
    with open(filename, "rb") as file:
        rawdata = file.read()
    return chardet.detect(rawdata)["encoding"]


def list_files(root_dir, include_tests=False):
    files = []
    for filename in Path(root_dir).rglob("*.py"):
        if not include_tests and is_test(filename.as_posix()):
            continue
        files.append(filename.relative_to(root_dir).as_posix())
    return files


def ingest_directory_contents(root_dir, include_tests=False):
    files_content = {}
    for relative_path in list_files(root_dir, include_tests=include_tests):
        filename = os.path.join(root_dir, relative_path)
        encoding = detect_encoding(filename)
        if encoding is None:
            content = "[BINARY DATA FILE]"
        else:
            try:
                with open(filename, encoding=encoding) as file:
                    content = file.read()
            except (UnicodeDecodeError, LookupError):
                content = "[BINARY DATA FILE]"
        files_content[relative_path] = content
    return files_content


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
    
def build_python_dependency_graph(source_dir:Path):
    """
    Build a file-level import dependency graph by manually parsing AST,
    automatically detecting top-level roots (e.g., 'src', 'lib', etc.),
    and mapping absolute imports to the correct directory.
    """
    project_root = Path(source_dir).resolve()

    # Detect candidate roots: project_root plus any immediate subdir containing .py files
    roots = [project_root]
    for sub in project_root.iterdir():
        if sub.is_dir() and any(sub.rglob("*.py")):
            roots.append(sub)
    # Deduplicate while preserving order
    seen = set()
    roots = [r for r in roots if not (r in seen or seen.add(r))]

    G = nx.DiGraph()

    # Add all Python files as nodes
    py_files = list(project_root.rglob("*.py"))
    for py_file in py_files:
        rel_path = py_file.relative_to(project_root).as_posix()
        G.add_node(rel_path)

    # Parse imports and add edges
    for py_file in py_files:
        try:
            rel_src = py_file.relative_to(project_root).as_posix()
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(rel_src))
        except SyntaxError as err:
            print(f"Syntax error in {rel_src}:{err.lineno}:{err.offset} - {err.msg!r}")
            if (err.text):
                print(err.text.rstrip())
                print(" " + str((err.offset - 1)) + "^")

            try:
                refactored = refactorer.refactor_string(source, name=rel_src)
            except ParseError as pe:
                print(f"Refactor parse error {rel_src} - {pe!r}")
                continue
            except Exception as e_ref:
                print(f"Refactor error: {rel_src} - {type(e_ref).__name__}: {e_ref}")
                traceback.print_exc()
                continue
            try:
                # Sometimes it's a py2 vs py3 error, try to fix
                tree = ast.parse(str(refactored), filename=rel_src)
            except Exception as e:
                # Skip files that fail to parse
                print(f"Could not parse {rel_src} after refactoring, got: {e!r}")
                continue
        except Exception as exc:
            # This can happen on files that point to files that don't exist (e.g. __init__.py)
            print(f"Got unexpected error while attempting to parse {rel_src}: {exc!r}")

        for node in ast.walk(tree):
            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                module_names = [base] + [f"{base}.{alias.name}" for alias in node.names] if base else [alias.name for alias in node.names]
            else:
                continue

            for mod in module_names:
                if not mod:
                    continue
                parts = mod.split(".")
                # Try each root for mapping
                for root in roots:
                    candidate_file = root.joinpath(*parts).with_suffix(".py")
                    rel_dst = candidate_file.relative_to(project_root).as_posix()
                    if candidate_file.exists():
                        G.add_edge(rel_src, str(rel_dst))
                        break
                    candidate_pkg = root.joinpath(*parts) / "__init__.py"
                    rel_dst = candidate_pkg.relative_to(project_root).as_posix()
                    if candidate_pkg.exists():
                        G.add_edge(rel_src, str(rel_dst))
                        break

    return G