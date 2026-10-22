import ast
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


REQUIRED_IMPORTS = {
    ("import", "os", None),
    ("from", "yolox_torch.yolox.data", "get_yolox_datadir", None),
    ("from", "yolox_torch.yolox.exp", "Exp", "MyExp"),
}

ALLOWED_IMPORTS = REQUIRED_IMPORTS | {
    ("from", "yolox_torch.yolox.data", "VOCDetection", None),
    ("from", "yolox_torch.yolox.data", "TrainTransform", None),
    ("from", "yolox_torch.yolox.data", "ValTransform", None),
    ("from", "yolox_torch.yolox.evaluators", "VOCEvaluator", None),
}

REQUIRED_METHOD_ARGUMENTS = {
    "get_dataset": {"cache"},
    "get_eval_dataset": set(),
    "get_evaluator": {"batch_size", "is_distributed"},
}

REQUIRED_METHOD_CALLS = {
    "get_dataset": {"VOCDetection", "TrainTransform"},
    "get_eval_dataset": {"VOCDetection", "ValTransform"},
    "get_evaluator": {"VOCEvaluator", "self.get_eval_loader"},
}

REQUIRED_RETURN_CALLS = {
    "get_dataset": "VOCDetection",
    "get_eval_dataset": "VOCDetection",
    "get_evaluator": "VOCEvaluator",
}

FORBIDDEN_FUNCTIONS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
}

FORBIDDEN_METHOD_CALLS = {
    "os.system",
    "os.popen",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.rename",
    "os.replace",
    "os.chmod",
    "os.chown",
    "shutil.rmtree",
}


def get_name(node: ast.AST) -> str | None:
    """
    Name / Attribute ノードを文字列に変換する。

    例:
    - VOCDetection
    - self.get_eval_loader
    - os.system
    """
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = get_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"

    return None


def import_to_string(import_item: tuple) -> str:
    """正規化済み import 情報を Python の import 文へ変換する。"""
    if import_item[0] == "import":
        _, module_name, alias = import_item

        if alias:
            return f"import {module_name} as {alias}"

        return f"import {module_name}"

    _, module_name, imported_name, alias = import_item

    if alias:
        return f"from {module_name} import {imported_name} as {alias}"

    return f"from {module_name} import {imported_name}"


def get_imports(tree: ast.Module) -> set[tuple]:
    """
    ファイル全体の import を収集する。

    クラスやメソッド内に書かれた import も収集対象にする。
    """
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(("import", alias.name, alias.asname))

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""

            for alias in node.names:
                imports.add(
                    ("from", module_name, alias.name, alias.asname)
                )

    return imports


def validate_imports(tree: ast.Module, result: ValidationResult) -> None:
    """必須 import と、許可されていない import を検証する。"""
    actual_imports = get_imports(tree)

    missing_imports = REQUIRED_IMPORTS - actual_imports
    unexpected_imports = actual_imports - ALLOWED_IMPORTS

    for import_item in sorted(missing_imports):
        result.errors.append(
            f"必須の import がありません: `{import_to_string(import_item)}`"
        )

    for import_item in sorted(unexpected_imports):
        result.errors.append(
            f"許可されていない import です: `{import_to_string(import_item)}`"
        )


def get_method_calls(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """メソッド内で呼び出される関数・メソッドを収集する。"""
    calls = set()

    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue

        function_name = get_name(node.func)

        if function_name:
            calls.add(function_name)

    return calls


def get_return_call_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """return 文で直接呼び出される関数・コンストラクタ名を収集する。"""
    return_calls = set()

    for node in ast.walk(method):
        if not isinstance(node, ast.Return):
            continue

        if not isinstance(node.value, ast.Call):
            continue

        function_name = get_name(node.value.func)

        if function_name:
            return_calls.add(function_name)

    return return_calls


def calls_parent_init(
    init_method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """
    以下のいずれかを検出する。

    super().__init__()
    super(Exp, self).__init__()
    """
    for node in ast.walk(init_method):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Attribute):
            continue

        if node.func.attr != "__init__":
            continue

        if (
            isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ):
            return True

    return False


def validate_exp_class(tree: ast.Module, result: ValidationResult) -> None:
    """Exp の継承、必須メソッド、および各メソッドの内容を検証する。"""
    exp_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Exp"
    ]

    if not exp_classes:
        result.errors.append(
            "トップレベルに `class Exp(MyExp)` が定義されていません。"
        )
        return

    if len(exp_classes) > 1:
        result.errors.append(
            "`class Exp(MyExp)` は 1 つだけ定義してください。"
        )
        return

    exp_class = exp_classes[0]

    base_class_names = {
        get_name(base_class)
        for base_class in exp_class.bases
    }

    if "MyExp" not in base_class_names:
        result.errors.append(
            "`Exp` クラスは `MyExp` を継承してください。"
        )

    methods = {
        node.name: node
        for node in exp_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    init_method = methods.get("__init__")

    if init_method is None:
        result.errors.append(
            "`Exp` クラスに `__init__()` が定義されていません。"
        )
    elif not calls_parent_init(init_method):
        result.errors.append(
            "`__init__()` 内で `super().__init__()` を呼び出してください。"
        )

    for method_name, required_arguments in REQUIRED_METHOD_ARGUMENTS.items():
        method = methods.get(method_name)

        if method is None:
            result.errors.append(
                f"Exp クラスに必要なメソッドがありません: `{method_name}`"
            )
            continue

        argument_names = {
            argument.arg
            for argument in method.args.args
        }

        missing_arguments = required_arguments - argument_names

        if missing_arguments:
            result.errors.append(
                "`{}` に必要な引数がありません: {}".format(
                    method_name,
                    ", ".join(sorted(missing_arguments)),
                )
            )

        method_calls = get_method_calls(method)
        missing_calls = REQUIRED_METHOD_CALLS[method_name] - method_calls

        if missing_calls:
            result.errors.append(
                "`{}` 内に必要な呼び出しがありません: {}".format(
                    method_name,
                    ", ".join(sorted(missing_calls)),
                )
            )

        return_calls = get_return_call_names(method)
        expected_return_call = REQUIRED_RETURN_CALLS[method_name]

        if expected_return_call not in return_calls:
            result.errors.append(
                "`{}` は `{}` を return してください。".format(
                    method_name,
                    expected_return_call,
                )
            )


def validate_top_level_statements(
    tree: ast.Module,
    result: ValidationResult,
) -> None:
    """
    import 時に実行されるトップレベル処理を拒否する。

    docstring、import、class 定義、代入だけを許可する。
    """
    for node in tree.body:
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )

        is_allowed_statement = isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        )

        if not is_docstring and not is_allowed_statement:
            result.errors.append(
                f"{node.lineno} 行目: "
                "トップレベルでの実行文は許可されていません。"
            )


def validate_dangerous_calls(
    tree: ast.Module,
    result: ValidationResult,
) -> None:
    """禁止された組み込み関数・ファイル操作・OS 操作を検出する。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function_name = get_name(node.func)

        if function_name in FORBIDDEN_FUNCTIONS:
            result.errors.append(
                f"{node.lineno} 行目: "
                f"`{function_name}()` は許可されていません。"
            )

        if function_name in FORBIDDEN_METHOD_CALLS:
            result.errors.append(
                f"{node.lineno} 行目: "
                f"`{function_name}()` は許可されていません。"
            )


def validate_exp_source(source: str) -> ValidationResult:
    """
    アップロードされた Exp Python ソースを実行せずに検証する。

    ast.parse() のみを利用するため、import、Exp の生成、
    関数の呼び出し、学習処理は実行されない。
    """
    result = ValidationResult()

    try:
        tree = ast.parse(source, filename="custom_exp.py")
    except SyntaxError as error:
        result.errors.append(
            f"構文エラー: {error.msg}（{error.lineno} 行目）"
        )
        return result

    validate_imports(tree, result)
    validate_exp_class(tree, result)
    validate_top_level_statements(tree, result)
    validate_dangerous_calls(tree, result)

    # if result.is_valid:
    #     result.warnings.append(
    #         "Exp ファイルの画面側検証を通過しました。"
    #     )
    #     result.warnings.append(
    #         "学習開始前に Docker コンテナ内で Exp の import・生成を検証してください。"
    #     )

    return result