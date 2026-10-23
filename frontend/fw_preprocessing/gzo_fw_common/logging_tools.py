#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
**作成日:** 2021/05/25
**作成者:** 成田　伊織 \n

ログ出力の共通処理

"""
from fw_preprocessing.gzo_fw_common.kqi_processing import input_handler
import pathlib
import logging
import os
from datetime import datetime
from fw_preprocessing.gzo_fw_common.common.gzo_fw_exception import GetConfigException
from logging.handlers import TimedRotatingFileHandler
from fw_preprocessing.gzo_fw_common.common.common_tools import load_common_config


# ログレベルの名称(String)とPython設定変数(int)のマッピング
LOG_LEVEL_DICT = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

# ログ設定ステータス定義
LOG_SETTING_STATUS_DICT = {
    "ini": "(i) common_config.ini",
    "py": "(ii) default in source (py)",
}

# デフォルトのTimeRotatingFileHandlerの引数
DEFAULT_HANDLER_FILENAME = "/kqi/output/log"
DEFAULT_HANDLER_WHEN = "D"
DEFAULT_HANDLER_INTERVAL = 1
DEFAULT_HANDLER_BACKUP_COUNT = 30
DEFAULT_HANDLER_ENCODING = "utf-8"

# デフォルトのログレベル
DEFAULT_LOG_LEVEL = logging.INFO

# ログフォーマット
LOG_FORMAT = "%(asctime)s <%(levelname)s> %(message)s by %(funcName)s at line %(lineno)d of %(name)s"
# 日付フォーマット
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
# ログローテーション拡張子
LOG_SUFFIX = "%Y-%m-%d_%H-%M-%S"


def set_logger(log_output_dir_path: str = None, config_filename=None):
    """
    ログの標準出力フォーマットを設定する。
    ログファイルの出力及びローテーションの設定を行う。

    Returns:
        Logger: logger -- python標準のロガーオブジェクト
    """

    # common_config.iniからユーザ設定を読み込む
    try:
        # cfg設定適応ステータスに設定
        log_setting_status = "ini"
        message = "common_config.ini normal setting"

        # 共通設定の読込が完了するまではデフォルトフラグを立てる
        use_default_setting = True

        # 共通設定を読み込む
        try:
            if config_filename is None:
                common_config = load_common_config()
            else:
                common_config = input_handler.load_config(config_filename)
        except FileNotFoundError:
            raise

        # 共通設定から変数を取得
        try:
            output_dir_path = (
                str(pathlib.Path(log_output_dir_path))
                if log_output_dir_path is not None
                else str(os.getcwd())
            )
            log_file_stem = pathlib.Path(common_config.get("logging", "log_file_name"))
            when = common_config.get("logging", "log_when")
            interval = int(common_config.get("logging", "log_interval"))
            backupCount = int(common_config.get("logging", "log_backup_count"))
            log_level_name = common_config.get("logging", "log_level")
        except (ValueError, NameError):
            # log_whenに想定外の文字列が設定されている
            raise
        except Exception as e:
            # その他、変数取得時のエラー
            raise GetConfigException(e)

        # 共通設定から出力ディレクトリを取得
        output_dir = pathlib.Path(output_dir_path)

        # 存在チェック
        if output_dir.exists() is False:
            os.makedirs(str(output_dir))

        # ディレクトリチェック
        if output_dir.is_dir() is False:
            raise NotADirectoryError("ディレクトリではありません：" + str(output_dir))

        # ログファイル名を作成
        filename = str(output_dir / log_file_stem)

        # ログレベルオブジェクトの取得
        log_level = LOG_LEVEL_DICT[log_level_name]

        # ログローテーションの設定を行う
        rotation_handler = TimedRotatingFileHandler(
            filename, when, interval, backupCount, encoding="utf-8"
        )

        # cfg設定が完了したらデフォルトフラグは立てない
        use_default_setting = False

    # エラーを取得した場合メッセージを設定する
    except GetConfigException as e:
        # iniファイルから変数の取得に失敗した場合
        message = "iniファイルから変数を取得できません：" + str(e)
    except FileNotFoundError as e:
        # 共通設定ファイルを読み込むことができない
        # ログ出力先に存在しないディレクトリが指定されている
        message = e
    except NotADirectoryError as e:
        # ログ出力先にファイルが指定されている
        message = e
    except LookupError as e:
        # ログレベルがLOG＿LEVEL_DICTのキー以外である
        message = "キーを取得できません：" + str(e)
    except (ValueError, NameError) as e:
        # log_whenに想定外の文字列が設定されている
        message = "想定外の値が設定されています：" + str(e)

    finally:
        # common_config.iniが読み込めない場合は定数定義のデフォルト設定とする
        if use_default_setting:
            log_setting_status = "py"
            rotation_handler = rotation_handler = TimedRotatingFileHandler(
                filename=DEFAULT_HANDLER_FILENAME,
                when=DEFAULT_HANDLER_WHEN,
                interval=DEFAULT_HANDLER_INTERVAL,
                backupCount=DEFAULT_HANDLER_BACKUP_COUNT,
                encoding=DEFAULT_HANDLER_ENCODING,
            )
            log_level = DEFAULT_LOG_LEVEL

        # ハンドラーのフォーマットを設定
        rotation_handler.setFormatter(logging.Formatter(LOG_FORMAT))

        # ハンドラーの拡張子を設定
        rotation_handler.suffix = LOG_SUFFIX

        # ログフォーマット設定
        logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT)

        # ロガー取得
        logger = logging.getLogger("log")

        # ハンドラーを追加
        logger.addHandler(rotation_handler)

        # ログレベル設定
        logger.setLevel(log_level)

        # 適応したログ設定の表示
        logger.info(
            "Log setting from {0} because {1}".format(
                LOG_SETTING_STATUS_DICT[log_setting_status], message
            )
        )

        return logger


class FWTimer:
    """
    処理時間の計測と、計測結果の出力を行うクラス\n
        How to use:\n
            FWTimerインスタンスの取得：fw_timer = FWTimer(logger) \n
            タグ名に紐づいた処理時間計測の開始：fw_timer.start("任意のタグ名") \n
            タグ名に紐づいた処理時間計測の終了：fw_timer.end("任意のタグ名") \n
        ※ end()を実行すると、タグに紐づいた処理時間計測の記録は削除される

    Attributes:
        timer_dict (dict): タイマー辞書 -- タグ名キーとして、タグ名ごとに処理の開始時間を保持する辞書
        logger (Logger): python標準のロガーオブジェクト

    """

    def __init__(self, logger):
        """
        コンストラクタ \n
        タイマー辞書の初期化とロガーの設定を行う

        Args:
            logger (Logger): python標準のロガーオブジェクト

        Raises:
            TypeError: 引数loggerにlogging.Loggerクラスではないオブジェクト設定されている場合に発生
        """

        self.timer_dict = {}
        self.logger = logger

        # 型チェック
        if type(logger) is not logging.Logger:
            raise TypeError(
                "引数loggerにlogging.Loggerクラスではないオブジェクト設定されています：" + str(logger)
            )

    def start(self, tag):
        """
        処理時間の計測を開始する

        Args:
            tag (str): 処理時間計測のタグ名 (タグ名ごとに異なる処理時間の計測を実行可能)

        Raises:
            TypeError: 引数tagにはstr型を以外が指定されている場合に発生
        """
        # 型チェック
        if type(tag) is not str:
            raise TypeError("引数tagにはstr型を指定してください：" + str(tag))

        # タイマー辞書に処理開始時間を設定
        self.timer_dict[tag] = datetime.now()

        # ログ出力
        self.logger.info("[計測開始] tag: " + str(tag))

    def end(self, tag):
        """
        処理時間の計測を終了し、タイマー辞書からタグに紐づいた記録を削除する

        Args:
            tag (str): 処理時間計測のタグ名 (タグ名ごとに異なる処理時間の計測を実行可能)

        Raises:
            TypeError: 引数tagにはstr型を以外が指定されている場合に発生
            KeyError: 引数tagに指定したタグに対応する計測結果が存在しない場合に発生
        """
        end_time = datetime.now()

        # 型チェック
        if type(tag) is not str:
            raise TypeError("引数tagにはstr型を指定してください：" + str(tag))

        # タイマー辞書からタグに関連する処理開始時間を取得
        try:
            start_time = self.timer_dict[tag]
            del self.timer_dict[tag]

        # 引数tagに指定したタグに対応する計測結果が存在しない場合
        except KeyError:
            raise KeyError("引数tagに指定したタグに対応する計測結果がありません：" + str(tag))

        # 処理時間を取得
        duration = end_time - start_time

        # ログ出力
        self.logger.info(
            "[計測終了] tag: {},start: {} end: {}, duration: {}".format(
                str(tag), str(start_time), str(end_time), str(duration)
            )
        )
