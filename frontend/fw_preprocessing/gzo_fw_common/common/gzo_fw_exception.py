#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
**作成日:** 2021/05/25
**作成者:** 成田　伊織 \n

ユーザー定義例外クラスを定義する
"""


class ImageReadError(Exception):
    """
    画像の読み込みに失敗した場合に発生
    """

    pass


class RatioValueError(Exception):
    """
    引数に渡す比率の値が不正な場合に発生
    """

    pass


class GetConfigException(Exception):
    """
    共通設定からの変数の呼び出しに失敗した場合に発生
    """

    pass
