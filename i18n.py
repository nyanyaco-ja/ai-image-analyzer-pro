"""
多言語対応（i18n）マネージャー

使い方:
    from i18n import I18n

    i18n = I18n()
    print(i18n.t('app.title'))  # AI Image Analyzer Pro

    i18n.set_language('ja')
    print(i18n.t('app.title'))  # AI画像分析ツール Pro
"""

import json
from pathlib import Path


class I18n:
    """多言語対応マネージャー"""

    def __init__(self, default_language='ja'):
        """
        初期化

        Args:
            default_language: デフォルト言語（'ja' or 'en'）
        """
        self.current_language = default_language
        self.translations = {}
        self.locales_dir = Path(__file__).parent / 'locales'

        # 翻訳ファイルをロード
        self.load_translations()

    def load_translations(self):
        """翻訳ファイルをロード"""
        for lang_file in self.locales_dir.glob('*.json'):
            lang_code = lang_file.stem  # 'en' or 'ja'
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {lang_file}: {e}")

    def set_language(self, lang_code):
        """
        言語を切り替え

        Args:
            lang_code: 'en' or 'ja'
        """
        if lang_code in self.translations:
            self.current_language = lang_code
            return True
        else:
            print(f"Warning: Language '{lang_code}' not found")
            return False

    def get_language(self):
        """現在の言語コードを取得"""
        return self.current_language

    def t(self, key, **kwargs):
        """
        翻訳テキストを取得

        Args:
            key: 翻訳キー（例: 'app.title', 'tabs.single_analysis'）
            **kwargs: プレースホルダーの置換値

        Returns:
            翻訳されたテキスト

        Examples:
            i18n.t('app.title')
            i18n.t('messages.processing', count=100)
        """
        # キーを'.'で分割してネストされた辞書を辿る
        keys = key.split('.')
        value = self.translations.get(self.current_language, {})

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = None
                break

        # 見つからない場合はキー自体を返す
        if value is None:
            # フォールバック: 英語を試す
            if self.current_language != 'en':
                value = self.translations.get('en', {})
                for k in keys:
                    if isinstance(value, dict):
                        value = value.get(k)
                    else:
                        value = None
                        break

            if value is None:
                return f"[{key}]"

        # プレースホルダーを置換
        if kwargs and isinstance(value, str):
            try:
                value = value.format(**kwargs)
            except KeyError as e:
                print(f"Warning: Missing placeholder {e} in key '{key}'")

        return value

    def get_available_languages(self):
        """利用可能な言語のリストを取得"""
        return list(self.translations.keys())

    def get_language_name(self, lang_code):
        """言語コードから言語名を取得"""
        names = {
            'en': 'English',
            'ja': '日本語'
        }
        return names.get(lang_code, lang_code)


# グローバルインスタンス（シングルトン）
_global_i18n = None

def get_i18n(default_language='ja'):
    """グローバルi18nインスタンスを取得"""
    global _global_i18n
    if _global_i18n is None:
        _global_i18n = I18n(default_language)
    return _global_i18n
