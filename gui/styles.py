"""Единая визуальная тема главного окна ChatKodi.

Стили строятся из ``core.theme_manager.theme_palette``. Это важно: лаунчер,
экран загрузки и главное окно больше не хранят разные наборы цветов.
"""

from __future__ import annotations

from core.theme_manager import normalize_theme, theme_palette


def load_styles(theme: str = "classic") -> str:
    theme = normalize_theme(theme)
    p = theme_palette(theme)
    return f"""
    * {{ font-family: 'Onest', 'Segoe UI', sans-serif; outline: 0; }}
    QMainWindow, #centralWidget {{ background-color: {p['bg']}; color: {p['fg']}; }}
    QWidget {{ color: {p['fg']}; }}

    QMenuBar {{ background-color: {p['sunken']}; color: {p['fg']}; border-bottom: 1px solid {p['border_soft']}; }}
    QMenuBar::item {{ padding: 6px 12px; background: transparent; border-radius: 8px; }}
    QMenuBar::item:selected {{ background: {p['accent_bg']}; color: {p['title']}; }}
    QMenu {{ background-color: {p['panel']}; color: {p['fg']}; border: 1px solid {p['border']}; border-radius: 10px; padding: 6px; }}
    QMenu::item {{ padding: 8px 12px; border-radius: 8px; }}
    QMenu::item:selected {{ background: {p['accent_bg_strong']}; color: {p['title']}; }}

    #sidebarFrame {{ background-color: {p['sunken']}; border-right: 1px solid {p['border_soft']}; }}
    #brandCard, #sidebarInfoCard, #pageHeader, #modeFrame, #inputFrame, #surfaceFrame, #inspectorPanel, #statCard {{
        background-color: {p['panel']}; border: 1px solid {p['border']}; border-radius: 16px;
    }}
    #pageHeader, #inputFrame {{ background-color: {p['surface_soft']}; }}
    #modeFrame {{ background: transparent; border: none; border-bottom: 1px solid {p['border_soft']}; border-radius: 0; }}
    #inputFrame {{ border-radius: 20px; }}
    #transparentFrame, #chatContainer {{ background: transparent; border: none; }}
    #sidebarTitle, #pageTitle, #surfaceTitle {{ color: {p['title']}; }}
    QLabel#pageTitle {{ font-size: 22px; font-weight: 800; }}
    #sidebarSubtitle, #pageSubtitle, #surfaceHint, #headerHint, #headerInfo, #modeStatus, #statHint, #sectionCaption, #sectionCaptionInline, #sidebarInfoValue {{ color: {p['muted']}; }}
    #sectionCaption, #sectionCaptionInline {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }}

    QPushButton {{ border-radius: 12px; padding: 9px 13px; background-color: {p['button']}; color: {p['fg']}; border: 1px solid {p['border']}; }}
    QPushButton:hover {{ background-color: {p['button_hover']}; border-color: {p['accent_border']}; }}
    QPushButton:pressed {{ background-color: {p['button_pressed']}; }}
    QPushButton:disabled {{ color: {p['muted']}; background-color: {p['surface_soft']}; border-color: {p['border_soft']}; }}
    QPushButton#navButton {{ text-align: left; padding: 7px 10px; background: transparent; border: 1px solid transparent; border-radius: 10px; color: {p['muted']}; font-weight: 500; }}
    QPushButton#navButton:hover {{ background-color: {p['button']}; color: {p['title']}; border-color: transparent; }}
    QPushButton#navButton:checked {{ background-color: {p['accent_bg']}; color: {p['title']}; border-color: transparent; }}
    /* 2.0 sidebar: wordmark, new chat, chat list */
    QLabel#sidebarWordmark {{ padding: 2px 4px; font-family: 'Unbounded SemiBold', 'Onest'; font-size: 19px; }}
    QPushButton#newChatButton {{ text-align: left; padding: 9px 12px; border-radius: 12px; background-color: {p['accent_bg']}; border: 1px solid {p['accent_border']}; color: {p['accent_soft']}; font-weight: 700; }}
    QPushButton#newChatButton:hover {{ background-color: {p['accent_bg_strong']}; color: {p['title']}; }}
    QLineEdit#chatSearch {{ padding: 6px 10px; border-radius: 10px; background-color: {p['sunken']}; border: 1px solid {p['border_soft']}; font-size: 12px; }}
    QListWidget#chatList {{ background: transparent; border: none; padding: 0; }}
    QListWidget#chatList::item {{ padding: 0; margin: 1px 0; border-radius: 10px; }}
    QListWidget#chatList::item:hover {{ background: {p['button']}; }}
    QListWidget#chatList::item:selected {{ background: {p['accent_bg']}; }}
    QWidget#chatRow {{ background: transparent; }}
    QLabel#chatRowTitle {{ color: {p['fg']}; font-size: 13px; }}
    QWidget#chatRow[current="true"] QLabel#chatRowTitle {{ color: {p['title']}; font-weight: 700; }}
    QLabel#chatRowTime {{ color: {p['muted_2']}; font-size: 11px; }}
    QLabel#chatRowMarkOk {{ color: {p['accent']}; font-weight: 800; }}
    QLabel#chatRowMarkWarn {{ color: #f59e0b; font-weight: 800; }}
    QFrame#sidebarSections {{ border-top: 1px solid {p['border_soft']}; }}
    QPushButton#resourceToggle {{ text-align: left; padding: 6px 10px; background: transparent; border: 1px solid transparent; border-radius: 10px; color: {p['muted']}; font-size: 12px; }}
    QPushButton#resourceToggle:hover {{ background-color: {p['button']}; color: {p['title']}; }}
    QPushButton#resourceToggle[level="warn"] {{ color: #fbbf24; }}
    QPushButton#resourceToggle[level="hot"] {{ color: #f87171; }}
    /* 2.0 answer: progress trail, file cards, paper sheets, next steps */
    QLabel#progressSummary {{ color: {p['muted']}; font-size: 12px; padding: 1px 0; }}
    QLabel#progressSummary:hover {{ color: {p['accent_soft']}; }}
    QLabel#progressDone {{ font-size: 12px; }}
    QFrame#fileCard {{ background-color: {p['sunken']}; border: 1px solid {p['border_soft']}; border-radius: 12px; }}
    QFrame#fileCard:hover {{ border-color: {p['accent_border']}; }}
    QLabel#fileThumb {{ border-radius: 6px; border: 1px solid {p['border_soft']}; }}
    QLabel#fileCardName {{ color: {p['title']}; font-weight: 700; font-size: 12px; }}
    QLabel#fileCardMeta {{ color: {p['muted']}; font-size: 11px; }}
    QLabel#fileCardStatus {{ color: {p['accent_soft']}; font-size: 11px; }}
    QLabel#fileCardStatus[warn="true"] {{ color: #fbbf24; }}
    QFrame#paperCard {{ background-color: {p['sunken']}; border: 1px solid {p['border_soft']}; border-radius: 14px; }}
    QFrame#paperCard:hover {{ border-color: {p['accent_border']}; }}
    QLabel#paperSheet {{ border: 1px solid #c9c2b2; border-radius: 3px; background: #fdfcf8; }}
    QLabel#paperTitle {{ color: {p['title']}; font-weight: 700; font-size: 13px; }}
    QLabel#paperMeta {{ color: {p['muted']}; font-size: 11px; }}
    QLabel#paperHint {{ color: {p['muted_2']}; font-size: 11px; }}
    QFrame#documentViewer {{ background-color: {p['sunken']}; border: 1px solid {p['border_soft']}; border-radius: 14px; }}
    QScrollArea#pageStrip {{ background: transparent; border: none; }}
    QScrollArea#pageStrip > QWidget > QWidget, QWidget#pageHolder {{ background: transparent; }}
    QTextEdit#paperEditor {{ background-color: #fdfcf8; color: #23221f; border: 1px solid #c9c2b2; border-radius: 6px; padding: 24px 32px; font-family: 'Times New Roman'; font-size: 15px; selection-background-color: #bbf7d0; selection-color: #052e16; }}
    QPushButton#suggestionChip {{ background-color: transparent; color: {p['accent_soft']}; border: 1px solid {p['accent_border']}; border-radius: 13px; padding: 5px 12px; font-size: 12px; font-weight: 500; }}
    QPushButton#suggestionChip:hover {{ background-color: {p['accent_bg']}; color: {p['title']}; }}
    /* 2.0 source beside the chat */
    QFrame#sourcePane {{ background-color: {p['panel']}; border-left: 1px solid {p['border_soft']}; }}
    QLabel#sourcePaneTitle {{ color: {p['title']}; font-size: 14px; font-weight: 700; }}
    QLabel#sourcePaneHint {{ color: {p['muted']}; font-size: 12px; }}
    QPushButton#sourcePaneClose {{ background: transparent; border: none; color: {p['muted']}; font-size: 14px; padding: 2px; }}
    QPushButton#sourcePaneClose:hover {{ color: {p['title']}; }}
    QTextEdit#sourcePaneText {{ background-color: #fdfcf8; color: #23221f; border: 1px solid #c9c2b2; border-radius: 8px; padding: 18px 22px; font-family: 'Times New Roman'; font-size: 15px; selection-background-color: #bbf7d0; selection-color: #052e16; }}
    /* 2.0 Ctrl+K palette */
    QFrame#paletteCard {{ background-color: {p['panel']}; border: 1px solid {p['accent_border']}; border-radius: 18px; }}
    QLineEdit#paletteSearch {{ font-size: 16px; padding: 12px 14px; border-radius: 12px; background-color: {p['sunken']}; border: 1px solid {p['border_soft']}; }}
    QListWidget#paletteResults {{ background: transparent; border: none; padding: 0; }}
    QListWidget#paletteResults::item {{ border-radius: 10px; padding: 0; margin: 0; }}
    QListWidget#paletteResults::item:selected {{ background: {p['accent_bg_strong']}; }}
    QLabel#paletteSection {{ color: {p['muted_2']}; font-size: 10px; font-weight: 700; letter-spacing: 1px; padding: 8px 10px 2px 10px; }}
    QLabel#paletteTitle {{ color: {p['title']}; font-size: 13px; }}
    QLabel#paletteHint {{ color: {p['muted']}; font-size: 11px; background-color: {p['sunken']}; border: 1px solid {p['border_soft']}; border-radius: 6px; padding: 1px 6px; }}
    QLabel#paletteFooter {{ color: {p['muted_2']}; font-size: 11px; padding: 2px 6px; }}
    /* 2.0 chat bar: title, mode switch */
    QLabel#chatTitle {{ color: {p['title']}; font-size: 16px; font-weight: 700; }}
    QFrame#modeSwitch {{ background-color: {p['sunken']}; border: 1px solid {p['border_soft']}; border-radius: 12px; }}
    QFrame#modeSwitch QPushButton#modeButton {{ background: transparent; border: none; border-radius: 9px; padding: 6px 11px; color: {p['muted']}; font-weight: 500; }}
    QFrame#modeSwitch QPushButton#modeButton:hover {{ color: {p['title']}; background-color: {p['button']}; }}
    QFrame#modeSwitch QPushButton#modeButton:checked {{ background-color: {p['accent']}; color: {p['accent_text']}; font-weight: 700; }}
    QPushButton#modeButton {{ background-color: {p['button']}; border-color: {p['border_soft']}; padding: 8px 12px; font-weight: 650; }}
    QPushButton#modeButton:checked, QPushButton#sendButton {{ background-color: {p['accent']}; border-color: {p['accent_hover']}; color: {p['accent_text']}; font-weight: 850; }}
    QPushButton#sendButton:hover {{ background-color: {p['accent_hover']}; }}
    QPushButton#secondaryButton, QPushButton#themeButton, QPushButton#iconButton {{ background-color: {p['button']}; }}
    QPushButton#iconButton {{ min-width: 36px; max-width: 36px; min-height: 36px; max-height: 36px; padding: 0; }}

    /* Три компактных кружка выбора темы в правом верхнем углу. */
    QFrame#themeChipsFrame {{ background: transparent; border: none; }}
    QPushButton#themeChipButton {{ min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px; border-radius: 9px; padding: 0; }}
    QPushButton#themeChipButton[themeKey="classic"] {{ background-color: #22c55e; border: 1px solid rgba(255,255,255,0.55); }}
    QPushButton#themeChipButton[themeKey="dark"] {{ background-color: #07110f; border: 1px solid #2dd46f; }}
    QPushButton#themeChipButton[themeKey="light"] {{ background-color: #f8fafc; border: 1px solid #cbd5e1; }}
    QPushButton#themeChipButton:checked {{ border: 2px solid {p['accent_hover']}; min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px; border-radius: 10px; }}
    QLabel#themeChipsLabel {{ color: {p['muted']}; font-size: 11px; }}

    QPushButton#advancedToggleButton {{ background-color: {p['button']}; color: {p['muted']}; border: 1px solid {p['border_soft']}; border-radius: 12px; padding: 8px 12px; font-weight: 650; }}
    QPushButton#advancedToggleButton:hover {{ color: {p['title']}; background-color: {p['accent_bg']}; border-color: {p['accent_border']}; }}
    QPushButton#advancedToggleButton:checked {{ color: {p['accent_soft']}; background-color: {p['accent_bg_strong']}; border-color: {p['accent_border']}; }}
    QLabel#autoModeBadge {{ background: transparent; color: {p['accent']}; border: none; padding: 7px 4px; font-weight: 700; font-size: 12px; }}

    /* Welcome card with example questions. */
    QFrame#welcomeCard {{ background-color: {p['welcome_bubble']}; border: 1px solid {p['accent_border']}; border-radius: 18px; margin: 6px 10px; }}
    QLabel#welcomeTitle {{ color: {p['title']}; font-size: 20px; font-weight: 800; }}
    QLabel#welcomeText {{ color: {p['fg']}; font-size: 13px; }}
    QLabel#welcomeHint {{ color: {p['muted_2']}; font-size: 11px; }}
    QPushButton#exampleChip {{ text-align: left; background-color: {p['button']}; color: {p['fg']}; border: 1px solid {p['border_soft']}; border-radius: 14px; padding: 10px 14px; font-weight: 600; }}
    QPushButton#exampleChip:hover {{ background-color: {p['accent_bg']}; border-color: {p['accent_border']}; color: {p['title']}; }}
    QPushButton#exampleChip:pressed {{ background-color: {p['accent_bg_strong']}; }}

    /* "Typing…" indicator bubble. */
    QFrame#typingBubble {{ background: {p['ai_bubble']}; border: 1px solid {p['border']}; border-radius: 17px; margin: 6px 10px; }}
    QLabel#typingText {{ color: {p['muted']}; font-size: 12px; }}
    QPushButton#stopButton {{ background-color: {p['button']}; color: {p['fg']}; border: 1px solid {p['border']}; }}
    /* Assistant: attach / voice buttons, attachment chips, file and command rows. */
    QPushButton#attachButton, QPushButton#voiceButton {{ background-color: {p['button']}; color: {p['fg']}; border: 1px solid {p['border']}; font-size: 18px; padding: 4px 0; }}
    QPushButton#attachButton:hover, QPushButton#voiceButton:hover {{ border-color: {p['accent_border']}; background-color: {p['accent_bg']}; }}
    QPushButton#voiceButton:checked {{ background-color: #b91c1c; border-color: #ef4444; color: #ffffff; }}
    QFrame#attachmentChip {{ background-color: {p['accent_bg']}; border: 1px solid {p['accent_border']}; border-radius: 12px; }}
    QPushButton#attachmentRemove {{ background: transparent; border: none; color: {p['muted']}; padding: 0; }}
    QPushButton#attachmentRemove:hover {{ color: #ef4444; }}
    QPushButton#fileActionButton {{ background-color: {p['button']}; border: 1px solid {p['border_soft']}; border-radius: 10px; padding: 4px 10px; font-weight: 600; }}
    QPushButton#fileActionButton:hover {{ border-color: {p['accent_border']}; background-color: {p['accent_bg']}; }}
    QPushButton#commandRunButton {{ background-color: {p['accent']}; color: {p['accent_text']}; border: 1px solid {p['accent_hover']}; border-radius: 10px; padding: 5px 14px; font-weight: 800; }}
    QPushButton#commandRunButton:disabled {{ background-color: {p['button']}; color: {p['muted']}; border-color: {p['border_soft']}; }}
    QPushButton#stopButton:hover {{ border-color: #ef4444; color: #fca5a5; }}
    #advancedSearchPanel {{ background-color: {p['panel_2']}; border: 1px solid {p['border_soft']}; border-radius: 14px; }}
    QLabel#advancedHint {{ color: {p['muted']}; font-size: 12px; }}

    #chatScrollArea, #workspaceStack, #settingsScroll, QWidget#settingsPage {{ background: transparent; border: none; }}
    QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QComboBox {{ background-color: {p['input']}; color: {p['fg']}; border: 1px solid {p['border_soft']}; border-radius: 13px; padding: 10px 12px; selection-background-color: {p['accent_bg_strong']}; selection-color: {p['title']}; }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: {p['accent_border']}; }}
    QComboBox#compactCombo {{ min-width: 132px; padding: 7px 10px; border-color: {p['border']}; background-color: {p['input']}; color: {p['fg']}; }}
    QComboBox#compactCombo:hover {{ border-color: {p['accent_border']}; background-color: {p['button_hover']}; }}
    QComboBox QAbstractItemView {{ background-color: {p['panel']}; color: {p['fg']}; selection-background-color: {p['accent_bg_strong']}; selection-color: {p['title']}; border: 1px solid {p['border']}; outline: none; padding: 6px; }}
    QListWidget::item {{ padding: 10px 8px; border-radius: 10px; margin: 2px 0; }}
    QListWidget::item:selected {{ background: {p['accent_bg_strong']}; color: {p['title']}; }}
    #documentPreview, #diagnosticsView, #inspectorText {{ background-color: {p['input']}; }}

    QFrame[userBubble="true"] {{ background: {p['user_bubble']}; border: 1px solid {p['accent_border']}; border-radius: 17px; margin: 6px 10px 6px 80px; }}
    QFrame[aiBubble="true"] {{ background: {p['ai_bubble']}; border: 1px solid {p['border']}; border-radius: 17px; margin: 6px 80px 6px 10px; }}
    QFrame[welcomeBubble="true"] {{ background: {p['welcome_bubble']}; border-color: {p['accent_border']}; }}
    QLabel[msgAuthor="user"] {{ color: {p['accent_soft']}; font-weight: 750; }}
    QLabel[msgAuthor="ai"] {{ color: {p['title']}; font-weight: 750; }}
    QLabel[msgText="true"] {{ color: {p['title']}; }}
    QLabel[msgMeta="true"] {{ color: {p['muted_2']}; font-size: 11px; }}
    #statTitle {{ color: {p['muted']}; font-size: 11px; font-weight: 700; }}
    #statValue {{ color: {p['title']}; font-size: 18px; font-weight: 850; }}
    #statusBar {{ background-color: {p['sunken']}; color: {p['muted']}; border-top: 1px solid {p['border_soft']}; }}

    #sourceDropdown, #contextAskDropdown, #responseActions {{ background: transparent; border: none; }}
    QPushButton#sourceToggleButton, QPushButton#contextAskToggleButton, QPushButton#explainAnswerButton {{ text-align: center; background-color: {p['button']}; color: {p['accent_soft']}; border: 1px solid {p['accent_border']}; border-radius: 12px; padding: 7px 10px; font-size: 12px; font-weight: 750; }}
    QPushButton#sourceToggleButton:hover, QPushButton#contextAskToggleButton:hover, QPushButton#explainAnswerButton:hover {{ background-color: {p['accent_bg']}; color: {p['title']}; }}
    QPushButton#saveGeneratedDocxButton {{ background-color: {p['accent']}; color: {p['accent_text']}; border: 1px solid {p['accent_hover']}; border-radius: 12px; padding: 7px 12px; font-size: 12px; font-weight: 850; }}
    QPushButton#saveGeneratedDocxButton:hover {{ background-color: {p['accent_hover']}; }}
    #sourcePanel, #contextAskPanel {{ background-color: {p['panel_2']}; border: 1px solid {p['accent_border']}; border-radius: 14px; }}
    #sourceDocRow {{ background-color: {p['panel']}; border: 1px solid {p['border']}; border-radius: 11px; }}
    #sourceDocLabel {{ color: {p['title']}; font-size: 12px; font-weight: 650; }}
    #sourceHintLabel, #contextAskHintLabel {{ color: {p['accent_soft']}; font-size: 11px; }}
    QPushButton#sourceShowButton, QPushButton#contextAskSubmitButton {{ background-color: {p['accent_bg']}; color: {p['accent_soft']}; border: 1px solid {p['accent_border']}; border-radius: 10px; padding: 6px 10px; font-size: 12px; font-weight: 800; }}
    QPushButton#sourceOpenButton {{ background-color: {p['accent']}; color: {p['accent_text']}; border: 1px solid {p['accent_hover']}; border-radius: 10px; padding: 6px 10px; font-size: 12px; font-weight: 850; }}
    QLineEdit#contextAskInput {{ background-color: {p['input']}; border: 1px solid {p['border_soft']}; border-radius: 11px; padding: 7px 10px; }}

    QScrollBar:vertical {{ background: transparent; width: 12px; margin: 4px; }}
    QScrollBar::handle:vertical {{ background: {p['border']}; border-radius: 6px; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {p['accent_border']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 4px; }}
    QScrollBar::handle:horizontal {{ background: {p['border']}; border-radius: 6px; min-width: 24px; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    """
