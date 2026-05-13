from typing import  Optional
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.theme import Theme

import re
from pathlib import Path


################################################################################################

class MarkdownPrint:
    """
    Markdown 출력과 스트리밍 출력(Live rendering)을 동시에 지원하는 유틸리티 클래스.

    주요 기능:
    - 일반 print처럼 Markdown 출력 가능
    - LLM streaming 출력 지원
    - 성능 최적화를 위한 update interval 제어
    - Markdown 코드블록 깨짐 방지 (render 시 임시 보정)
    - with 문을 통한 자동 close 처리

    사용 예시:

    [일반 출력]
        MarkdownPrint("## Hello World")

    [스트리밍 출력]
        with MarkdownPrint(stream=True, end="", flush=False) as md_print:
            for token in response:
                md_print(token.content)
    """

    def __init__(
        self,
        text: str = "",
        *,
        inline_code_theme: str = "bold #FF7F50 on #E8E8E8",
        code_theme: str = "friendly",
        stream: bool = False,
        refresh_per_second: int = 12,

        update_every_chars: int = 30,
        update_every_tokens: int = 1,
        update_every_seconds: float = 0.05,

        stabilize_code_block: bool = True,

        end: str = "\n",
        flush: bool = False,
    ):
        """
        MarkdownPrint 객체를 초기화합니다.

        Args:
            text (str):
                초기 출력할 텍스트. (옵션)

            inline_code_theme (str):
                인라인 코드 스타일 설정 (rich theme 형식)

            code_theme (str):
                Markdown 코드 블록 테마 (friendly, monokai 등)
                    (For dark Themes)
                    . 'monokai'         : 가장 대중적임. 어두운 배경에 형광색 포인트. 기본값, 가장 무난하고 가독성이 좋음.
                    . 'native'          :고전적인 터미널 코드 색상.	심플하고 군더더기 없는 스타일.
                    . 'dracula'         : 보라색/분홍색 계열의 세련된 어두운 테마.	눈이 편안하며 현대적인 느낌.
                    . 'solarized-dark'  : 눈의 피로를 최소화한 부드러운 색감.	장시간 터미널을 보는 개발자에게 최적.
                    . 'github-dark'	    : GitHub의 다크 모드와 유사한 깔끔한 색상.	코드 리뷰나 문서 확인 시 익숙함.
                     
                    (For light Themes)
                    . 'friendly'	        : 밝은 배경에 최적화된 가장 표준적인 테마.	가독성이 가장 좋고 깔끔함.
                    . 'colorful'	        : 색상이 뚜렷하고 대비가 강함.	코드의 구조를 한눈에 파악하기 좋음.
                    . 'bw'	            : 흑백(Black & White) 테마.	색상 없이 굵기나 기울임으로만 강조하여 매우 정갈함.
                    . 'pastie'	        : 파스텔 톤의 부드러운 색감.	눈이 편안하며 문서 출력물 같은 느낌.
                    . 'tango'	            : 명확한 색상 대비를 가진 고전적인 테마.	가독성과 심미성을 동시에 잡음.

            stream (bool):
                True일 경우 Live streaming 모드로 동작
                False일 경우 일반 print처럼 동작

            refresh_per_second (int):
                Live 렌더링 프레임 갱신 속도
                값이 높을수록 부드럽지만 CPU 사용 증가

            update_every_chars (int):
                누적 문자 수 기준으로 렌더링 업데이트 수행

            update_every_tokens (int):
                토큰 개수 기준으로 렌더링 업데이트 수행

            update_every_seconds (float):
                마지막 업데이트 이후 경과 시간 기준 업데이트 수행

            stabilize_code_block (bool):
                코드블록(```)이 닫히지 않은 상태에서
                렌더링 시 임시로 닫아 Markdown 깨짐을 방지

            end (str):
                print의 end와 동일

            flush (bool):
                즉시 렌더링 여부
        """

        self.inline_code_theme = inline_code_theme
        self.code_theme = code_theme
        self.stream = stream
        self.refresh_per_second = refresh_per_second

        self.update_every_chars = update_every_chars
        self.update_every_tokens = update_every_tokens
        self.update_every_seconds = update_every_seconds

        self.stabilize_code_block = stabilize_code_block

        self.end = end
        self.flush = flush

        self.content = ""
        self._buffer_chars = 0
        self._buffer_tokens = 0
        self._last_update_time = time.time()

        custom_theme = Theme({
            "markdown.code": inline_code_theme,
        })

        self.console = Console(theme=custom_theme)
        self.live: Optional[Live] = None

        if self.stream:
            self.live = Live(
                "",
                console=self.console,
                refresh_per_second=self.refresh_per_second,
                transient=False,
            )
            self.live.start()

        if text:
            self.print(text, end=end, flush=flush)

    def _is_code_block_open(self, text: str) -> bool:
        """
        현재 텍스트에서 코드블록이 열려있는지 판단합니다.

        Returns:
            bool:
                True → 코드블록이 아직 닫히지 않은 상태
        """
        return text.count("```") % 2 == 1

    def _renderable_content(self) -> str:
        """
        실제 content를 변경하지 않고,
        렌더링 시 Markdown 안정성을 위한 보정 수행

        Returns:
            str:
                렌더링용 안전한 Markdown 문자열
        """
        if not self.stabilize_code_block:
            return self.content

        if self._is_code_block_open(self.content):
            return self.content + "\n```"

        return self.content

    def _should_update(self, force: bool = False) -> bool:
        """
        렌더링 업데이트를 수행해야 하는지 판단

        조건:
        - 문자 수
        - 토큰 수
        - 시간

        Returns:
            bool:
                업데이트 필요 여부
        """
        if force:
            return True

        now = time.time()

        if self.update_every_chars and self._buffer_chars >= self.update_every_chars:
            return True

        if self.update_every_tokens and self._buffer_tokens >= self.update_every_tokens:
            return True

        if self.update_every_seconds and (now - self._last_update_time) >= self.update_every_seconds:
            return True

        return False

    def _reset_update_counter(self):
        """
        업데이트 후 카운터 초기화
        """
        self._buffer_chars = 0
        self._buffer_tokens = 0
        self._last_update_time = time.time()

    def _update_live(self, force: bool = False):
        """
        Live rendering 수행

        Args:
            force (bool):
                강제 렌더링 여부
        """
        if not self.live:
            return

        if not self._should_update(force):
            return

        md = Markdown(
            self._renderable_content(),
            code_theme=self.code_theme,
        )

        self.live.update(md, refresh=True)
        self._reset_update_counter()

    def print(
        self,
        text: str,
        *,
        end: Optional[str] = None,
        flush: Optional[bool] = None,
    ):
        """
        print()와 동일한 인터페이스로 Markdown 출력 수행

        Args:
            text (str):
                출력할 문자열

            end (str):
                문자열 끝에 추가될 값

            flush (bool):
                즉시 렌더링 여부
        """
        if end is None:
            end = self.end

        if flush is None:
            flush = self.flush

        text = str(text)

        if self.stream:
            self.content += text
            self._buffer_chars += len(text)
            self._buffer_tokens += 1

            self._update_live(force=flush)

        else:
            md = Markdown(text + end, code_theme=self.code_theme)
            self.console.print(md, end="")

    def write(self, text: str, **kwargs):
        """
        print() alias (file-like interface 호환)
        """
        self.print(text, **kwargs)

    def flush_now(self):
        """
        즉시 강제 렌더링 수행
        """
        if self.stream:
            self._update_live(force=True)

    def close(self):
        """
        스트리밍 종료 및 최종 렌더링 수행
        """
        if self.live:
            md = Markdown(self.content, code_theme=self.code_theme)
            self.live.update(md, refresh=True)
            self.live.stop()
            self.live = None

    def __call__(self, text: str, **kwargs):
        """
        객체를 함수처럼 호출 가능하도록 지원
        """
        self.print(text, **kwargs)

    def __enter__(self):
        """
        with 문 진입 시 객체 반환
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        with 문 종료 시 자동 close 수행
        """
        self.close()
################################################################################################




################################################################################################

class MarkdownCodeBlock:
    """
    Markdown 코드블록 하나를 표현하는 객체입니다.

    `MarkdownParser.codeblock()` 또는 `MarkdownParser.codeblocks()`의 반환값으로 사용됩니다.
    코드블록 내부 코드만 보고 싶을 때는 `.content`를 사용하고,
    다시 Markdown 코드블록 형태로 출력하고 싶을 때는 `.code`를 사용합니다.

    Attributes:
        text (str):
            코드블록 내부의 원본 코드 문자열입니다.

        language (str):
            코드블록의 언어 이름입니다.
            예를 들어 ```python 코드블록이면 `"python"`이 저장됩니다.

    Examples:
        >>> block = MarkdownCodeBlock('print("hello")', "python")
        >>> block.content
        'print("hello")'

        >>> print(block.code)
        ```python
        print("hello")
        ```
    """

    def __init__(self, content: str, language: str = ""):
        """
        MarkdownCodeBlock 객체를 생성합니다.

        Args:
            content (str):
                코드블록 내부 코드입니다.
                바깥쪽 ``` fence는 포함하지 않습니다.

            language (str, optional):
                코드블록 언어 이름입니다.
                예: `"python"`, `"javascript"`, `"bash"`.
                기본값은 빈 문자열입니다.
        """
        self.text = content
        self.language = language.strip()

    @property
    def content(self) -> str:
        """
        코드블록 내부 코드만 반환합니다.

        바깥쪽 Markdown fence인 ```는 포함하지 않습니다.

        Returns:
            str:
                앞뒤 공백이 제거된 코드 문자열입니다.

        Examples:
            >>> block = MarkdownCodeBlock('print("hello")', "python")
            >>> block.content
            'print("hello")'
        """
        return self.text.strip()

    @property
    def code(self) -> str:
        """
        Markdown 코드블록 형태의 문자열을 반환합니다.

        `.content`가 코드 내부만 반환한다면,
        `.code`는 Markdown에서 바로 렌더링 가능한 fenced code block 형태로 반환합니다.

        Returns:
            str:
                ```language ... ``` 형태의 Markdown 코드블록 문자열입니다.

        Examples:
            >>> block = MarkdownCodeBlock('print("hello")', "python")
            >>> print(block.code)
            ```python
            print("hello")
            ```

            >>> block = MarkdownCodeBlock("hello")
            >>> print(block.code)
            ```
            hello
            ```
        """

        if self.language:
            return f"```{self.language}\n{self.content}\n```"

        return f"```\n{self.content}\n```"

    def exists(self) -> bool:
        """
        코드블록에 실제 내용이 있는지 확인합니다.

        Returns:
            bool:
                `.content`가 비어 있지 않으면 True,
                비어 있으면 False입니다.

        Examples:
            >>> bool(MarkdownCodeBlock("print(1)", "python"))
            True

            >>> bool(MarkdownCodeBlock(""))
            False
        """
        return bool(self.content)

    def __bool__(self) -> bool:
        """
        객체를 bool 값으로 평가할 때 사용됩니다.

        Returns:
            bool:
                코드블록에 내용이 있으면 True,
                없으면 False입니다.
        """
        return self.exists()

    def __str__(self) -> str:
        """
        객체를 문자열로 변환할 때 코드 내부 내용만 반환합니다.

        Returns:
            str:
                `.content`와 같은 값입니다.
        """
        return self.content

    def __repr__(self) -> str:
        """
        디버깅용 객체 표현 문자열을 반환합니다.

        Returns:
            str:
                언어와 코드 미리보기를 포함한 문자열입니다.
        """
        preview = self.content.replace("\n", "\\n")

        if len(preview) > 80:
            preview = preview[:80] + "..."

        if self.language:
            return f"MarkdownCodeBlock(language={self.language!r}, content={preview!r})"

        return f"MarkdownCodeBlock(content={preview!r})"


class MarkdownParser:
    """
    Markdown 텍스트에서 섹션, 코드블록, 마커 블록을 추출하기 위한 간단한 파서입니다.

    BeautifulSoup의 `find()` / `find_all()` 스타일처럼,
    단수 메서드는 첫 번째 하나를 반환하고,
    복수 메서드는 모든 결과를 리스트로 반환합니다.

    제공하는 주요 메서드는 다음과 같습니다.

    - `section(heading)`:
        첫 번째 Markdown heading 섹션을 반환합니다.

    - `sections(heading)`:
        같은 heading을 가진 모든 섹션을 리스트로 반환합니다.

    - `codeblock(language=None)`:
        첫 번째 코드블록을 반환합니다.

    - `codeblocks(language=None)`:
        모든 코드블록을 리스트로 반환합니다.

    - `marked_block(name)`:
        첫 번째 `<!-- BEGIN:name --> ... <!-- END:name -->` 블록을 반환합니다.

    - `marked_blocks(name)`:
        같은 이름의 모든 마커 블록을 리스트로 반환합니다.

    Attributes:
        text (str):
            파싱 대상 Markdown 원문입니다.

    Examples:
        >>> md = MarkdownParser(Path("B.md").read_text(encoding="utf-8"))

        >>> md.section("# Content").content
        '# Content 아래 내용...'

        >>> md.section("# Content").section("## Detail").content
        'Detail 아래 내용...'

        >>> md.sections("## Detail")
        [MarkdownParser(...), MarkdownParser(...)]

        >>> md.codeblock("python").content
        'print("hello")'

        >>> md.codeblock("python").code
        '```python\\nprint("hello")\\n```'

        >>> md.marked_block("ABC").content
        'ABC 마커 블록 내부 내용...'
    """

    def __init__(self, text: str):
        """
        MarkdownParser 객체를 생성합니다.

        Args:
            text (str):
                파싱할 Markdown 문자열입니다.
        """
        self.text = text

    @property
    def content(self) -> str:
        """
        현재 MarkdownParser가 들고 있는 Markdown 텍스트를 반환합니다.

        섹션이나 마커 블록을 추출한 뒤에는 해당 범위의 내용만 반환합니다.

        Returns:
            str:
                앞뒤 공백이 제거된 Markdown 문자열입니다.

        Examples:
            >>> md.section("# Content").content
            'Content 섹션 내부 내용'
        """
        return self.text.strip()

    def section(self, heading: str) -> "MarkdownParser":
        """
        특정 heading에 해당하는 첫 번째 섹션을 반환합니다.

        BeautifulSoup의 `find()`처럼 동작합니다.
        같은 heading이 여러 개 있어도 첫 번째 결과만 반환합니다.
        결과가 없으면 빈 `MarkdownParser("")`를 반환합니다.

        Args:
            heading (str):
                찾을 Markdown heading입니다.
                반드시 `#`, `##`, `###`처럼 heading marker를 포함해야 합니다.

                예:
                    `"# Content"`
                    `"## Detail"`
                    `"### More Detail"`

        Returns:
            MarkdownParser:
                해당 heading 아래 내용을 담은 `MarkdownParser` 객체입니다.
                찾지 못한 경우 빈 `MarkdownParser("")`를 반환합니다.

        Examples:
            >>> content = md.section("# Content")
            >>> print(content.content)

            >>> detail = md.section("# Content").section("## Detail")
            >>> print(detail.content)

        Notes:
            현재 heading보다 같거나 높은 레벨의 다음 heading이 나오면 섹션 추출을 멈춥니다.

            예를 들어 `# Content`를 찾으면 다음 `# Appendix` 전까지 가져오고,
            그 안의 `## Detail`, `### More Detail` 같은 하위 섹션은 포함합니다.
        """

        found = self.sections(heading)

        if not found:
            return MarkdownParser("")

        return found[0]

    def sections(self, heading: str) -> list["MarkdownParser"]:
        """
        특정 heading에 해당하는 모든 섹션을 리스트로 반환합니다.

        BeautifulSoup의 `find_all()`처럼 동작합니다.
        결과가 없으면 빈 리스트 `[]`를 반환합니다.

        Args:
            heading (str):
                찾을 Markdown heading입니다.
                반드시 `#`, `##`, `###`처럼 heading marker를 포함해야 합니다.

                예:
                    `"# Content"`
                    `"## Detail"`
                    `"### More Detail"`

        Returns:
            list[MarkdownParser]:
                매칭된 각 섹션의 내용을 담은 `MarkdownParser` 객체 리스트입니다.

        Raises:
            ValueError:
                `heading`이 `#`으로 시작하지 않는 경우 발생합니다.

        Examples:
            >>> details = md.sections("## Detail")
            >>> for detail in details:
            ...     print(detail.content)

            >>> details = md.section("# Content").sections("## Detail")
            >>> len(details)
            2

        Notes:
            반환 리스트의 각 요소는 문자열이 아니라 `MarkdownParser`입니다.
            따라서 각 요소에서 다시 `.section()`, `.sections()`, `.codeblock()`,
            `.marked_block()` 등을 체이닝할 수 있습니다.
        """

        heading = heading.strip()
        level = len(heading) - len(heading.lstrip("#"))

        if level == 0:
            raise ValueError("heading은 '# Content' 또는 '## Detail'처럼 #으로 시작해야 합니다.")

        escaped_heading = re.escape(heading)

        pattern = rf"""
        ^[ \t]*{escaped_heading}[ \t]*$
        (?P<body>.*?)
        (?=^[ \t]*\#{{1,{level}}}[ \t]+|\Z)
        """

        matches = re.finditer(
            pattern,
            self.text,
            flags=re.MULTILINE | re.DOTALL | re.VERBOSE
        )

        return [
            MarkdownParser(match.group("body").strip())
            for match in matches
        ]

    def codeblock(self, language: str | None = None) -> MarkdownCodeBlock:
        """
        첫 번째 코드블록을 `MarkdownCodeBlock` 객체로 반환합니다.

        BeautifulSoup의 `find()`처럼 동작합니다.
        언어를 지정하면 해당 언어의 첫 번째 코드블록만 찾습니다.
        언어를 지정하지 않으면 첫 번째 fenced code block을 찾습니다.

        Args:
            language (str | None, optional):
                찾을 코드블록 언어입니다.

                예:
                    `"python"`
                    `"javascript"`
                    `"bash"`

                기본값은 None이며, 이 경우 언어와 관계없이 첫 번째 코드블록을 반환합니다.

        Returns:
            MarkdownCodeBlock:
                첫 번째 코드블록을 나타내는 객체입니다.
                찾지 못한 경우 빈 `MarkdownCodeBlock("", language or "")`를 반환합니다.

        Examples:
            >>> code = md.codeblock("python")
            >>> code.content
            'print("hello")'

            >>> print(code.code)
            ```python
            print("hello")
            ```

            >>> if md.codeblock("python"):
            ...     print("Python 코드블록이 있습니다.")
        """

        found = self.codeblocks(language)

        if not found:
            return MarkdownCodeBlock("", language or "")

        return found[0]

    def codeblocks(self, language: str | None = None) -> list[MarkdownCodeBlock]:
        """
        모든 코드블록을 `MarkdownCodeBlock` 객체 리스트로 반환합니다.

        BeautifulSoup의 `find_all()`처럼 동작합니다.
        언어를 지정하면 해당 언어의 코드블록만 반환합니다.
        언어를 지정하지 않으면 모든 fenced code block을 반환합니다.

        Args:
            language (str | None, optional):
                필터링할 코드블록 언어입니다.

                예:
                    `"python"`이면 ```python 코드블록만 반환합니다.
                    None이면 모든 코드블록을 반환합니다.

        Returns:
            list[MarkdownCodeBlock]:
                매칭된 코드블록 객체 리스트입니다.
                결과가 없으면 빈 리스트 `[]`를 반환합니다.

        Examples:
            >>> python_codes = md.codeblocks("python")
            >>> for code in python_codes:
            ...     print(code.content)

            >>> for code in md.codeblocks("python"):
            ...     print(code.code)

            >>> content_codes = md.section("# Content").codeblocks("python")
        """

        if language:
            pattern = rf"""
            ^```[ \t]*(?P<language>{re.escape(language)})[^\n]*\n
            (?P<body>.*?)
            ^```[ \t]*$
            """
        else:
            pattern = r"""
            ^```[ \t]*(?P<language>[^\s`]*)[^\n]*\n
            (?P<body>.*?)
            ^```[ \t]*$
            """

        matches = re.finditer(
            pattern,
            self.text,
            flags=re.MULTILINE | re.DOTALL | re.VERBOSE
        )

        return [
            MarkdownCodeBlock(
                content=match.group("body").strip(),
                language=match.group("language") or ""
            )
            for match in matches
        ]

    def marked_block(self, name: str) -> "MarkdownParser":
        """
        특정 이름을 가진 첫 번째 마커 블록을 반환합니다.

        BeautifulSoup의 `find()`처럼 동작합니다.
        내부적으로 아래 형태의 마커를 찾습니다.

        `<!-- BEGIN:name -->`
        `<!-- END:name -->`

        Args:
            name (str):
                마커 이름입니다.

                예:
                    `"ABC"`를 넘기면 아래 블록을 찾습니다.

                    `<!-- BEGIN:ABC -->`
                    `...`
                    `<!-- END:ABC -->`

        Returns:
            MarkdownParser:
                첫 번째 마커 블록 내부 내용을 담은 `MarkdownParser` 객체입니다.
                찾지 못한 경우 빈 `MarkdownParser("")`를 반환합니다.

        Examples:
            >>> block = md.marked_block("ABC")
            >>> print(block.content)

            >>> md.marked_block("ABC").section("## Detail").content
        """

        found = self.marked_blocks(name)

        if not found:
            return MarkdownParser("")

        return found[0]

    def marked_blocks(self, name: str) -> list["MarkdownParser"]:
        """
        특정 이름을 가진 모든 마커 블록을 리스트로 반환합니다.

        BeautifulSoup의 `find_all()`처럼 동작합니다.
        마커는 한 줄에 단독으로 있어야 합니다.

        Args:
            name (str):
                마커 이름입니다.

                예:
                    `"ABC"`를 넘기면 아래 형태의 블록을 모두 찾습니다.

                    `<!-- BEGIN:ABC -->`
                    `...`
                    `<!-- END:ABC -->`

        Returns:
            list[MarkdownParser]:
                매칭된 각 마커 블록 내부 내용을 담은 `MarkdownParser` 객체 리스트입니다.
                결과가 없으면 빈 리스트 `[]`를 반환합니다.

        Examples:
            >>> blocks = md.marked_blocks("ABC")
            >>> for block in blocks:
            ...     print(block.content)

            >>> first = md.marked_blocks("ABC")[0]
            >>> print(first.content)

        Notes:
            설명 문장이나 코드 안에 `<!-- END:ABC -->`가 등장하더라도,
            한 줄 전체가 종료 마커인 경우에만 종료 지점으로 인식합니다.
        """

        name = name.strip()

        start_marker = f"<!-- BEGIN:{name} -->"
        end_marker = f"<!-- END:{name} -->"

        pattern = rf"""
        ^[ \t]*{re.escape(start_marker)}[ \t]*$
        (?P<body>.*?)
        ^[ \t]*{re.escape(end_marker)}[ \t]*$
        """

        matches = re.finditer(
            pattern,
            self.text,
            flags=re.MULTILINE | re.DOTALL | re.VERBOSE
        )

        return [
            MarkdownParser(match.group("body").strip())
            for match in matches
        ]

    def exists(self) -> bool:
        """
        현재 MarkdownParser가 실제 내용을 가지고 있는지 확인합니다.

        Returns:
            bool:
                `.content`가 비어 있지 않으면 True,
                비어 있으면 False입니다.

        Examples:
            >>> md.section("# Content").exists()
            True

            >>> md.section("# Missing").exists()
            False
        """
        return bool(self.content)

    def __bool__(self) -> bool:
        """
        객체를 bool 값으로 평가할 때 사용됩니다.

        Returns:
            bool:
                현재 Parser에 내용이 있으면 True,
                비어 있으면 False입니다.

        Examples:
            >>> if md.section("# Content"):
            ...     print("있음")

            >>> if not md.section("# Missing"):
            ...     print("없음")
        """
        return self.exists()

    def __str__(self) -> str:
        """
        객체를 문자열로 변환할 때 현재 Markdown 내용을 반환합니다.

        Returns:
            str:
                `.content`와 같은 값입니다.
        """
        return self.content

    def __repr__(self) -> str:
        """
        디버깅용 객체 표현 문자열을 반환합니다.

        Returns:
            str:
                현재 Markdown 내용의 짧은 미리보기를 포함한 문자열입니다.
        """
        preview = self.content.replace("\n", "\\n")

        if len(preview) > 80:
            preview = preview[:80] + "..."

        return f"MarkdownParser({preview!r})"

################################################################################################



################################################################################################

# # b_text = Path("B.md").read_text(encoding="utf-8")
# b_text = """
# # Content

# Content 본문입니다.

# ## Detail

# 첫 번째 Detail입니다.

# ```python
# print("first")
# ```

# ## Detail

# 두 번째 Detail입니다.

# ```javascript
# console.log("second")
# ```

# # Blocks

# <!-- BEGIN:ABC -->
# 첫 번째 ABC 블록입니다.
# <!-- END:ABC -->

# <!-- BEGIN:ABC -->
# 두 번째 ABC 블록입니다.
# <!-- END:ABC -->

# <!-- BEGIN:XYZ -->
# XYZ 블록입니다.
# <!-- END:XYZ -->
# """

# md = MarkdownParser(b_text)

# # section(), sections() 예제 ----------------------------------------------------------------------------
# detail_one = md.section("## Detail")
# MarkdownPrint(detail_one)

# all_details = md.sections("## Detail")
# all_details
# print("문서 전체의 ## Detail 개수:", len(all_details))

# # 체이닝 예제
# more_detail = (
#     md
#     .section("# Content")
#     .section("## Detail")
#     .section("### More Detail")
# )

# MarkdownPrint(more_detail)


# # codeblock(), codeblocks() 예제 ----------------------------------------------------------------------------
# first_code = md.codeblock()
# MarkdownPrint(first_code.content)
# MarkdownPrint(first_code.code)

# all_codes = md.codeblocks()
# all_codes
# print("전체 코드블록 개수:", len(all_codes))


# # 체이닝 예제
# content_python_codes = md.section("# Content").codeblocks("python")

# # marked_block(), marked_blocks() 예제 ----------------------------------------------------------------------------
# abc = md.marked_block("ABC")
# print(abc.content)

# abc_blocks = md.marked_blocks("ABC")
# print("ABC 블록 개수:", len(abc_blocks))

# # exists() / bool() 예제 ----------------------------------------------------------------------------
# missing = md.section("# Missing")

# print("missing.content:", repr(missing.content))
# print("missing.exists():", missing.exists())
# print("bool(missing):", bool(missing))


# # A.md 템플릿 치환 예제 ----------------------------------------------------------------------------
# # a_text = Path("A.md").read_text(encoding="utf-8")
# a_text = """
# # Result

# ## From Content

# {CONTENT}

# ## From ABC Block

# {ABC}

# ## From Python Code

# {PYTHON_CODE}
# """


# # A.md 읽기
# a_text = Path("A.md").read_text(encoding="utf-8")

# # B.md 읽기
# b_text = Path("B.md").read_text(encoding="utf-8")

# # B.md 파싱
# md = MarkdownParser(b_text)

# # B.md에서 필요한 부분 추출
# content = md.section("# Content").content
# abc = md.marked_block("ABC").content
# python_code = md.codeblock("python").code

# # A.md의 placeholder 치환
# result = (
#     a_text
#     .replace("{CONTENT}", content)
#     .replace("{ABC}", abc)
#     .replace("{PYTHON_CODE}", python_code)
# )

# # 결과 저장
# Path("output.md").write_text(result, encoding="utf-8")

# # 결과 확인
# print(result)

# ################################################################################################
