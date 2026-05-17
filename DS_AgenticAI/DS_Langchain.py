from __future__ import annotations

import os
import sys
import requests
import base64
import json
import time
import aiohttp
import html
import re

from typing import Any, Dict, List, Optional, Iterator, Iterable, Union, Type, Callable, Sequence
from tqdm.auto import tqdm
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.theme import Theme


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompt_values import PromptValue 
from langchain_core.output_parsers import PydanticOutputParser
try:
    from langchain_core.runnables import RunnableWithFallbacks
except Exception:
    RunnableWithFallbacks = None

from pydantic import BaseModel

from dataclasses import dataclass, field
from IPython.display import display, HTML, update_display

from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.front_matter import front_matter_plugin  

from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound





# --------------------------------------------------------------------------------------------------------------

def langsmith(project_name=None, set_enable=True):

    if set_enable:
        langchain_key = os.environ.get("LANGCHAIN_API_KEY", "")
        langsmith_key = os.environ.get("LANGSMITH_API_KEY", "")

        # 더 긴 API 키 선택
        if len(langchain_key.strip()) >= len(langsmith_key.strip()):
            result = langchain_key
        else:
            result = langsmith_key

        if result.strip() == "":
            print(
                "LangChain/LangSmith API Key가 설정되지 않았습니다. 참고: https://wikidocs.net/250954"
            )
            return

        os.environ["LANGSMITH_ENDPOINT"] = (
            "https://api.smith.langchain.com"  # LangSmith API 엔드포인트
        )
        os.environ["LANGSMITH_TRACING"] = "true"  # true: 활성화
        os.environ["LANGSMITH_PROJECT"] = project_name  # 프로젝트명
        print(f"LangSmith 추적을 시작합니다.\n[프로젝트명]\n{project_name}")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"  # false: 비활성화
        print("LangSmith 추적을 하지 않습니다.")


# --------------------------------------------------------------------------------------------------------------

def env_variable(key, value):
    os.environ[key] = value
# --------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------
def read_messages(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return f.read()




# def print_md(texts, inline_code_theme="bold #FF7F50 on #E8E8E8" , code_theme='frendly'):
#     # (For black Themes)
#     # 'monokai'         : 가장 대중적임. 어두운 배경에 형광색 포인트. 기본값, 가장 무난하고 가독성이 좋음.
#     # 'native'          :고전적인 터미널 코드 색상.	심플하고 군더더기 없는 스타일.
#     # 'dracula'         : 보라색/분홍색 계열의 세련된 어두운 테마.	눈이 편안하며 현대적인 느낌.
#     # 'solarized-dark'  : 눈의 피로를 최소화한 부드러운 색감.	장시간 터미널을 보는 개발자에게 최적.
#     # 'github-dark'	    : GitHub의 다크 모드와 유사한 깔끔한 색상.	코드 리뷰나 문서 확인 시 익숙함.
#     #
#     # (For white Themes)
#     # friendly	        : 밝은 배경에 최적화된 가장 표준적인 테마.	가독성이 가장 좋고 깔끔함.
#     # colorful	        : 색상이 뚜렷하고 대비가 강함.	코드의 구조를 한눈에 파악하기 좋음.
#     # bw	            : 흑백(Black & White) 테마.	색상 없이 굵기나 기울임으로만 강조하여 매우 정갈함.
#     # pastie	        : 파스텔 톤의 부드러운 색감.	눈이 편안하며 문서 출력물 같은 느낌.
#     # tango	            : 명확한 색상 대비를 가진 고전적인 테마.	가독성과 심미성을 동시에 잡음.
#     custom_theme = Theme({
#         "markdown.code": inline_code_theme,  # 인라인 코드 스타일 제거
#     })
#     Console(theme=custom_theme).print(Markdown(texts, code_theme=code_theme))

##############################################################################################################

@dataclass
class _Fence:
    lang: str
    code: str


class MarkdownPrint:
    """
    Markdown 출력과 스트리밍 출력(Live rendering)을 동시에 지원하는 유틸리티 클래스.
    + html_output=True 일 때: VSCode Jupyter Interactive에서 HTML로 실시간 렌더링 지원
    """

    _LANG_RE = re.compile(r"^[A-Za-z0-9_+-]+$")

    # 라인 시작 fence: 최대 3칸 들여쓰기 후 ```
    _FENCE_START_RE = re.compile(r"^(?P<indent>[ \t]{0,3})```(?P<info>[^\n]*)[ \t]*$")

    def __init__(
        self,
        text: str = "",
        *,
        inline_code_theme: str = "bold #FF7F50 on #E8E8E8",
        code_theme: str = "friendly",
        stream: bool = False,
        refresh_per_second: int = 20,
        update_every_chars: int = 100,
        update_every_tokens: int = 30,
        update_every_seconds: float = 0.03,
        stabilize_code_block: bool = True,
        end: str = "\n",
        flush: bool = False,

        html_output: bool = True,
        pygments_style: str = "friendly",
    ):
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

        self.html_output = html_output
        self.pygments_style = pygments_style

        self.content = ""
        self._buffer_chars = 0
        self._buffer_tokens = 0
        self._last_update_time = time.time()

        # ---------- 콘솔(rich) 모드 ----------
        custom_theme = Theme({"markdown.code": inline_code_theme})
        self.console = Console(theme=custom_theme)
        self.live: Optional[Live] = None

        if self.stream and (not self.html_output):
            self.live = Live(
                "",
                console=self.console,
                refresh_per_second=self.refresh_per_second,
                transient=False,
            )
            self.live.start()

        # ---------- HTML(Jupyter) 모드 ----------
        self._display_id = None

        # markdown-it 인스턴스
        self._md = MarkdownIt(
            "commonmark",
            {"html": False, "linkify": True, "typographer": True},
        ).use(front_matter_plugin)

        # 핵심: fence rule 커스터마이즈
        self._patch_markdownit_fence_rule()

        if text:
            self.print(text, end=end, flush=flush)

    # ------------------------------
    # markdown-it fence rule 패치(핵심)
    # ------------------------------
    def _parse_info_lang(self, info: str) -> str:
        info = (info or "").strip()
        if not info:
            return ""
        first = info.split()[0].strip()
        return first if self._LANG_RE.match(first) else ""

    def _patch_markdownit_fence_rule(self):
        """
        markdown-it의 block rule 중 fence 규칙을 교체.
        - ``` 다음 첫 토큰이 허용된 언어 토큰이면: 기존 fence 파싱
        - 아니면 fence로 보지 않고 일반 텍스트로 처리(= 코드블록 시작 차단)
        """
        original_fence = self._md.block.ruler.__rules__[self._md.block.ruler.__find__("fence")].fn

        def strict_fence(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
            # 현재 라인 원문 추출
            pos = state.bMarks[startLine] + state.tShift[startLine]
            max_pos = state.eMarks[startLine]
            line = state.src[pos:max_pos]

            m = self._FENCE_START_RE.match(line)
            if not m:
                # fence 후보조차 아니면 기존 로직에 맡김
                return original_fence(state, startLine, endLine, silent)

            info = (m.group("info") or "")
            lang = self._parse_info_lang(info)

            # lang이 비었는데 info가 존재한다면(예: ```가, ```한글, ```!!!)
            # => fence로 파싱하지 않게 강제 (텍스트로 남김)
            if info.strip() and not lang:
                return False

            # lang이 없고 info도 없다(그냥 ``` )는 정상 fence로 허용할지 정책 필요
            # 여기서는 "``` 단독 fence"는 허용(정상 코드블록)
            return original_fence(state, startLine, endLine, silent)

        # markdown-it-py 내부 구조상 ruler에서 rule 교체
        idx = self._md.block.ruler.__find__("fence")
        self._md.block.ruler.__rules__[idx].fn = strict_fence

    # ------------------------------
    # 공통 유틸
    # ------------------------------
    def _should_update(self, force: bool = False) -> bool:
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
        self._buffer_chars = 0
        self._buffer_tokens = 0
        self._last_update_time = time.time()

    # ------------------------------
    # HTML 렌더링
    # ------------------------------
    def _render_markdown_to_html(self, markdown_text: str) -> str:
        html_body = self._md.render(markdown_text)
        html_body = self._postprocess_codeblocks(html_body)
        return self._wrap_with_template(html_body)

    def _postprocess_codeblocks(self, html_body: str) -> str:
        pattern = re.compile(
            r"<pre><code(?: class=\"language-([^\"]+)\")?>\s*([\s\S]*?)\s*</code></pre>",
            re.MULTILINE
        )

        def repl(m):
            lang_raw = (m.group(1) or "").strip()
            lang = lang_raw if self._LANG_RE.match(lang_raw) else ""

            code_html_escaped = m.group(2)
            code_text = html.unescape(code_html_escaped)

            highlighted = self._pygments_highlight(code_text, lang)
            code_for_attr = html.escape(code_text).replace("\n", "&#10;")

            return f"""
<div class="mdp-codeblock">
  <div class="mdp-codeblock-toolbar">
    <div class="mdp-codeblock-lang">{html.escape(lang) if lang else ""}</div>
    <button class="mdp-copy-btn" data-code="{code_for_attr}">copy</button>
  </div>
  <div class="mdp-codeblock-body">{highlighted}</div>
</div>
"""

        return pattern.sub(repl, html_body)

    def _pygments_highlight(self, code: str, lang: str) -> str:
        try:
            if lang:
                lexer = get_lexer_by_name(lang, stripall=False)
            else:
                lexer = guess_lexer(code)
        except ClassNotFound:
            lexer = get_lexer_by_name("text", stripall=False)

        formatter = HtmlFormatter(style=self.pygments_style, noclasses=False)
        return highlight(code, lexer, formatter)

    def _wrap_with_template(self, body_html: str) -> str:
        return f"""
<div class="mdp-root">
  {self._css_bundle()}
  {self._js_bundle()}
  <div class="mdp-markdown">
    {body_html}
  </div>
</div>
"""

    def _css_bundle(self) -> str:
        formatter = HtmlFormatter(style=self.pygments_style)
        pyg_css = formatter.get_style_defs(".highlight")
        return f"""
<style>
.mdp-root {{
  font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji";
  line-height: 1.55;
}}
.mdp-markdown pre {{ margin: 0; }}

.mdp-codeblock {{
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 10px;
  overflow: hidden;
  margin: 12px 0;
}}
.mdp-codeblock-toolbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  background: rgba(0,0,0,0.04);
  border-bottom: 1px solid rgba(0,0,0,0.08);
}}
.mdp-codeblock-lang {{
  font-size: 12px;
  color: rgba(0,0,0,0.55);
}}
.mdp-copy-btn {{
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 8px;
  border: 1px solid rgba(0,0,0,0.18);
  background: white;
  cursor: pointer;
}}
.mdp-copy-btn:active {{ transform: translateY(1px); }}
.mdp-codeblock-body {{
  padding: 10px 12px;
  background: white;
  overflow-x: auto;
}}

{pyg_css}
</style>
"""

    def _js_bundle(self) -> str:
        return """
<script>
(function() {
  document.addEventListener("click", async function(e) {
    const btn = e.target.closest(".mdp-copy-btn");
    if (!btn) return;

    const code = btn.getAttribute("data-code")
      .replaceAll("&#10;", "\\n");

    try {
      await navigator.clipboard.writeText(code);
      const old = btn.textContent;
      btn.textContent = "copied";
      setTimeout(() => btn.textContent = old, 900);
    } catch (err) {
      const old = btn.textContent;
      btn.textContent = "failed";
      setTimeout(() => btn.textContent = old, 900);
    }
  });
})();
</script>
"""

    def _update_html(self, force: bool = False):
        if not self._should_update(force):
            return

        html_out = self._render_markdown_to_html(self.content)

        if self._display_id is None:
            handle = display(HTML(html_out), display_id=True)
            self._display_id = handle.display_id
        else:
            update_display(HTML(html_out), display_id=self._display_id)

        self._reset_update_counter()

    # ------------------------------
    # rich(Live) 렌더링
    # ------------------------------
    def _update_live(self, force: bool = False):
        if not self.live:
            return
        if not self._should_update(force):
            return

        md = Markdown(self.content, code_theme=self.code_theme)
        self.live.update(md, refresh=True)
        self._reset_update_counter()

    # ------------------------------
    # public API
    # ------------------------------
    def print(self, text: str, *, end: Optional[str] = None, flush: Optional[bool] = None):
        if end is None:
            end = self.end
        if flush is None:
            flush = self.flush

        text = str(text)

        if self.stream:
            self.content += text
            self._buffer_chars += len(text)
            self._buffer_tokens += 1

            if self.html_output:
                self._update_html(force=flush)
            else:
                self._update_live(force=flush)
        else:
            if self.html_output:
                self.content = text + end
                self._update_html(force=True)
            else:
                md = Markdown(text + end, code_theme=self.code_theme)
                self.console.print(md, end="")

    def write(self, text: str, **kwargs):
        self.print(text, **kwargs)

    def flush_now(self):
        if self.stream:
            if self.html_output:
                self._update_html(force=True)
            else:
                self._update_live(force=True)

    def close(self):
        if self.stream:
            if self.html_output:
                self._update_html(force=True)
            else:
                if self.live:
                    md = Markdown(self.content, code_theme=self.code_theme)
                    self.live.update(md, refresh=True)
                    self.live.stop()
                    self.live = None

    def __call__(self, text: str, **kwargs):
        self.print(text, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


##############################################################################################################

class StreamResponse:
    """
    content: 스트리밍 중 사용자에게 보여준 문자열 누적
    object : 최종 파싱된 객체(final_parser.parse(content)) 또는 스트림이 직접 내보낸 구조화 객체

    - 텍스트 스트림(AIMessageChunk/str)만 오면: 종료 후 final_parser로 object 생성 (PydanticOutputParser 포함)
    - dict/list 스트림이 오면: object를 즉시 갱신하고 content는 JSON 문자열 스냅샷으로 유지
    """

    def __init__(
        self,
        stream: Iterable,
        custom_metadata: Optional[Dict[str, Any]] = None,
        *,
        final_parser: Optional[Any] = None,  # parse(str) 기대 (PydanticOutputParser 포함)
        structured_render: str = "oneline_json",  # "oneline_json" | "pretty_json" | "str"
        structured_overwrite: bool = True,
        structured_max_width: int = 180,
        structured_flush: bool = True,
        markdown_stream: bool = True,
        markdown_end: str = "",
        markdown_flush: bool = False,
        structured_use_markdownprint: bool = False,
    ):
        self.stream = stream
        self.content: str = ""
        self.object: Any = None

        self.metadata = custom_metadata or {}
        self.chunk_metadata: Dict[str, Any] = {}
        self.token_usage: Dict[str, Any] = {}

        self.final_parser = final_parser

        self.structured_render = structured_render
        self.structured_overwrite = structured_overwrite
        self.structured_max_width = structured_max_width
        self.structured_flush = structured_flush

        self.markdown_stream = markdown_stream
        self.markdown_end = markdown_end
        self.markdown_flush = markdown_flush
        self.structured_use_markdownprint = structured_use_markdownprint

        self._last_line_len = 0
        self._saw_structured_object = False

        self.stream_and_process()

        # 텍스트만 스트리밍된 경우: 종료 후 파서로 최종 객체 생성 (PydanticOutputParser 포함)
        if (not self._saw_structured_object) and self.final_parser is not None:
            self.object = self._final_parse(self.content, self.final_parser)

    # -------- parsing helpers --------
    def _final_parse(self, text: str, parser: Any) -> Any:
        """
        parser.parse(text)를 표준으로 사용.
        (PydanticOutputParser는 parse()가 pydantic model을 반환)
        """
        if parser is None:
            return None
        parse = getattr(parser, "parse", None)
        if not callable(parse):
            raise TypeError("final_parser는 parse(text: str) 메서드를 제공해야 합니다.")
        return parse(text)

    # -------- structured rendering --------
    def _render_structured(self, obj: Union[dict, list]) -> str:
        if self.structured_render == "pretty_json":
            return json.dumps(obj, ensure_ascii=False, indent=2)
        if self.structured_render == "str":
            return str(obj)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    def _trim_one_line(self, s: str) -> str:
        s = s.replace("\n", "\\n")
        if self.structured_max_width and len(s) > self.structured_max_width:
            s = s[: self.structured_max_width - 1] + "…"
        return s

    def _overwrite_line(self, line: str):
        line = self._trim_one_line(line)
        pad = max(0, self._last_line_len - len(line))
        sys.stdout.write("\r" + line + (" " * pad))
        if self.structured_flush:
            sys.stdout.flush()
        self._last_line_len = len(line)

    def _println(self, line: str = ""):
        sys.stdout.write(line + "\n")
        if self.structured_flush:
            sys.stdout.flush()
        self._last_line_len = 0

    # -------- main --------
    def stream_and_process(self, return_output: bool = False) -> Union[None, Dict[str, Any]]:
        start_time = time.time()

        def _separate_overwrite_line_if_needed():
            if self.structured_overwrite and self._last_line_len:
                self._println("")

        with MarkdownPrint(stream=self.markdown_stream, end=self.markdown_end, flush=self.markdown_flush) as print_md:
            for chunk in self.stream:

                # 1) LLM 토큰 스트림
                if isinstance(chunk, AIMessageChunk):
                    _separate_overwrite_line_if_needed()
                    self.content += chunk.content
                    print_md(chunk.content)

                    if getattr(chunk, "response_metadata", None):
                        self.chunk_metadata.update(chunk.response_metadata)

                    if getattr(chunk, "usage_metadata", None):
                        self.token_usage = chunk.usage_metadata
                    elif getattr(chunk, "response_metadata", None) and "token_usage" in chunk.response_metadata:
                        self.token_usage = chunk.response_metadata["token_usage"]

                # 2) 문자열 스트림
                elif isinstance(chunk, str):
                    _separate_overwrite_line_if_needed()
                    self.content += chunk
                    print_md(chunk)

                # 3) 구조화 객체 스트림
                elif isinstance(chunk, (dict, list)):
                    self._saw_structured_object = True
                    self.object = chunk  # 최신 스냅샷 유지

                    s = self._render_structured(chunk)
                    self.content = s  # 스냅샷 문자열로 유지

                    if self.structured_use_markdownprint:
                        _separate_overwrite_line_if_needed()
                        print_md(s if self.structured_render != "pretty_json" else (s + "\n"))
                    else:
                        if self.structured_overwrite:
                            self._overwrite_line(s)
                        else:
                            self._println(s if self.structured_render == "pretty_json" else self._trim_one_line(s))

                # 4) 기타 타입
                else:
                    _separate_overwrite_line_if_needed()
                    text = str(chunk)
                    self.content += text
                    print_md(text)

        if self.structured_overwrite and self._last_line_len:
            self._println("")

        self.metadata["latency_seconds"] = round(time.time() - start_time, 4)
        self.metadata["total_length"] = len(self.content)

        final_metadata = {**self.metadata, **self.chunk_metadata, "token_usage": self.token_usage}

        if return_output:
            return {"content": self.content, "object": self.object, "metadata": final_metadata}

    def __str__(self) -> str:
        return self.content

##############################################################################################################










# --------------------------------------------------------------------------------------------------------------

class PgptLLM(BaseChatModel):
    """ PGPT 전용 LangChain Wrapper Class 
    
    【 LLM MODELS 】
    [{'이름': 'GPT-5.2 Series', '채팅 모델': 'gpt-5.2-chat', '추론 모델': 'gpt-5.2'},
    {'이름': 'GPT-5.1 Series', '채팅 모델': 'gpt-5.1-chat', '추론 모델': 'gpt-5.1-codex'},
    {'이름': 'GPT-5 Series', '채팅 모델': 'gpt-5-chat', '추론 모델': 'gpt-5'},
    {'이름': 'GPT-5 Edge (Mini/Nano)', '채팅 모델': '-', '추론 모델': 'gpt-5-mini\ngpt-5-nano'},
    {'이름': 'GPT-4.1 Series', '채팅 모델': 'gpt-4.1', '추론 모델': '-'},
    {'이름': 'GPT-4o Series', '채팅 모델': 'gpt-4o', '추론 모델': 'gpt-4o-mini'}]

    1. 채팅 모델 (Chat Model)
    우리가 흔히 아는 ChatGPT(GPT-3.5, GPT-4o), Claude 3.5 Sonnet 등이 여기에 속합니다.
        . 작동 방식: 사용자의 질문을 받으면, 직관적인 패턴 매칭을 통해 가장 자연스러운 다음 단어(Next Token)를 즉시 예측하여 출력합니다.
        . 주요 특징:
            . 빠른 속도: 질문하자마자 실시간으로 글자가 타라락 써집니다.
            . 지시사항 준수: "JSON 형태로 출력해", "3줄로 요약해", "친절한 톤으로 말해" 같은 형식(Format) 지시를 매우 잘 따릅니다.
            . 문맥 유지: 대화의 흐름(System, User, Assistant 역할)을 기억하고 티키타카를 하는 데 최적화되어 있습니다.
        . 단점 (한계):
            .   복잡한 수학 문제나 다단계 논리 퍼즐을 주면, 깊게 생각하지 않고 '그럴싸해 보이는 오답'을 즉시 뱉어내는 환각(Hallucination) 현상이 잦습니다.
        . 실무 활용처:
            . RAG (검색 증강 생성): 검색된 사내 문서를 바탕으로 사용자에게 깔끔하게 요약해서 답변할 때.
            . 고객 CS 챗봇: 빠른 응답 속도가 생명인 서비스.
            . 번역 및 텍스트 교정.

    2. 추론 모델 (Reasoning Model)
    최근에 발표된 OpenAI의 o1, o3-mini, 그리고 DeepSeek-R1 등이 대표적인 추론 모델입니다.

    . 작동 방식: 답변을 바로 뱉어내지 않습니다. 내부적으로 **'생각의 사슬(Chain of Thought, CoT)'**이라는 과정을 거칩니다. 문제를 여러 단계로 쪼개고, 스스로 가설을 세우고, 틀리면 다시 돌아가서 수정하는 과정을 거친 후 최종 답변만 출력합니다.
                예를 들어, 복잡한 논리 문제를 풀 때 내부적으로 다음과 같은 검증 과정을 거칩니다.
                    P → Q, -Q ☞ -P
                이러한 대우명제와 같은 논리적 단계를 스스로 수십 번 반복하며 정답을 찾아냅니다.
    . 주요 특징:
        . 압도적인 문제 해결력: 복잡한 코딩, 고난도 수학, 난해한 데이터 분석 로직을 짜는 데 있어 채팅 모델과 비교할 수 없을 정도로 뛰어납니다.
        . 프롬프트 엔지니어링 최소화: "단계별로 생각해(Think step by step)"라고 지시할 필요가 없습니다. 알아서 깊게 생각합니다.
    . 단점 (한계):
        . 느린 속도와 높은 비용: 생각하는 시간(Thinking time)이 짧게는 10초에서 길게는 몇 분까지 걸립니다. 그만큼 API 호출 비용(토큰)도 많이 듭니다.
        . 형식 무시: "반드시 JSON으로만 답해"라고 해도, 자기 생각 과정을 주저리주저리 늘어놓느라 형식을 깨뜨리는 경우가 종종 있습니다.
    . 실무 활용처:
        . 복잡한 데이터 분석 (Pandas Agent): 앞서 질문하신 데이터프레임 분석 시, 복잡한 상관관계나 재무 수식을 코드로 짜야 할 때.
        . 소프트웨어 개발: 수백 줄의 코드를 디버깅하거나 아키텍처를 설계할 때.
    """
    
    api_key: str
    emp_no: str
    comp_no: str = "30"
    model_name: str = "gpt-5.2-chat" # 테스트 코드에 맞춰 변경
    
    # 일반 URL과 스트리밍 URL 명확히 분리
    base_url: str = "http://pgpt.posco.com/s0la01-gpt/gptApi/personalApi"
    stream_url: str = "http://pgpt.posco.com/s0la01-gpt/gptApi/personalApiStream" 
    
    temperature: float = 1.0
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    need_origin: bool = True
    timeout: int = 240

    def _is_reasoning_model(self) -> bool:
        name = self.model_name.lower()
        if name in ["gpt-5.2", "gpt-5.1-codex", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o-mini"]:
            return True
        if "chat" in name:
            return False
        if any(keyword in name for keyword in ["mini", "nano", "o1", "r1", "codex"]):
            return True
        return False
    
    def _prepare_request(self, messages: List[BaseMessage]):
        """API 요청을 위한 헤더와 페이로드를 준비하는 공통 함수"""
        auth_data = {
            "apiKey": self.api_key,
            "empNo": self.emp_no,
            "compNo": self.comp_no
        }
        token = base64.b64encode(json.dumps(auth_data).encode('utf-8')).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        role_mapping = {"human": "user", "ai": "assistant", "system": "system"}
        formatted_messages = [
            {"role": role_mapping.get(m.type, m.type), "content": m.content} 
            for m in messages
        ]
        
        # 🌟 기본 페이로드 (모든 모델 공통)
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "need_origin": self.need_origin
        }

        # 추론 모델이 아닐 때만(일반 채팅 모델일 때만) 파라미터 추가
        if not self._is_reasoning_model():
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p
            payload["frequency_penalty"] = self.frequency_penalty

        return headers, payload

    # ==========================================
    # 1. 일반 출력용 메서드 (llm.invoke)
    # ==========================================
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> ChatResult:
        headers, payload = self._prepare_request(messages)
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")
            
        response_data = response.json()
        
        try:
            answer_text = response_data['choices'][0]['message']['content']
        except KeyError:
            answer_text = json.dumps(response_data, ensure_ascii=False)
        
        message = AIMessage(content=answer_text)
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        headers, payload = self._prepare_request(messages)
        
        response = requests.post(self.stream_url, headers=headers, json=payload, stream=True, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"Stream API Error {response.status_code}: {response.text}")

        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                
                if line_text.startswith('data:'):
                    data_str = line_text.removeprefix('data:').strip()
                    
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data_json = json.loads(data_str)
                        
                        if 'response' in data_json:
                            chunk_content = data_json['response']
                            
                            # 🌟 추가된 로직: 사내 API 응답에서 메타데이터 추출
                            chunk_metadata = {}
                            usage_metadata = {}
                            
                            # 예: 사내 API가 'finish_reason'을 내려준다면
                            if 'finish_reason' in data_json:
                                chunk_metadata['finish_reason'] = data_json['finish_reason']
                                
                            # 예: 사내 API가 'usage' 키로 토큰 정보를 내려준다면
                            if 'usage' in data_json:
                                usage_metadata = data_json['usage']
                            
                            # 🌟 수정된 로직: 청크 생성 시 메타데이터를 함께 주입
                            chunk = AIMessageChunk(
                                content=chunk_content,
                                response_metadata=chunk_metadata,
                                usage_metadata=usage_metadata if usage_metadata else None
                            )
                            yield ChatGenerationChunk(message=chunk)
                            
                    except json.JSONDecodeError:
                        continue

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """
        커스텀 LLM을 위한 with_structured_output 오버라이딩.
        내부적으로 PydanticOutputParser를 강제 적용합니다.
        """
        if not issubclass(schema, BaseModel):
            raise ValueError("schema는 Pydantic BaseModel이어야 합니다.")

        parser = PydanticOutputParser(pydantic_object=schema)
        format_instructions = parser.get_format_instructions()

        # 🌟 수정된 함수: PromptValue를 안전하게 처리하도록 변경
        def inject_instructions(input_val: Union[PromptValue, List[BaseMessage]]) -> List[BaseMessage]:
            # 1. 입력값이 PromptValue 객체인 경우 (프롬프트 템플릿에서 넘어올 때)
            if hasattr(input_val, "to_messages"):
                messages_copy = input_val.to_messages()
            # 2. 이미 리스트 형태인 경우
            elif isinstance(input_val, list):
                messages_copy = list(input_val)
            else:
                raise ValueError(f"지원하지 않는 입력 타입입니다: {type(input_val)}")

            if not messages_copy:
                return messages_copy

            last_message = messages_copy[-1]
            
            # 기존 내용에 JSON 형식 지시문 추가
            new_content = f"{last_message.content}\n\n{format_instructions}"
            
            # 메시지 타입에 따라 새로 생성하여 교체
            if isinstance(last_message, HumanMessage) or last_message.type == "human":
                messages_copy[-1] = HumanMessage(content=new_content)
            elif isinstance(last_message, SystemMessage) or last_message.type == "system":
                messages_copy[-1] = SystemMessage(content=new_content)
            else:
                # 만약 다른 타입이라면 강제로 HumanMessage로 래핑
                messages_copy[-1] = HumanMessage(content=new_content)
                
            return messages_copy

        # [지시문 주입] -> [LLM 실행] -> [Pydantic 파싱] 체인 반환
        chain = inject_instructions | self | parser
        return chain
    
    @property
    def _llm_type(self) -> str:
        return "posco-gpt"

# --------------------------------------------------------------------------------------------------------------





# # --------------------------------------------------------------------------------------------------------------
# class PgptEmbeddings(Embeddings):
#     """ PGPT 전용 Embedding Wrapper Class
#     【 Embedding Model 】
#     1. text-embedding-ada-002 (레거시 표준 모델)
#     2022년 말에 출시되어 전 세계적으로 가장 많이 사용되었던 2세대 표준 임베딩 모델입니다.
#     . 출력 차원(Dimensions): 1536 차원
#     . 비용: 100만 토큰당 약 $0.10
#     . 장점: 오랫동안 표준으로 사용되었기 때문에 인터넷상의 대부분의 튜토리얼과 오픈소스 생태계가 이 모델을 기준으로 작성되어 있어 호환성이 좋습니다.
#     . 단점: 최신 3세대 모델에 비해 다국어(한국어 포함) 이해 능력이 떨어지며, 비용도 3-small에 비해 5배나 비쌉니다.
#     . 실무 가이드: 신규 구축 시에는 사용하지 마세요. 과거에 이미 ada-002로 수십만 건의 문서를 벡터 DB에 구축해 두어서, 모델을 바꾸려면 전체 DB를 다시 임베딩해야 하는(마이그레이션 비용이 큰) 유지보수 상황에서만 사용합니다.

#     2. text-embedding-3-small (현재의 기본/추천 모델)
#     ada-002의 직접적인 후속작으로, 가성비와 성능을 극대화한 3세대 경량화 모델입니다.
#     . 출력 차원(Dimensions): 1536 차원 (기본값) / 차원 축소 가능
#     . 비용: 100만 토큰당 약 $0.02 (ada-002 대비 5배 저렴)
#     . 장점:
#     . 비용이 압도적으로 저렴하면서도 벤치마크(MTEB) 성능은 ada-002보다 뛰어납니다.
#     . 한국어를 포함한 다국어 처리 능력이 크게 향상되었습니다.
#     . 단점: 초고도화된 전문 지식 검색에서는 3-large에 비해 미세한 의미 구분이 아쉬울 수 있습니다.
#     . 실무 가이드: 일반적인 사내 챗봇, RAG 시스템, 고객센터 매뉴얼 검색 등 90% 이상의 실무 프로젝트에 가장 적합합니다. 비용 부담 없이 대규모 데이터를 임베딩할 수 있습니다.

#     3. text-embedding-3-large (고성능/전문가용 모델)
#     OpenAI가 제공하는 임베딩 모델 중 가장 성능이 뛰어나고 똑똑한 최상위 모델입니다.
#     . 출력 차원(Dimensions): 3072 차원 (기본값) / 차원 축소 가능
#     . 비용: 100만 토큰당 약 $0.13
#     . 장점:
#     . 최대 3072 차원의 벡터를 생성하여, 문장의 아주 미세한 뉘앙스나 복잡한 문맥까지 정확하게 수치화합니다.
#     . 어려운 전문 용어가 섞인 문서에서 탁월한 검색 성능을 보여줍니다.
#     . 단점: 3-small에 비해 비용이 약 6.5배 비싸며, 기본 차원 수가 커서 벡터 DB의 저장 공간(메모리/디스크)을 2배 더 차지합니다.
#     . 실무 가이드: 법률 판례 분석, 의료/논문 데이터 검색, 고도의 정확성이 요구되는 금융 분석 등 비용보다 '검색의 정확도'가 최우선인 프로젝트에 도입합니다.
#     """
    
#     # 🌟 해결: __init__ 메서드를 명시적으로 추가하여 변수들을 초기화합니다.
#     def __init__(
#         self, 
#         api_key: str, 
#         emp_no: str, 
#         comp_no: str = "30",
#         # model_name: str = "text-embedding-ada-002",
#         model_name: str = "text-embedding-3-small",
#         # model_name: str = "text-embedding-3-large",
#         embed_url: str = "http://pgpt.posco.com/s0la01-gpt/gptApi/embeddingApi"
#     ):
#         self.api_key = api_key
#         self.emp_no = emp_no
#         self.comp_no = comp_no
#         self.model_name = model_name
#         self.embed_url = embed_url

#     def _get_headers(self):
#         auth_data = {"apiKey": self.api_key, "empNo": self.emp_no, "compNo": self.comp_no}
#         token = base64.b64encode(json.dumps(auth_data).encode('utf-8')).decode('utf-8')
#         return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """여러 문서를 한 번에 벡터로 변환 (DB 저장용)"""
#         payload = {"model": self.model_name, "input": texts}
#         response = requests.post(self.embed_url, headers=self._get_headers(), json=payload)
        
#         if response.status_code != 200:
#             raise Exception(f"Embedding Error {response.status_code}: {response.text}")
            
#         data = response.json()
        
#         # ⚠️ 주의: 사내 API의 응답 JSON 구조가 OpenAI 표준과 다를 수 있습니다.
#         # 만약 여기서 KeyError가 난다면 print(data)를 해서 구조를 확인해야 합니다.
#         return [item["embedding"] for item in data["data"]]

#     def embed_query(self, text: str) -> List[float]:
#         """사용자의 질문 하나를 벡터로 변환 (검색용)"""
#         return self.embed_documents([text])[0]

    



class PgptEmbeddings(Embeddings):
    """ PGPT 전용 Embedding Wrapper Class
    【 Embedding Model 】
    1. text-embedding-ada-002 (레거시 표준 모델)
    2022년 말에 출시되어 전 세계적으로 가장 많이 사용되었던 2세대 표준 임베딩 모델입니다.
    . 출력 차원(Dimensions): 1536 차원
    . 비용: 100만 토큰당 약 $0.10
    . 장점: 오랫동안 표준으로 사용되었기 때문에 인터넷상의 대부분의 튜토리얼과 오픈소스 생태계가 이 모델을 기준으로 작성되어 있어 호환성이 좋습니다.
    . 단점: 최신 3세대 모델에 비해 다국어(한국어 포함) 이해 능력이 떨어지며, 비용도 3-small에 비해 5배나 비쌉니다.
    . 실무 가이드: 신규 구축 시에는 사용하지 마세요. 과거에 이미 ada-002로 수십만 건의 문서를 벡터 DB에 구축해 두어서, 모델을 바꾸려면 전체 DB를 다시 임베딩해야 하는(마이그레이션 비용이 큰) 유지보수 상황에서만 사용합니다.

    2. text-embedding-3-small (현재의 기본/추천 모델)
    ada-002의 직접적인 후속작으로, 가성비와 성능을 극대화한 3세대 경량화 모델입니다.
    . 출력 차원(Dimensions): 1536 차원 (기본값) / 차원 축소 가능
    . 비용: 100만 토큰당 약 $0.02 (ada-002 대비 5배 저렴)
    . 장점:
    . 비용이 압도적으로 저렴하면서도 벤치마크(MTEB) 성능은 ada-002보다 뛰어납니다.
    . 한국어를 포함한 다국어 처리 능력이 크게 향상되었습니다.
    . 단점: 초고도화된 전문 지식 검색에서는 3-large에 비해 미세한 의미 구분이 아쉬울 수 있습니다.
    . 실무 가이드: 일반적인 사내 챗봇, RAG 시스템, 고객센터 매뉴얼 검색 등 90% 이상의 실무 프로젝트에 가장 적합합니다. 비용 부담 없이 대규모 데이터를 임베딩할 수 있습니다.

    3. text-embedding-3-large (고성능/전문가용 모델)
    OpenAI가 제공하는 임베딩 모델 중 가장 성능이 뛰어나고 똑똑한 최상위 모델입니다.
    . 출력 차원(Dimensions): 3072 차원 (기본값) / 차원 축소 가능
    . 비용: 100만 토큰당 약 $0.13
    . 장점:
    . 최대 3072 차원의 벡터를 생성하여, 문장의 아주 미세한 뉘앙스나 복잡한 문맥까지 정확하게 수치화합니다.
    . 어려운 전문 용어가 섞인 문서에서 탁월한 검색 성능을 보여줍니다.
    . 단점: 3-small에 비해 비용이 약 6.5배 비싸며, 기본 차원 수가 커서 벡터 DB의 저장 공간(메모리/디스크)을 2배 더 차지합니다.
    . 실무 가이드: 법률 판례 분석, 의료/논문 데이터 검색, 고도의 정확성이 요구되는 금융 분석 등 비용보다 '검색의 정확도'가 최우선인 프로젝트에 도입합니다.
    """
    
    def __init__(
        self, 
        api_key: str, 
        emp_no: str, 
        comp_no: str = "30",
        model_name: str = "text-embedding-3-small",
        embed_url: str = "http://pgpt.posco.com/s0la01-gpt/gptApi/embeddingApi"
    ):
        self.api_key = api_key
        self.emp_no = emp_no
        self.comp_no = comp_no
        self.model_name = model_name
        self.embed_url = embed_url

    def _get_headers(self) -> dict:
        """API 인증을 위한 헤더 생성"""
        auth_data = {"apiKey": self.api_key, "empNo": self.emp_no, "compNo": self.comp_no}
        token = base64.b64encode(json.dumps(auth_data).encode('utf-8')).decode('utf-8')
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # ==========================================
    # 1. 동기(Synchronous) 메서드
    # ==========================================
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """(내부용) 실제 API를 호출하여 임베딩을 수행하는 동기 메서드"""
        payload = {"model": self.model_name, "input": texts}
        response = requests.post(self.embed_url, headers=self._get_headers(), json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Embedding Error {response.status_code}: {response.text}")
            
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None, latency_time=0.1) -> List[List[float]]:
        """여러 문서를 벡터로 변환 (DB 저장용) - Batch 처리 지원"""
        # batch_size가 None이면 전체를 한 번에 처리
        if batch_size is None:
            return self._embed_batch(texts)
        
        # batch_size가 지정된 경우 청크 단위로 나누어 처리
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(embeddings)
            time.sleep(latency_time)
            
        return all_embeddings

    def embed_query(self, text: str, batch_size: Optional[int] = None) -> List[float]:
        """사용자의 질문 하나를 벡터로 변환 (검색용)"""
        # 단일 텍스트이므로 batch_size는 기능적으로 큰 의미가 없으나, 인터페이스 통일성을 위해 유지
        return self.embed_documents([text], batch_size=batch_size)[0]

    # ==========================================
    # 2. 비동기(Asynchronous) 메서드
    # ==========================================

    async def _aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """(내부용) 실제 API를 호출하여 임베딩을 수행하는 비동기 메서드"""
        payload = {"model": self.model_name, "input": texts}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.embed_url, headers=self._get_headers(), json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Async Embedding Error {response.status}: {error_text}")
                
                data = await response.json()
                return [item["embedding"] for item in data["data"]]

    async def aembed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """여러 문서를 비동기적으로 벡터 변환 - Batch 처리 지원"""
        if batch_size is None:
            return await self._aembed_batch(texts)
        
        all_embeddings = []
        # 비동기 환경에서도 API 서버 과부하를 막기 위해 순차적으로 await 처리 (필요시 asyncio.gather로 동시 실행 가능)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            embeddings = await self._aembed_batch(batch_texts)
            all_embeddings.extend(embeddings)
            
        return all_embeddings

    async def aembed_query(self, text: str, batch_size: Optional[int] = None) -> List[float]:
        """사용자의 질문 하나를 비동기적으로 벡터 변환"""
        embeddings = await self.aembed_documents([text], batch_size=batch_size)
        return embeddings[0]

    def __str__(self):
        return f"<DS_AgenticAI.PgptEmbeddings object model_name='{self.model_name}'>"

















################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import Runnable, RunnableLambda
# from langchain_core.runnables.config import RunnableConfig

# # 선택: 있으면 활용
# try:
#     from langchain_core.runnables import RunnableWithFallbacks
# except Exception:
#     RunnableWithFallbacks = None




################################################################################################################################################


class BasicLLMChain:
    def __init__(self, llm, prompts=None, parser=None, auto_build=True):
        self.llm = llm
        self.prompts = prompts or []
        self.parser = parser

        self.prompts_template = None
        self.chain = None

        self.initialize_prompt_template()
        if auto_build:
            self.build()

    # --- Template ---
    def make_template(self):
        prompt_concat = "\n".join(self.prompts)
        return PromptTemplate.from_template(prompt_concat)

    def initialize_prompt_template(self, partials=None):
        self.prompts_template = self.make_template()
        if partials:
            self.prompts_template = self.prompts_template.partial(**partials)
        self.chain = None  # invalidate

    def partial(self, inputs=None):
        inputs = inputs or {}
        prev = (self.prompts_template.partial_variables or {}).copy()
        prev.update(inputs)
        self.prompts_template = self.prompts_template.partial(**prev)
        self.chain = None  # invalidate
        return self

    def set_prompts(self, prompts):
        self.prompts = prompts or []
        prev_partials = self.prompts_template.partial_variables if self.prompts_template else {}
        self.prompts_template = self.make_template().partial(**prev_partials)
        self.chain = None  # invalidate
        return self

    # --- Chain build ---
    def build(self):
        self.chain = self.prompts_template | self.llm
        if self.parser:
            self.chain = self.chain | self.parser
        return self.chain

    def as_runnable(self):
        return self.chain or self.build()

    # --- Execution ---
    def invoke(self, input, config=None, **kwargs):
        return self.as_runnable().invoke(input, config=config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        return self.as_runnable().stream(input, config=config, **kwargs)

    @property
    def runnable(self):
        return self.as_runnable()

    def __or__(self, other):
        return self.as_runnable().__or__(other)

    def __ror__(self, other):
        return self.as_runnable().__ror__(other)

    def __repr__(self):
        return repr(self.as_runnable())
    
    
################################################################################################################################################
################################################################################################################################################







# ----------------------------
# 정책/스펙 정의(선언적 설정)
# ----------------------------

@dataclass
class RetryPolicy:
    """재시도 정책.

    Attributes:
        enabled: 재시도 활성화 여부.
        max_attempts: 최대 시도 횟수.
        retry_on_exceptions: 재시도할 예외 타입들.
        retry_on_parse_error: 파싱 실패도 재시도 대상으로 볼지 여부(훅 확장용).
        allow_stream_retry: 스트리밍에서 재시도 허용 여부(중복 출력 위험).
    """
    enabled: bool = False
    max_attempts: int = 3
    retry_on_exceptions: tuple = (Exception,)
    retry_on_parse_error: bool = True
    allow_stream_retry: bool = False


@dataclass
class FallbackPolicy:
    """폴백(대체 체인) 정책.

    Attributes:
        enabled: 폴백 활성화 여부.
        fallbacks: 실패 시 대체로 실행할 Runnable 목록.
    """
    enabled: bool = False
    fallbacks: List[Runnable] = field(default_factory=list)


@dataclass
class ToolPolicy:
    """툴 사용 정책.

    Attributes:
        tools: LLM에 바인딩할 tools.
        tool_choice: 모델에 전달할 tool_choice 옵션.
        mode: "call_only"면 tool call만 생성, "auto_execute"면 executor로 실행.
        executor: auto_execute일 때 tool_call을 실행하는 함수.
                  (tool_call_dict) -> tool_result
    """
    tools: Sequence[Any] = field(default_factory=list)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    mode: str = "call_only"  # "call_only" | "auto_execute"
    executor: Optional[Callable[[Any], Any]] = None


@dataclass
class StructuredOutputPolicy:
    """구조화 출력(Structured Output) 정책.

    Attributes:
        schema: Pydantic model 또는 JSON Schema(dict).
        enabled: 구조화 출력 기능 사용 여부.
        prefer_llm_native: 가능하면 LLM의 native structured output 기능 사용.
    """
    schema: Optional[Any] = None
    enabled: bool = False
    prefer_llm_native: bool = True


# ----------------------------
# Step 정의
# ----------------------------

@dataclass
class StepSpec:
    """파이프라인 단계 정의.

    각 단계는 `Runnable`을 하나 제공하며, 최종 체인은
    steps를 왼쪽부터 `|`로 연결하여 구성한다.

    Args:
        name: 단계 식별자(디버깅/교체에 유용).
        runnable_factory: 실제 Runnable을 만드는 함수.
            - self(체인 인스턴스)를 인자로 받아 동적으로 Runnable 생성 가능.
            - 예: prompt template, tool-binding된 llm, parser 등.

    Example:
        StepSpec(
            name="prompt",
            runnable_factory=lambda self: self.prompts_template
        )
    """
    name: str
    runnable_factory: Callable[["BaseLLMChain"], Runnable]


# ----------------------------
# 베이스 체인 (Step 리스트 기반)
# ----------------------------
class BaseLLMChain:
    """Step 리스트 기반의 Base LLM Chain.

    핵심 아이디어:
    - build_* 훅 메서드 여러 개 대신, `steps()`에서 파이프라인 단계를 리스트로 선언한다.
    - 각 단계는 StepSpec(name, runnable_factory)로 표현된다.
    - 정책(tool/structured/retry/fallback)은 조립/실행 시 적용한다.

    Typical pipeline:
        prompt -> llm -> (tool_exec?) -> (postprocess?)

    Subclassing:
        - steps()를 오버라이드해서 단계 추가/교체
        - 또는 add_step/replace_step 유틸 사용

    Notes:
        - retry/fallback은 "전체 체인"에 wrapping으로 적용한다.
        - tool auto_execute는 llm 출력이 tool_call 구조라는 가정이 들어가므로
          프로젝트 표준에 맞게 executor/후속 단계를 커스터마이징 권장.
    """

    def __init__(
        self,
        llm: Runnable,
        prompts: Optional[List[str]] = None,
        *,
        parser: Optional[Runnable] = None,
        tool_policy: Optional[ToolPolicy] = None,
        structured_policy: Optional[StructuredOutputPolicy] = None,
        retry_policy: Optional[RetryPolicy] = None,
        fallback_policy: Optional[FallbackPolicy] = None,
        default_config: Optional[RunnableConfig] = None,
        auto_build: bool = True,
    ):
        """
        Args:
            llm: LangChain Runnable (LLM 또는 LLM chain).
            prompts: PromptTemplate로 합칠 문자열 리스트.
            parser: 최종 후처리 파서(선택).
            tool_policy: Tool 사용 정책.
            structured_policy: Structured output 정책.
            retry_policy: Retry 정책.
            fallback_policy: Fallback 정책.
            default_config: RunnableConfig 기본값(tags/metadata/callbacks 등).
            auto_build: True면 생성 시점에 build() 수행.

        Example:
            chain = BaseLLMChain(
                llm=my_llm,
                prompts=["You are helpful.", "Q: {q}\nA:"],
            )
            chain.invoke({"q": "hi"})
        """
        self.llm = llm
        self.prompts = prompts or []
        self.parser = parser

        self.tool_policy = tool_policy or ToolPolicy()
        self.structured_policy = structured_policy or StructuredOutputPolicy()
        self.retry_policy = retry_policy or RetryPolicy()
        self.fallback_policy = fallback_policy or FallbackPolicy()

        self.default_config: RunnableConfig = default_config or {}

        self.prompts_template: Optional[PromptTemplate] = None
        self._chain: Optional[Runnable] = None

        self.initialize_prompt_template()
        if auto_build:
            self.build()

    # ---------- invalidate ----------
    def invalidate(self) -> None:
        """내부적으로 캐시된 Runnable 체인을 무효화하여 다음 실행 시 재빌드되게 한다."""
        self._chain = None

    # ---------- Template ----------
    def make_template(self) -> PromptTemplate:
        """prompts 리스트를 하나의 PromptTemplate로 만든다."""
        prompt_concat = "\n".join(self.prompts)
        return PromptTemplate.from_template(prompt_concat)

    def initialize_prompt_template(self, partials: Optional[dict] = None) -> None:
        """PromptTemplate 초기화 및 partial 변수 적용.

        Args:
            partials: PromptTemplate.partial에 넣을 고정 변수들.

        Example:
            chain.initialize_prompt_template({"lang": "ko"})
        """
        self.prompts_template = self.make_template()
        if partials:
            self.prompts_template = self.prompts_template.partial(**partials)
        self.invalidate()

    def partial(self, inputs: Optional[dict] = None) -> "BaseLLMChain":
        """기존 partial 변수에 추가로 partial을 적용한다.

        Args:
            inputs: 추가 partial 변수 딕셔너리.

        Returns:
            self

        Example:
            chain.partial({"style": "concise"})
        """
        inputs = inputs or {}
        prev = (self.prompts_template.partial_variables or {}).copy() if self.prompts_template else {}
        prev.update(inputs)
        self.prompts_template = self.make_template().partial(**prev)
        self.invalidate()
        return self

    def set_prompts(self, prompts: List[str]) -> "BaseLLMChain":
        """프롬프트 문자열 리스트를 교체하고 템플릿을 재생성한다."""
        self.prompts = prompts or []
        prev_partials = self.prompts_template.partial_variables if self.prompts_template else {}
        self.prompts_template = self.make_template().partial(**prev_partials)
        self.invalidate()
        return self

    # ---------- 설정 변경용 setter(모두 invalidate) ----------
    def set_tools(
        self,
        tools: Sequence[Any],
        tool_choice: Optional[Union[str, dict]] = None,
        mode: str = "call_only",
        executor: Optional[Callable[[Any], Any]] = None,
    ) -> "BaseLLMChain":
        """툴 정책을 설정한다."""
        self.tool_policy.tools = tools or []
        self.tool_policy.tool_choice = tool_choice
        self.tool_policy.mode = mode
        if executor is not None:
            self.tool_policy.executor = executor
        self.invalidate()
        return self

    def set_structured_output(
        self,
        schema: Any,
        enabled: bool = True,
        prefer_llm_native: bool = True,
    ) -> "BaseLLMChain":
        """구조화 출력 정책을 설정한다."""
        self.structured_policy.schema = schema
        self.structured_policy.enabled = enabled
        self.structured_policy.prefer_llm_native = prefer_llm_native
        self.invalidate()
        return self

    def set_retry(self, enabled: bool = True, **kwargs) -> "BaseLLMChain":
        """재시도 정책을 설정한다."""
        self.retry_policy.enabled = enabled
        for k, v in kwargs.items():
            setattr(self.retry_policy, k, v)
        self.invalidate()
        return self

    def set_fallbacks(self, fallbacks: List[Runnable], enabled: bool = True) -> "BaseLLMChain":
        """폴백 runnable 목록을 설정한다."""
        self.fallback_policy.fallbacks = fallbacks or []
        self.fallback_policy.enabled = enabled
        self.invalidate()
        return self

    # ---------- step factories ----------
    def prompt_runnable(self) -> Runnable:
        """PromptTemplate Runnable을 반환한다."""
        if not self.prompts_template:
            self.initialize_prompt_template()
        return self.prompts_template

    def llm_runnable(self) -> Runnable:
        """정책(tool binding / structured output 등)을 반영한 LLM Runnable을 만든다."""
        llm = self.llm

        # tool binding
        if self.tool_policy.tools and hasattr(llm, "bind_tools"):
            kwargs = {}
            if self.tool_policy.tool_choice is not None:
                kwargs["tool_choice"] = self.tool_policy.tool_choice
            llm = llm.bind_tools(self.tool_policy.tools, **kwargs)

        # structured output (LLM native 우선)
        if self.structured_policy.enabled and self.structured_policy.schema is not None:
            if self.structured_policy.prefer_llm_native and hasattr(llm, "with_structured_output"):
                llm = llm.with_structured_output(self.structured_policy.schema)

        return llm

    def tool_exec_runnable(self) -> Optional[Runnable]:
        """tool auto_execute 단계 Runnable을 만든다.

        Returns:
            mode가 auto_execute인 경우 Runnable, 아니면 None.

        Raises:
            ValueError: auto_execute인데 executor가 없을 때.
        """
        if self.tool_policy.mode != "auto_execute":
            return None
        if not self.tool_policy.executor:
            raise ValueError("tool_policy.mode='auto_execute' 이면 tool_policy.executor가 필요합니다.")
        return RunnableLambda(lambda tool_call: self.tool_policy.executor(tool_call))

    def postprocess_runnable(self) -> Optional[Runnable]:
        """후처리 단계 Runnable.

        기본 구현:
        - 사용자가 parser를 제공했다면 parser를 사용
        - 아니면 None

        확장 포인트:
        - LLM native structured output을 못 쓰는 모델에 대해 JSON/Pydantic 파서를 자동 주입
        """
        return self.parser

    # ---------- steps (핵심) ----------
    def steps(self) -> List[StepSpec]:
        """파이프라인 단계 리스트를 반환한다.

        Subclass가 가장 자주 오버라이드하는 지점.

        Default:
            ["prompt", "llm", ("tool_exec" if enabled), ("postprocess" if exists)]
        """
        s: List[StepSpec] = [
            StepSpec("prompt", lambda self: self.prompt_runnable()),
            StepSpec("llm", lambda self: self.llm_runnable()),
        ]

        # tool auto execution step (optional)
        s.append(StepSpec("tool_exec", lambda self: self.tool_exec_runnable()))  # may be None

        # postprocess step (optional)
        s.append(StepSpec("postprocess", lambda self: self.postprocess_runnable()))  # may be None

        return s

    # ---------- step list utilities ----------
    def add_step(
        self,
        step: StepSpec,
        *,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> "BaseLLMChain":
        """단계를 추가한다.

        Args:
            step: 추가할 StepSpec
            after: 특정 name 뒤에 삽입
            before: 특정 name 앞에 삽입

        Notes:
            after/before 둘 다 없으면 맨 뒤에 추가.
            둘 다 제공되면 before를 우선한다(프로젝트 룰에 맞게 바꿔도 됨).
        """
        steps = self.steps()
        names = [x.name for x in steps]

        if before and before in names:
            idx = names.index(before)
            steps.insert(idx, step)
        elif after and after in names:
            idx = names.index(after)
            steps.insert(idx + 1, step)
        else:
            steps.append(step)

        # steps()가 동적 생성이므로, 인스턴스에 "고정 steps"를 저장하는 방식이 필요.
        # 가장 단순한 방식: _override_steps에 저장하고 steps()는 이를 우선 반환.
        self._override_steps = steps  # type: ignore[attr-defined]
        self.invalidate()
        return self

    def replace_step(self, name: str, new_step: StepSpec) -> "BaseLLMChain":
        """name으로 기존 단계를 교체한다."""
        steps = self.steps()
        for i, s in enumerate(steps):
            if s.name == name:
                steps[i] = new_step
                self._override_steps = steps  # type: ignore[attr-defined]
                self.invalidate()
                return self
        raise KeyError(f"Step '{name}' not found")

    def _effective_steps(self) -> List[StepSpec]:
        """오버라이드된 step 리스트가 있으면 우선 사용한다."""
        override = getattr(self, "_override_steps", None)
        return override if override is not None else self.steps()

    def compose(self) -> Runnable:
        """steps()를 기반으로 Runnable 파이프라인을 합성한다.

        - StepSpec.runnable_factory(self) 결과가 None이면 해당 단계는 skip
        - 반환된 Runnable들을 `|`로 연결한다.

        Raises:
            ValueError: 유효한 단계가 하나도 없을 때.
        """
        runnables: List[Runnable] = []
        for spec in self._effective_steps():
            r = spec.runnable_factory(self)
            if r is None:
                continue
            runnables.append(r)

        if not runnables:
            raise ValueError("No runnable steps to compose.")

        chain = runnables[0]
        for r in runnables[1:]:
            chain = chain | r
        return chain

    # ---------- wrappers ----------
    def wrap_with_fallbacks(self, runnable: Runnable) -> Runnable:
        """fallback 정책을 runnable에 적용한다."""
        if not self.fallback_policy.enabled or not self.fallback_policy.fallbacks:
            return runnable
        if RunnableWithFallbacks is None:
            return runnable
        return RunnableWithFallbacks(runnable=runnable, fallbacks=self.fallback_policy.fallbacks)

    def wrap_with_retry(self, runnable: Runnable, *, streaming: bool) -> Runnable:
        """retry 정책을 runnable에 적용한다.

        Notes:
            스트리밍 retry는 중복 출력 위험이 있어 기본적으로 비활성화.
        """
        if not self.retry_policy.enabled:
            return runnable
        if streaming and not self.retry_policy.allow_stream_retry:
            return runnable

        try:
            from langchain_core.runnables import RunnableRetry
            return RunnableRetry(
                runnable=runnable,
                max_attempt_number=self.retry_policy.max_attempts,
                retry_exception_types=self.retry_policy.retry_on_exceptions,
            )
        except Exception:
            return runnable

    # ---------- build / access ----------
    def build(self) -> Runnable:
        """현재 설정/정책/steps를 반영해 내부 runnable 체인을 빌드한다."""
        core = self.compose()
        core = self.wrap_with_fallbacks(core)
        self._chain = core
        return self._chain

    def as_runnable(self) -> Runnable:
        """내부 runnable 체인을 반환(없으면 build)."""
        return self._chain or self.build()

    @property
    def runnable(self) -> Runnable:
        """as_runnable의 별칭."""
        return self.as_runnable()

    # ---------- config merge ----------
    def _merge_config(self, config: Optional[RunnableConfig]) -> RunnableConfig:
        """default_config와 호출 config를 shallow merge한다."""
        merged = dict(self.default_config)
        if config:
            merged.update(config)
        return merged

    # ---------- execution ----------
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        """체인을 동기 호출한다."""
        r = self.as_runnable()
        r = self.wrap_with_retry(r, streaming=False)
        return r.invoke(input, config=self._merge_config(config), **kwargs)

    def stream(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs):
        """체인을 스트리밍 호출한다."""
        r = self.as_runnable()
        r = self.wrap_with_retry(r, streaming=True)
        return r.stream(input, config=self._merge_config(config), **kwargs)

    # ---------- piping ----------
    def __or__(self, other):
        """`chain | other` 구문 지원."""
        return self.as_runnable().__or__(other)

    def __ror__(self, other):
        """`other | chain` 구문 지원."""
        return self.as_runnable().__ror__(other)

    def __repr__(self):
        return repr(self.as_runnable())
    

# # ------------------------------------------------------------------------------------------------------------------------
# #  1) “전처리 step” 추가 (입력 키 정규화)
# from langchain_core.runnables import RunnableLambda

# class NormalizeInputChain(BaseLLMChain):
#     """입력 dict에서 question 키를 q로 통일하는 전처리 단계를 추가한 체인."""

#     def steps(self):
#         base = super().steps()

#         normalize = StepSpec(
#             "normalize_input",
#             lambda self: RunnableLambda(lambda x: {**x, "q": x.get("q") or x.get("question")})
#         )

#         # prompt 앞에 삽입
#         return [normalize] + base


# # 예시 2) “LLM 후처리 step” 추가 (문자열 트림/후가공)
# from langchain_core.runnables import RunnableLambda

# class TrimOutputChain(BaseLLMChain):
#     """LLM 출력 문자열을 strip하는 후처리 단계를 postprocess 뒤에 추가."""

#     def steps(self):
#         steps = super().steps()

#         trim = StepSpec(
#             "trim",
#             lambda self: RunnableLambda(lambda x: x.strip() if isinstance(x, str) else x)
#         )

#         # 마지막에 추가(또는 postprocess 뒤에 넣고 싶으면 after="postprocess"로 add_step 사용도 가능)
#         return steps + [trim]



# # 예시 3) tool auto_execute + “tool 결과로 최종 답변 생성” 2단 LLM
# class ToolThenAnswerChain(BaseLLMChain):
#     """1) LLM이 tool call 생성 -> 2) tool 실행 -> 3) 결과를 바탕으로 최종 답변 생성."""

#     def __init__(self, llm: Runnable, answer_llm: Runnable, prompts: List[str], answer_prompts: List[str], **kwargs):
#         super().__init__(llm=llm, prompts=prompts, **kwargs)
#         self.answer_llm = answer_llm
#         self.answer_prompts = answer_prompts
        

#     def answer_prompt_runnable(self) -> Runnable:
#         tmpl = PromptTemplate.from_template("\n".join(self.answer_prompts))
#         return tmpl

#     def steps(self):
#         # 1) prompt -> llm(tool call) -> tool_exec
#         # 2) tool_result -> answer_prompt -> answer_llm
#         base = [
#             StepSpec("prompt", lambda self: self.prompt_runnable()),
#             StepSpec("llm", lambda self: self.llm_runnable()),
#             StepSpec("tool_exec", lambda self: self.tool_exec_runnable()),
#             StepSpec("answer_prompt", lambda self: self.answer_prompt_runnable()),
#             StepSpec("answer_llm", lambda self: self.answer_llm),
#             StepSpec("postprocess", lambda self: self.postprocess_runnable()),
#         ]
#         return base



# ################################################################################################################################################


# from __future__ import annotations

# from typing import Any, Dict, List, Optional

# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import Runnable, RunnableLambda
# from langchain_core.output_parsers import SimpleJsonOutputParser


# COD_PROMPT_SYSTEM_ROLE = """
# # Role
# You Are an expert copy-writer.
# """

# COD_PROMPT_SYSTEM_CONTENTS = """
# # Goal
# you will write increasingly concise, entity-dense summaries of the user provided {content_category}. 
# The initial summary should be under {max_words} words and contain {entity_range} informative Descriptive Entities from the {content_category}.

# A Descriptive Entity is:
# - Relevant: to the main story.
# - Specific: descriptive yet concise (5 words or fewer).
# - Faithful: present in the {content_category}.
# - Anywhere: located anywhere in the {content_category}.

# # Your Summarization Process
# - Read through the {content_category} and the all the below sections to get an understanding of the task.
# - Pick {entity_range} informative Descriptive Entities from the {content_category} (";" delimited, do not add spaces).
# - In your output JSON list of dictionaries, write an initial summary of max {max_words} words containing the Entities.
# - You now have `[{{"missing_entities": "...", "denser_summary": "..."}}]`

# Then, repeat the below 2 steps {iterations} times:
# - Step 1. In a new dict in the same list, identify {entity_range} new informative Descriptive Entities from the {content_category} which are missing from the previously generated summary.
# - Step 2. Write a new, denser summary of identical length which covers every Entity and detail from the previous summary plus the new Missing Entities.

# A Missing Entity is:
# - An informative Descriptive Entity from the {content_category} as defined above.
# - Novel: not in the previous summary.

# # Guidelines
# - The first summary should be long (max {max_words} words) yet highly non-specific, containing little information beyond the Entities marked as missing. Use overly verbose language and fillers (e.g., "this {content_category} discusses") to reach ~{max_words} words.
# - Make every word count: re-write the previous summary to improve flow and make space for additional entities.
# - Make space with fusion, compression, and removal of uninformative phrases like "the {content_category} discusses".
# - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the {content_category}.
# - Missing entities can appear anywhere in the new summary.
# - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
# - You're finished when your JSON list has 1+{iterations} dictionaries of increasing density.

# # IMPORTANT
# - Remember, to keep each summary to max {max_words} words.
# - Never remove Entities or details. Only add more from the {content_category}.
# - Do not discuss the {content_category} itself, focus on the content: informative Descriptive Entities, and details.
# - Remember, if you're overusing filler phrases in later summaries, or discussing the {content_category} itself, not its contents, choose more informative Descriptive Entities and include more details from the {content_category}.
# - Answer with a minified JSON list of dictionaries with keys "missing_entities" and "denser_summary".
# - "denser_summary" should be written in the same language as the "content".

# ## Example output
# [{{"missing_entities": "ent1;ent2", "denser_summary": "<vague initial summary with entities 'ent1','ent2'>"}}, {{"missing_entities": "ent3", "denser_summary": "denser summary with 'ent1','ent2','ent3'"}}, ...]
# """

# COD_PROMPT_HUMAN = """
# {content_category}:
# {content}
# """


# class COD_Chain(BaseLLMChain):
#     """Chain-of-Density(COD) 요약 체인.

#     BaseLLMChainV2(step 리스트 기반)를 상속하여,
#     - prompt | llm | parser 로 JSON 결과를 만든 뒤
#     - output_type='final'이면 마지막 아이템의 denser_summary만 추출한다.

#     Args:
#         llm: 사용할 LLM Runnable.
#         output_type: 'json' 또는 'final'
#             - 'json': parser 결과(리스트/딕트 등)를 그대로 반환
#             - 'final': parser 결과에서 x[-1]["denser_summary"]만 반환
#         prompts: COD 프롬프트 문자열 리스트.
#         parser: LLM 출력 -> JSON으로 파싱하는 Runnable parser.
#     """

#     def __init__(
#         self,
#         llm: Runnable,
#         output_type: str = "final",
#         prompts: Optional[List[str]] = None,
#         parser: Optional[Runnable] = None,
#         *,
#         default_partials: Optional[Dict[str, Any]] = None,
#         **kwargs,
#     ):
#         super().__init__(llm=llm, prompts=prompts, parser=parser, auto_build=False, **kwargs)
#         self.output_type = output_type

#         prompts = prompts or [
#             COD_PROMPT_SYSTEM_ROLE,
#             COD_PROMPT_SYSTEM_CONTENTS,
#             COD_PROMPT_HUMAN,
#         ]
#         parser = parser or SimpleJsonOutputParser()

#         # COD_Chain 기존 payload 기본값을 partial로 고정
#         self.initialize_prompt_template(default_partials or {})

#         # steps()에서 output_type을 반영하므로 build
#         self.build()

#     # Base 쪽 initialize_prompt_template는 단순 partial만 지원했었는데,
#     # COD는 기본 payload 규칙이 있으니 여기서 오버라이드.
#     def initialize_prompt_template(self, inputs: Optional[dict] = None) -> None:
#         """COD 기본 partial payload 규칙을 적용해 PromptTemplate을 초기화한다.

#         Args:
#             inputs: content_category/entity_range/iterations/max_words 등을 부분 고정(partial)할 값.
#                 - content_category: 기본 "General Context"
#                 - entity_range: 기본 "1-3"
#                 - iterations: 기본 3
#                 - max_words: 기본 100

#         Example:
#             chain.initialize_prompt_template({
#                 "content_category": "Finance",
#                 "iterations": 5,
#                 "max_words": 120,
#             })
#         """
#         inputs = inputs or {}

#         self.prompts_template = self.make_template()

#         payload = {
#             "content_category": inputs.get("content_category", "General Context"),
#             "entity_range": inputs.get("entity_range", "1-3"),
#             "iterations": int(inputs.get("iterations", 3)),
#             "max_words": int(inputs.get("max_words", 100)),
#         }
#         self.prompts_template = self.prompts_template.partial(**payload)
#         self.invalidate()

#     def steps(self):
#         """output_type에 따라 step 구성을 바꾼다.

#         - json:  prompt -> llm -> parser
#         - final: prompt -> llm -> parser -> select_denser_summary
#         """
#         steps = [
#             StepSpec("prompt", lambda self: self.prompt_runnable()),
#             StepSpec("llm", lambda self: self.llm_runnable()),
#             StepSpec("parser", lambda self: self.parser),
#         ]

#         if self.output_type == "final":
#             steps.append(
#                 StepSpec(
#                     "select_denser_summary",
#                     lambda self: RunnableLambda(lambda x: x[-1]["denser_summary"]),
#                 )
#             )

#         return steps

#     def set_output_type(self, output_type: str) -> "COD_Chain":
#         """출력 타입을 변경하고 체인을 재빌드한다."""
#         if output_type not in ("json", "final"):
#             raise ValueError("output_type must be 'json' or 'final'")
#         self.output_type = output_type
#         self.invalidate()
#         self.build()
#         return self


################################################################################################################################################




