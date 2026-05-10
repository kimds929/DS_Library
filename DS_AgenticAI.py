import os
import requests
import base64
import json
import time
import aiohttp

from typing import Any, Dict, List, Optional, Iterator, Iterable, Union, Type, Callable
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
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.prompt_values import PromptValue 
from langchain_core.output_parsers import PydanticOutputParser



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


def env_variable(key, value):
    os.environ[key] = value





# --------------------------------------------------------------------------------------------------------------

class StreamResponse:
    def __init__(self, response: Iterable, custom_metadata: Optional[Dict[str, Any]] = None):
        """
        스트리밍 응답을 처리하고 메타데이터(토큰 사용량 포함)를 관리하는 클래스입니다.
        """
        self.response = response
        self.content = ""  # answer에서 content로 변경됨
        self.metadata = custom_metadata if custom_metadata is not None else {}
        self.chunk_metadata = {}
        self.token_usage = {} # 토큰 사용량을 저장할 딕셔너리 추가
        self.stream_and_process()

    def stream_and_process(self, return_output: bool = False) -> Union[None, Dict[str, Any]]:
        start_time = time.time()
        with MarkdownPrint(stream=True, end="", flush=False) as print_md:
            for token in self.response:
                if isinstance(token, AIMessageChunk):
                    self.content += token.content
                    # print(token.content, end="", flush=True)
                    print_md(token.content)
                    
                    # 1. 일반 메타데이터 추출 (finish_reason, model_name 등)
                    if hasattr(token, 'response_metadata') and token.response_metadata:
                        self.chunk_metadata.update(token.response_metadata)
                    
                    # 2. 토큰 사용량 추출 (LangChain 최신 표준: usage_metadata)
                    if hasattr(token, 'usage_metadata') and token.usage_metadata:
                        self.token_usage = token.usage_metadata
                    # (호환성) 일부 구버전이나 특정 모델은 response_metadata 안에 토큰 정보를 넣기도 함
                    elif hasattr(token, 'response_metadata') and 'token_usage' in token.response_metadata:
                        self.token_usage = token.response_metadata['token_usage']
                        
                elif isinstance(token, str):
                    self.content += token
                    # print(token, end="", flush=True)
                    print_md(token)
                
        # 응답 완료 후 처리 시간 및 길이 계산
        self.metadata['latency_seconds'] = round(time.time() - start_time, 4)
        self.metadata['total_length'] = len(self.content)
        
        # 최종 메타데이터 병합 (토큰 사용량 명시적 추가)
        final_metadata = {
            **self.metadata, 
            **self.chunk_metadata,
            "token_usage": self.token_usage
        }
        
        if return_output:
            return {
                "content": self.content, # 반환 키도 content로 변경
                "metadata": final_metadata
            }

    def __str__(self) -> str:
        return self.content
# --------------------------------------------------------------------------------------------------------------












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






