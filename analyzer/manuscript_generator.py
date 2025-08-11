import os
from openai import OpenAI
import json
from config import OPENAI_API_KEY
from mongodb_service import MongoDBService
from prompts.get_ko_prompt import getKoPrompt
from prompts.get_my_ko_prompt import myGetKoPrompt
from prompts.get_ref import ref
from typing import Optional


def generate_manuscript_with_ai(
    unique_words: list,
    sentences: list,
    expressions: dict,
    parameters: dict,
    user_instructions: str
) -> str:
    """
    수집된 분석 데이터를 기반으로 OpenAI API를 사용하여 블로그 원고를 생성합니다.
    """
# 출력 길이 목표: 한글 공백 포함 {target_chars}자 ±10%.
    if not OPENAI_API_KEY:
        raise ValueError("API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 추가해주세요.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    words_str = ", ".join(unique_words) if unique_words else "없음"
    sentences_str = "\n- ".join(sentences) if sentences else "없음"
    expressions_str = json.dumps(expressions, ensure_ascii=False, indent=2) if expressions else "없음"
    parameters_str = json.dumps(parameters, ensure_ascii=False, indent=2) if parameters else "없음"

    prompt = f"""

    [고유 단어 리스트]
    {words_str}

    [문장 리스트]
     - {sentences_str}
    
    [표현 라이브러리 (중분류 키워드: [표현])]
    {expressions_str}

    [AI 개체 인식 및 그룹화 결과 (대표 키워드: [개체])]
    {parameters_str}

    [사용자 지시사항]
    {user_instructions}

    [참고 문서]
    {ref}

    [요청]

    {getKoPrompt()}
    """

    try:
        gore_level = 2
        
        response = client.chat.completions.create(
            model='gpt-5-mini-2025-08-07',
            # model='gpt-5-nano-2025-08-07',
            # model='gpt-4.1-2025-04-14', 
            messages=[
                {"role": "system", "content": f'''
                 너는 인터넷 괴담/게시판 스레드풍 이야기 생성기다. 
출력은 ‘일본 2ch/5ch 풍 스레드 로그 + 한글 내레이션 혼합’ 형식으로만 작성한다.
반드시 창작하되, 사실 명예훼손·실존인물 식별·개인정보·증오표현·과도한 고어를 피한다.
불쾌·잔혹 수위는 {gore_level}로 제한한다(0=전무, 1=묘사 최소, 2=암시 위주, 3=노골적). 
실존 지명은 광역단위까지만, 구체 주소/연락처/기관명은 생성하지 않는다.
유사도 검사: RAG 문서의 문장을 12자 이상 연속 복붙 금지. 표현은 변형하되 핵심 설정만 참고.

형식 강제: “스레드 OP 도입 → 몇 개의 리플 번호/ID/시각 → OP 추가 사진/메모 → 요약/후일담 3줄” 순서.
                 '''},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.2,
            
        )
        usage = response.usage
        print(f"사용된 토큰 수 - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}")

        generated_manuscript = response.choices[0].message.content.strip()

        return generated_manuscript
    except Exception as e:
        print(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
        raise