from openai import OpenAI
from config import OPENAI_API_KEY

def categorize_keyword_with_ai(keyword: str) -> str:
    """
    주어진 키워드를 AI를 사용하여 분석하고, 가장 적합한 카테고리를 반환합니다.
    """
    if not OPENAI_API_KEY:
        raise ValueError("API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 추가해주세요.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 카테고리 목록 예시입니다. 필요에 따라 수정하거나 확장할 수 있습니다.
    categories = [
        "hospital", "legalese", "beauty-treatment", "functional-food", 
        "startup", "home-appliances", "other"
    ]

    prompt = f"""
    다음 키워드가 어떤 카테고리에 가장 적합한지 아래 목록에서 하나만 골라주세요.
    다른 설명 없이 카테고리 이름만 정확하게 반환해야 합니다.

    [키워드]
    {keyword}

    [카테고리 목록]
    {', '.join(categories)}
    """

    try:
        response = client.chat.completions.create(
            model="o3-2025-04-16",
            messages=[
                {"role": "system", "content": "You are a keyword categorization expert."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.0,
        )
        
        category = response.choices[0].message.content.strip()
        
        # AI가 목록에 없는 답변을 할 경우를 대비한 안전장치
        if category not in categories:
            return "기타"
            
        return category

    except Exception as e:
        print(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
        return "기타" # 오류 발생 시 기본값 반환



# model='gpt-5-mini-2025-08-07',        
            # model='gpt-5-2025-08-07',