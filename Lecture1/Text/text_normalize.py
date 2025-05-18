import re
from num2words import num2words

def convert_time(text):
    # 시간 변환 (4시 → 네시)
    time_pattern = r'(\d+)시'
    def time_replacer(match):
        hour = int(match.group(1))
        if hour == 1:
            return "한시"
        elif hour == 2:
            return "두시"
        elif hour == 3:
            return "세시"
        elif hour == 4:
            return "네시"
        elif hour == 5:
            return "다섯시"
        elif hour == 6:
            return "여섯시"
        elif hour == 7:
            return "일곱시"
        elif hour == 8:
            return "여덟시"
        elif hour == 9:
            return "아홉시"
        else:
            return f"{num2words(hour, lang='ko')}시"
    return re.sub(time_pattern, time_replacer, text)

def convert_fraction(text):
    # 분수 변환 (1/3 → 삼분의 일)
    fraction_pattern = r'(\d+)/(\d+)'
    def fraction_replacer(match):
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        return f"{num2words(denominator, lang='ko')}분의 {num2words(numerator, lang='ko')}"
    return re.sub(fraction_pattern, fraction_replacer, text)

def text_normalize(text):
    # 1. 시간 변환
    text = convert_time(text)
    
    # 2. 분수 변환
    text = convert_fraction(text)
    
    # 3. 숫자 → 단어
    text = re.sub(r'\d+(\.\d+)?', lambda x: num2words(x.group(), lang='ko'), text)
    
    # 4. 단위 처리 예시
    text = text.replace("kg", "킬로그램").replace("km", "킬로미터")
    
    # 5. 특수기호 처리
    text = text.replace("%", "퍼센트").replace("$", "달러")
    
    return text

if __name__ == '__main__':
    text = input()
    print(text_normalize(text))