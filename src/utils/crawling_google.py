import os
import time
import requests
import serpapi

# 1. 설정
API_KEY = "Key"
SEARCH_QUERY = "고객 filetype:xlsx"
SAVE_DIR = "downloads_google"
TOTAL_PAGES = 10  # 가져올 페이지 수 (페이지당 100개)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

client = serpapi.Client(api_key=API_KEY)

# 2. 페이지 반복 루프
for page in range(TOTAL_PAGES):
    start_index = page * 30
    print(f"\n--- {page + 1}페이지 검색 중 (시작 인덱스: {start_index}) ---")
    
    results = client.search(
        q=SEARCH_QUERY,
        engine="google",
        location="South Korea",
        hl="ko",
        gl="kr",
        num=20,
        start=start_index  # 페이지 시작 위치 지정
    )
    
    organic_results = results.get("organic_results", [])
    
    # 더 이상 검색 결과가 없으면 종료
    if not organic_results:
        print("더 이상의 검색 결과가 없습니다.")
        break

    # 3. 각 페이지의 결과 다운로드
    for idx, result in enumerate(organic_results):
        file_url = result.get("link")
        title = result.get("title", f"p{page}_file_{idx}")
        
        # 파일명 정제 및 저장
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '.', '_')).strip()
        file_name = f"{clean_title}.xlsx"
        file_path = os.path.join(SAVE_DIR, file_name)
        
        try:
            # 서버 부하 방지를 위한 미세한 지연 (선택 사항)
            time.sleep(0.5) 
            
            response = requests.get(file_url, timeout=15)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"   [성공] {file_name}")
        except Exception as e:
            print(f"   [오류] {file_url}: {e}")

print("\n모든 페이지 수집 및 다운로드가 완료되었습니다.")
