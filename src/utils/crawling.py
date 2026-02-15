import os, re, time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from playwright.sync_api import sync_playwright

BASE = "https://www.data.go.kr"
OUT = "downloads"
os.makedirs(OUT, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0"}

def list_detail_urls(keyword: str, pages: int = 3, per_page: int = 100):
    """
    FILE 데이터 중 'XLSX' 확장자 포함 항목만 목록에서 수집
    """
    urls = []
    for p in range(1, pages + 1):
        list_url = (
            f"{BASE}/tcs/dss/selectDataSetList.do"
            f"?dType=FILE&extsn=XLSX&keyword={requests.utils.quote(keyword)}"
            f"&currentPage={p}&perPage={per_page}"
        )
        html = requests.get(list_url, headers=UA, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.select("a[href*='/data/'][href$='/fileData.do']"):
            href = a.get("href")
            if href:
                urls.append(urljoin(BASE, href))

        time.sleep(0.8)

    return sorted(set(urls))

def download_from_detail(page, detail_url: str):
    page.goto(detail_url, wait_until="networkidle")

    # "기관자체에서 다운로드(제공데이터URL기재)"면 버튼이 '바로가기'로 뜨는 경우가 있음 :contentReference[oaicite:2]{index=2}
    # 그래서 1) 다운로드 시도 → 2) 실패하면 '바로가기' 클릭/URL 추출 로직으로 분기
    candidates = ["다운로드", "바로가기"]

    for label in candidates:
        loc = page.get_by_role("link", name=label)
        if loc.count() == 0:
            continue

        # 같은 텍스트가 여러 개일 수 있어서(예: 컬럼정보 다운로드도 있음) 첫 시도는 앞에서부터
        for i in range(min(loc.count(), 3)):
            target = loc.nth(i)
            try:
                with page.expect_download(timeout=8000) as d:
                    target.click()
                download = d.value
                suggested = download.suggested_filename or "file.xlsx"
                save_path = os.path.join(OUT, suggested)
                download.save_as(save_path)
                return ("downloaded", save_path)
            except Exception:
                # 다운로드가 아니고 새 페이지로 이동하는 형태일 수 있음(바로가기)
                # 클릭 후 URL 변화가 있으면 그 URL을 반환
                before = page.url
                try:
                    target.click()
                    page.wait_for_timeout(1000)
                except Exception:
                    pass
                after = page.url
                if after != before:
                    return ("redirect", after)

    return ("failed", detail_url)

def main(keyword: str, pages: int = 2):
    detail_urls = list_detail_urls(keyword, pages=pages)
    print("detail pages:", len(detail_urls))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for u in tqdm(detail_urls):
            status, info = download_from_detail(page, u)
            print(status, info)
            time.sleep(1.2)

        browser.close()

if __name__ == "__main__":
    main("통계", pages=2)
