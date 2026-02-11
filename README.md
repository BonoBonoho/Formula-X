# Formula-X: 판매 데이터 분석 웹 대시보드

요구사항 기반으로 다음 기능을 구현했습니다.

- 회원가입 / 로그인
- `data/raw` 폴더 내 엑셀(`.xlsx`) 또는 CSV 자동 로드
- 주별/월별 매출 및 객단가 분석
- 대시보드 차트 시각화

## 1) 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

브라우저에서 `http://localhost:5000` 접속.

대시보드에서 파일 업로드 후 즉시 반영할 수도 있습니다.

## 5) Render 배포 (권장)

Render는 기본적으로 **에페메랄 파일시스템**이므로 재배포/재시작 시 로컬 파일이 사라집니다.
지속 저장이 필요하면 **Persistent Disk**를 붙이고, 디스크 마운트 경로를 앱의 데이터 저장 경로로 사용해야 합니다.

### Render 설정 요약

1. Render에서 새 Web Service 생성 (GitHub 연결)
2. Build Command: `pip install -r requirements.txt`  
   Start Command: `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`
3. 환경변수 추가
   - `SECRET_KEY`: 랜덤 문자열
   - `DATA_DIR`: Render 디스크 마운트 경로 (예: `/var/data`)
4. Disks 탭에서 Persistent Disk 추가
   - Mount Path: `/var/data` (또는 위에서 설정한 경로)

> 참고: Persistent Disk는 유료 서비스에만 붙일 수 있으며, 마운트 경로 하위만 영속 저장됩니다.

## 2) 엑셀 파일 포맷

`data/raw/*.xlsx` 또는 `data/raw/*.csv` 파일이 아래 컬럼을 포함하면 자동 적재됩니다.

필수 컬럼 (판매 데이터):

- `결제일시`
- `결제금액`

선택 컬럼 (있으면 자동 반영):

- `지점명`
- `회원명`
- `환불지급액`

## 3) 매출 지표 정의

매출 = 결제금액 - 환불지급액 (순매출)

객단가 = 순매출 ÷ 거래 건수

## 4) 참고

헤더 행이 2~3번째 줄에 있거나 컬럼명이 약간 다르더라도 자동으로 인식하도록 처리했습니다.
