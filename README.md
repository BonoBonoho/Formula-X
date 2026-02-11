# Formula-X: 바디코디 성과판단 웹 대시보드

요구사항 기반으로 다음 기능을 구현했습니다.

- 회원가입 / 로그인
- `data/raw` 폴더 내 엑셀(`.xlsx`) 자동 로드
- 바디코디 지표(출석률, 재구매율, CSAT) 기반 성과 점수 산정 및 판단
- 대시보드 차트 시각화
- 네이버 플레이스 로그인정보 저장
- 네이버 플레이스 데이터 조회 후 자동입력(데모)

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

- `member_name`
- `branch`
- `attendance_rate`
- `repurchase_rate`
- `csat_score`
- `month`

## 3) 성과 판단 로직

종합 성과 점수 =

- 출석률 평균 × 0.4
- 재구매율 평균 × 0.35
- CSAT 평균(5점 만점)을 100점 환산 후 × 0.25

판단 기준:

- 85 이상: 우수
- 70 이상: 보통
- 그 미만: 개선 필요

## 4) 네이버 플레이스 연동 안내

현재는 **데모 모드**로 구성되어 있습니다.

- 로그인 정보 저장: DB에 저장
- 데이터 조회: 샘플 응답 생성
- 자동입력: 추천 지표를 바디코디 데이터에 즉시 반영

실제 운영 시 네이버 API 또는 공식 제휴 연동 방식으로 `fetch_naver_place_data()`를 교체하면 됩니다.
