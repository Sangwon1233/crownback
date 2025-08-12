"""
Flask 백엔드 – 얼굴 관상 분석 API (AWS 프리티어 친화 버전)

기능
1. POST /api/v1/metrics   → 지표(JSON)
2. POST /api/v1/annotate  → 지표 + 선 그린 이미지(Base64)
3. POST /api/v1/interpret → 지표 + 선 그린 이미지(Base64) + GPT/Gemini 해석
4. GET  /api/v1/models    → 사용 가능 LLM 목록
5. GET  /health           → 헬스체크

특징
- LLM 호출 비동기(ThreadPoolExecutor) – 서버 블로킹 최소화
- 요청당 이미지 크기 제한(2 MB)
- 작업 후 임시 파일 삭제로 메모리·디스크 절약
- S3 업로드 Stub 포함 (`upload_to_s3`) – 현재는 "" 반환
"""
import os, math, json, base64, tempfile, shutil, concurrent.futures
from statistics import mean

from openai import OpenAI
import google.generativeai as genai
import cv2, numpy as np, face_recognition, requests
from flask import Flask, request, jsonify 
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
# ── 설정 ------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# print("OPENAI_API_KEY : ", OPENAI_API_KEY)
# print("GEMINI_API_KEY : ", GEMINI_API_KEY)

ALLOWED_EXT = {"jpg", "jpeg", "png"}
MAX_IMG_SIZE = 2 * 1024 * 1024  # 2 MB

# ── Flask ----------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMG_SIZE
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── 유틸 -----------------------------------------------------------------

def allowed(fn:str) -> bool:
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXT

def centroid(pts):
    xs, ys = zip(*pts)
    return (mean(xs), mean(ys))

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def poly_len(pts):
    return sum(dist(pts[i], pts[i+1]) for i in range(len(pts)-1))

# ── 얼굴 지표 -------------------------------------------------------------
def extract_metrics(path: str) -> dict:
    img = face_recognition.load_image_file(path)
    locs = face_recognition.face_locations(img)
    lms  = face_recognition.face_landmarks(img)
    if not locs or not lms:
        raise RuntimeError("얼굴 검출 실패")
    (top,right,bottom,left) = locs[0]
    fw, fh = right-left, bottom-top
    lm = lms[0]
    le, re = lm['left_eye'], lm['right_eye']
    leb, reb = lm['left_eyebrow'], lm['right_eyebrow']
    nb, nt = lm['nose_bridge'], lm['nose_tip']
    tl, bl = lm['top_lip'], lm['bottom_lip']
    ch = lm['chin']
    le_c, re_c = centroid(le), centroid(re)
    nt_c = centroid(nt)
    mouth_c = centroid(tl+bl)
    chin_b = ch[len(ch)//2]
    m = {
        "face_width": fw,
        "face_height": fh,
        "eye_distance": dist(le_c, re_c),
        "left_eye_width": dist(le[0], le[-1]),
        "right_eye_width": dist(re[0], re[-1]),
        "left_eye_height": dist(le[1], le[5]),
        "right_eye_height": dist(re[1], re[5]),
        "nose_length": dist(nb[0], nt[-1]),
        "nose_width": dist(nt[0], nt[-1]),
        "mouth_width": dist(tl[0], tl[6]),
        "mouth_height": dist(centroid(tl), centroid(bl)),
        "eye_to_mouth": dist(((le_c[0]+re_c[0])/2,(le_c[1]+re_c[1])/2), mouth_c),
        "eye_left_to_chin": dist(le_c, chin_b),
        "eye_right_to_chin": dist(re_c, chin_b),
        "nose_to_mouth": dist(nt_c, mouth_c),
        "nose_to_chin": dist(nt_c, chin_b),
        "jaw_width": dist(ch[0], ch[-1]),
        "jaw_length": poly_len(ch),
    }
    # 추가 몇 개
    m["mouth_tilt_angle"] = math.degrees(math.atan2(tl[6][1]-tl[0][1], tl[6][0]-tl[0][0]))
    m["left_eb_to_eye_dist"]  = abs(centroid(leb)[1]-le_c[1])
    m["right_eb_to_eye_dist"] = abs(centroid(reb)[1]-re_c[1])
    return m

# ── 전체 시각화 -----------------------------------------------------------

def annotate_image(path: str, metrics: dict) -> str:
    """모든 지표 선/다각형/텍스트를 원본 위에 시각화하고 Base64(JPEG) 문자열을 반환"""
    img = cv2.imread(path)
    lm  = face_recognition.face_landmarks(face_recognition.load_image_file(path))[0]

    # 랜드마크 추출
    le, re  = lm['left_eye'], lm['right_eye']
    leb, reb = lm['left_eyebrow'], lm['right_eyebrow']
    nb, nt   = lm['nose_bridge'], lm['nose_tip']
    tl, bl   = lm['top_lip'], lm['bottom_lip']
    ch       = lm['chin']

    # 핵심 좌표
    le_c, re_c = centroid(le), centroid(re)
    nt_c       = centroid(nt)
    mouth_c    = centroid(tl + bl)
    chin_b     = ch[len(ch) // 2]

    # 볼록껍질 (얼굴 외곽)
    hull = cv2.convexHull(np.array(le + re + leb + reb + nb + nt + tl + bl + ch))

    mapping = {
        # 기본 길이 / 각도
        "eye_distance":            (le_c, re_c, (0, 255, 0)),
        "left_eye_width":          (le[0], le[-1], (0, 200, 0)),
        "right_eye_width":         (re[0], re[-1], (0, 200, 0)),
        "left_eye_height":         (le[1], le[5], (0, 150, 0)),
        "right_eye_height":        (re[1], re[5], (0, 150, 0)),
        "nose_length":             (nb[0], nt[-1], (0, 0, 255)),
        "nose_width":              (nt[0], nt[-1], (0, 0, 200)),
        "mouth_width":             (tl[0], tl[6], (255, 0, 0)),
        "mouth_height":            (centroid(tl), centroid(bl), (200, 0, 0)),
        "eye_to_mouth":            (((le_c[0]+re_c[0])/2, (le_c[1]+re_c[1])/2), mouth_c, (0, 255, 255)),
        "eye_left_to_chin":        (le_c, chin_b, (0, 200, 200)),
        "eye_right_to_chin":       (re_c, chin_b, (0, 200, 200)),
        "nose_to_mouth":           (nt_c, mouth_c, (255, 255, 0)),
        "nose_to_chin":            (nt_c, chin_b, (255, 200, 0)),
        "jaw_width":               (ch[0], ch[-1], (150, 150, 150)),
        "jaw_length":              (ch, None, (100, 100, 100)),
        # 추가 지표
        "mouth_tilt_angle":        (tl[0], tl[6], (0, 255, 255)),
        "left_eb_to_eye_dist":     (centroid(leb), le_c, (255, 0, 255)),
        "right_eb_to_eye_dist":    (centroid(reb), re_c, (255, 0, 255)),
        "eye_mouth_axis_diff":     (le_c, re_c, (0, 200, 255)),
        "mouth_axis":              (tl[0], tl[6], (200, 200, 0)),
        "nose_chin_vector_angle":  (nt_c, chin_b, (0, 0, 200)),
        "cheek_to_jaw_ratio":      (le_c, ch[0], (255, 255, 0)),
        "cheek_asymmetry":         (re_c, ch[-1], (255, 255, 0)),
        "left_eb_cheek_ratio":     (leb, None, (0, 255, 150)),
        "right_eb_cheek_ratio":    (reb, None, (0, 255, 150)),
        "nose_bridge_length":      (nb, None, (255, 0, 255)),
        "philtrum_triangle_area":  ([nt[2], tl[0], tl[6]], None, (0, 255, 0)),
        "hull":                    (hull.squeeze(), None, (0, 0, 0)),
        "left_eb_area":            (leb, None, (255, 150, 0)),
        "right_eb_area":           (reb, None, (255, 150, 0)),
        "left_eb_height_diff":     (leb[0], leb[-1], (0, 150, 255)),
        "right_eb_height_diff":    (reb[0], reb[-1], (0, 150, 255)),
        "left_eye_to_nose_dist":   (le_c, nt_c, (150, 0, 150)),
        "right_eye_to_nose_dist":  (re_c, nt_c, (150, 0, 150)),
    }

    # ─ 그리기 루프 -------------------------------------------------------
    for key, (p1, p2, col) in mapping.items():
        if isinstance(p1, (list, np.ndarray)):
            # 다각형(턱, 눈썹 등)
            pts = np.array([tuple(map(int, pt)) for pt in p1])
            cv2.polylines(img, [pts], True, col, 2)
            mid = tuple(pts[len(pts)//2])
        else:
            pt1 = tuple(map(int, p1))
            pt2 = tuple(map(int, p2)) if p2 is not None else pt1
            cv2.line(img, pt1, pt2, col, 2)
            mid = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)

        val = metrics.get(key)
        if val is not None:
            cv2.putText(img, f"{key}:{val:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')

# ── 비동기 LLM 호출 ─────────────────────────────────────────────────────────
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)



def gpt_call(metrics: dict) -> str:
    """OpenAI GPT-4o-mini 모델로 4,000자 이상 한국식 관상 분석을 반환"""
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt_system = (
        "당신은 15년 경력의 한국식 관상 전문가입니다. 동양 관상 전통(관골, 산근, 미간, 법령, 인중, 구순, 턱선, 이마·광대·코·입·눈썹의 상·중·하정 조화 등)과 현대적 관찰(비율, 좌우대칭, 질감·탄력, 표정 습관)을 함께 참고하여 사람의 기세와 흐름을 풀이합니다. 설명은 “점괘처럼"
        "단정”이 아니라 “~할 가능성이 높다”는 확률적 어투를 쓰되, 한국인이 관상을 볼 때 기대하는 어휘와 문맥(기운, 형상, 복·흉, 귀인, 흐름, 시기감)을 적극 사용하십시오. 수치 지표(JSON)가 주어지면 해당 부위의 형상·비율을 관상 용어로 번역해 풀이하고, 근거를 간단히 덧붙입니다. 긍·>부정 포인트를 균형 있게 제시합니다.\n\n"
        "출력 형식:\n"
        "• 최소 4000자 이상, 한국어, 순수 텍스트(HTML/마크다운 금지)\n"
        "• 각 섹션 헤더는 [섹션명] 형태로 표기\n"
        "• 데이터 기반 수치나 비율(%)는 가능하면 제시\n"
        "• 과도한 단정·예언 금지, 현실적 조언 포함\n\n"
        "섹션 구성(7개):\n"
        "[관상 총운] 얼굴 상·중·하정의 균형, 좌우 비대칭, 피부·근육의 탄력감, 표정 습관을 바탕으로 전반적 기세를 평가. 핵심 강점 3가지와 보완점 3가지를 요약.\n"
        "[금전운] 코(산근·비익·비두), 법령, 턱·하관의 안정감으로 재물의 모임·지킴·새는 구간을 해석. 저축/투자/소비 패턴의 경향과 유의할 시그널을 확률적으로 제시.\n"
        "[연애·결혼운] 눈매·눈꼬리, 입술의 윤곽·광택, 인중·턱선 조합으로 매력 표현 방식, 관계의 지속성, 갈등 시 대처 성향을 분석.\n"
        "[직업·사업운] 이마·미간(계획·결단), 관골·광대(리더십·대외성), 입·턱(실행·마무리)로 커리어 적성, 조직/자영업/프리랜서 적합도와 사업 타이밍을 풀이.\n"
        "[건강운] 안색·다크서클·입술색·피부결·하관의 긴장도로 생활 습관·컨디션·스트레스 항목을 시사. 의료 조언이 아닌 생활 관리 팁으로 한정.\n"
        "[인간관계·귀인운] 눈썹(인연의 폭), 눈빛(신뢰감), 법령·입꼬리(말복·대화운)로 협업·네트워크의 흐름과 도움을 주는 유형을 제시.\n"
        "[삶의 질 행동 가이드] 위의 해석을 바탕으로 즉시 실천 가능한 5가지 이상 행동 가이드(표정·습관·패션/그루밍·커뮤니케이션·업무 루틴 등). 각 항목은 근거 부위와 기대 효과를 1줄로 명시.\n\n"
        "작성 규칙(중요):\n"
        "1) 관상 어휘 예시를 적극 활용하되, 과학 논문식 어투는 피하고 “형상·기운·흐름” 중심으로 설명.\n"
        "   - 예시 어휘: 기세가 오른다/눌린다, 산근이 또렷하다, 미간이 정돈돼 있다, 법령이 탄탄하다, 구순이 단정하다, 하관이 안정적이다 등\n"
        "2) JSON 지표를 관상 용어로 자연스럽게 매핑(예: 코폭 대비 비익 비율↑ → 재물 모임의 그릇이 크다고 해석할 가능성, 좌우대칭 편차↑ → 변동성↑로 풀이할 가능성 등).\n"
        "3) 각 섹션마다 긍정 포인트와 주의 포인트를 각각 최소 2개 이상 제시.\n"
        "4) “언제나 그렇다” 식 표현 금지. “~할 가능성이 높다/완만하다/변동성이 있다/유리하게 작용할 수 있다” 등 확률적 어투 사용.\n"
        "5) 민감한 의료·법률 단정 금지. 건강은 생활 습관 제안 중심.\n"
        "6) 마지막에 전체 요약(5줄 이내)을 덧붙이되, 새 정보 추가 없이 핵심만 재정리.\n\n"
        "입력: 사용자는 얼굴 정량 지표(JSON)를 제공합니다.\n"
        "출력: 위 7개 섹션을 순서대로, 순수 텍스트로만 작성하십시오."
    )

    prompt_user = (
        "아래는 얼굴 정량 지표 JSON입니다. 관상 용어로 자연스럽게 번역해 해석해 주세요. "
        "해석할 때 각 수치가 의미하는 부위를 간단히 덧붙여 주세요.\n" +
        json.dumps(metrics, ensure_ascii=False)
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user",   "content": prompt_user}
        ],
        temperature=0.85,
        max_tokens=5500,  # 4000자 이상 길게 출력되도록 여유 있게 확장
    )
    return resp.choices[0].message.content.strip()




# ── Gemini 호출 (GPT와 동일한 프롬프트 철학 적용) ---------------------------
def gemini_call(metrics: dict) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    # ✅ GPT와 동일한 system 프롬프트 (그대로 복붙)
    prompt_system = (
        "당신은 15년 경력의 한국식 관상 전문가입니다. 동양 관상 전통(관골, 산근, 미간, 법령, 인중, 구순, 턱선, 이마·광대·코·입·눈썹의 상·중·하정 조화 등)과 현대적 관찰(비율, 좌우대칭, 질감·탄력, 표정 습관)을 함께 참고하여 사람의 기세와 흐름을 풀이합니다. 설명은 “점괘처럼\n"
        " 단정”이 아니라 “~할 가능성이 높다”는 확률적 어투를 쓰되, 한국인이 관상을 볼 때 기대하는 어휘와 문맥(기운, 형상, 복·흉, 귀인, 흐름, 시기감)을 적극 사용하십시오. 수치 지표(JSON)가 주어지면 해당 부위의 형상·비율을 관상 용어로 번역해 풀이하고, 근거를 간단히 덧붙입니다. 긍·>부정 포인트를 균형 있게 제시합니다.\n\n"
        "출력 형식:\n"
        "• 최소 4000자 이상, 한국어, 순수 텍스트(HTML/마크다운 금지)\n"
        "• 각 섹션 헤더는 [섹션명] 형태로 표기\n"
        "• 데이터 기반 수치나 비율(%)는 가능하면 제시\n"
        "• 과도한 단정·예언 금지, 현실적 조언 포함\n\n"
        "섹션 구성(7개):\n"
        "[관상 총운] 얼굴 상·중·하정의 균형, 좌우 비대칭, 피부·근육의 탄력감, 표정 습관을 바탕으로 전반적 기세를 평가. 핵심 강점 3가지와 보완점 3가지를 요약.\n"
        "[금전운] 코(산근·비익·비두), 법령, 턱·하관의 안정감으로 재물의 모임·지킴·새는 구간을 해석. 저축/투자/소비 패턴의 경향과 유의할 시그널을 확률적으로 제시.\n"
        "[연애·결혼운] 눈매·눈꼬리, 입술의 윤곽·광택, 인중·턱선 조합으로 매력 표현 방식, 관계의 지속성, 갈등 시 대처 성향을 분석.\n"
        "[직업·사업운] 이마·미간(계획·결단), 관골·광대(리더십·대외성), 입·턱(실행·마무리)로 커리어 적성, 조직/자영업/프리랜서 적합도와 사업 타이밍을 풀이.\n"
        "[건강운] 안색·다크서클·입술색·피부결·하관의 긴장도로 생활 습관·컨디션·스트레스 항목을 시사. 의료 조언이 아닌 생활 관리 팁으로 한정.\n"
        "[인간관계·귀인운] 눈썹(인연의 폭), 눈빛(신뢰감), 법령·입꼬리(말복·대화운)로 협업·네트워크의 흐름과 도움을 주는 유형을 제시.\n"
        "[삶의 질 행동 가이드] 위의 해석을 바탕으로 즉시 실천 가능한 5가지 이상 행동 가이드(표정·습관·패션/그루밍·커뮤니케이션·업무 루틴 등). 각 항목은 근거 부위와 기대 효과를 1줄로 명시.\n\n"
        "작성 규칙(중요):\n"
        "1) 관상 어휘 예시를 적극 활용하되, 과학 논문식 어투는 피하고 “형상·기운·흐름” 중심으로 설명.\n"
        "   - 예시 어휘: 기세가 오른다/눌린다, 산근이 또렷하다, 미간이 정돈돼 있다, 법령이 탄탄하다, 구순이 단정하다, 하관이 안정적이다 등\n"
        "2) JSON 지표를 관상 용어로 자연스럽게 매핑(예: 코폭 대비 비익 비율↑ → 재물 모임의 그릇이 크다고 해석할 가능성, 좌우대칭 편차↑ → 변동성↑로 풀이할 가능성 등).\n"
        "3) 각 섹션마다 긍정 포인트와 주의 포인트를 각각 최소 2개 이상 제시.\n"
        "4) “언제나 그렇다” 식 표현 금지. “~할 가능성이 높다/완만하다/변동성이 있다/유리하게 작용할 수 있다” 등 확률적 어투 사용.\n"
        "5) 민감한 의료·법률 단정 금지. 건강은 생활 습관 제안 중심.\n"
        "6) 마지막에 전체 요약(5줄 이내)을 덧붙이되, 새 정보 추가 없이 핵심만 재정리.\n\n"
        "입력: 사용자는 얼굴 정량 지표(JSON)를 제공합니다.\n"
        "출력: 위 7개 섹션을 순서대로, 순수 텍스트로만 작성하십시오."
    )

    # ✅ GPT의 user 프롬프트와 동일
    prompt_user = (
        "아래는 얼굴 정량 지표 JSON입니다. 관상 용어로 자연스럽게 번역해 해석해 주세요. "
        "해석할 때 각 수치가 의미하는 부위를 간단히 덧붙여 주세요.\n" +
        json.dumps(metrics, ensure_ascii=False)
    )

    # Gemini는 system → system_instruction, user → 텍스트로 전달
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=prompt_system
    )

    try:
        response = model.generate_content(
            [prompt_user],
            generation_config=genai.types.GenerationConfig(
                temperature=0.85,
                top_p=0.9,
                top_k=40,
                max_output_tokens=5500,
            )
        )

        text = getattr(response, "text", None)
        if not text:
            # 안전망: 후보에서 텍스트 수집
            if hasattr(response, "candidates") and response.candidates:
                parts = []
                for c in response.candidates:
                    if hasattr(c, "content") and getattr(c.content, "parts", None):
                        parts.extend([p.text for p in c.content.parts if hasattr(p, "text")])
                text = "\n".join([t for t in parts if t]) or ""
        if not text:
            raise RuntimeError("Gemini 응답에 텍스트가 비어 있습니다.")
        return text.strip()

    except Exception as e:
        print(f"[Gemini 오류] {e}")
        raise RuntimeError("Gemini 호출 실패")




LLM_FUNCS = {"gpt": gpt_call, "gemini": gemini_call}

# S3 Stub (미구현) ------------------------------------------------------------

def upload_to_s3(_path: str) -> str:
    """향후 S3 업로드 예정. 현재는 빈 문자열 반환"""
    return ""

# ── API 엔드포인트 ──────────────────────────────────────────────────────────

def _save_temp(file) -> str:
    fname = secure_filename(file.filename)
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, fname)
    file.save(path)
    return path, tmp_dir

@app.route('/api/v1/metrics', methods=['POST'])
def api_metrics():
    if 'image' not in request.files:
        return jsonify({'error':'image 필요'}), 400
    f = request.files['image']
    if not allowed(f.filename):
        return jsonify({'error':'jpg/png 만 허용'}), 400
    path, tmp = _save_temp(f)
    try:
        metrics = extract_metrics(path)
        return jsonify(metrics)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.route('/api/v1/annotate', methods=['POST'])
def api_annotate():
    if 'image' not in request.files:
        return jsonify({'error':'image 필요'}), 400
    f = request.files['image']
    if not allowed(f.filename):
        return jsonify({'error':'jpg/png 만 허용'}), 400
    path, tmp = _save_temp(f)
    try:
        metrics = extract_metrics(path)
        b64 = annotate_image(path, metrics)
        return jsonify({'metrics':metrics,'annotated_image':f"data:image/jpeg;base64,{b64}"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.route('/api/v1/interpret', methods=['POST'])
def api_interpret():
    if 'image' not in request.files:
        return jsonify({'error':'image 필요'}), 400
    llm_choice = request.form.get('llm','gpt').lower()
    if llm_choice not in LLM_FUNCS:
        return jsonify({'error':'llm 값은 gpt 또는 gemini'}), 400
    f = request.files['image']
    if not allowed(f.filename):
        return jsonify({'error':'jpg/png 만 허용'}), 400
    path, tmp = _save_temp(f)
    try:
        metrics = extract_metrics(path)
        b64_img = annotate_image(path, metrics)
        # 비동기 LLM
        fut = POOL.submit(LLM_FUNCS[llm_choice], metrics)
        interp = fut.result(timeout=60)
        s3_url = upload_to_s3("")  # 현재는 "" 반환
        return jsonify({
            'metrics':metrics,
            'interpretation':interp,
            'annotated_image':f"data:image/jpeg;base64,{b64_img}",
            'annotated_image_url': s3_url
        })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.route('/api/v1/models')
def api_models():
    return jsonify({'available_models': list(LLM_FUNCS.keys())})

@app.route('/health')
def health():
    return jsonify({'status':'healthy'})

# ── 엔트리 포인트 ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    # PORT 환경변수(리버스 프록시용) 없으면 5000
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
