import sys
import os
import re
import json
from datetime import datetime
import configparser
import pyodbc


# Azure Text Analytics
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    FollowEvent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage
)


from translate import Translator
import pandas as pd
import logging

# 建立 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 建立輸出格式與 handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



#Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

#Config Azure Analytics
credential = AzureKeyCredential(config['AzureLanguage']['API_KEY'])

app = Flask(__name__)

channel_access_token = config['Line']['CHANNEL_ACCESS_TOKEN']
channel_secret = config['Line']['CHANNEL_SECRET']
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

handler = WebhookHandler(channel_secret)

configuration = Configuration(
    access_token=channel_access_token
)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # parse webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'
    
@app.route("/")
def index():
    return "Hello, Flask is running!"



# 翻譯函式：中文 → 英文
def translate_to_english(text):
    try:
        translator = Translator(from_lang="zh", to_lang="en")
        return translator.translate(text)
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return text  # fallback: 使用原文





def log_healthcare_result(result, original_text, translated_text, gemini_result,
                          prob, percent_to_threshold, final_message,
                          threshold, model_version, log_dir="logs", prefix="healthcare"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{prefix}_{ts}.json")

    payload = {
        "meta": {
            "original_text": original_text,
            "translated_text": translated_text,
            "gemini_result": gemini_result,
            "probability": prob,
            "percent_to_threshold": percent_to_threshold,
            "final_message": final_message,
            "threshold": threshold,
            "model_version": model_version
        },
        "azure_result": []
    }

    for doc in result:
        if doc.is_error:
            payload["azure_result"].append({"id": getattr(doc, "id", None), "error": True})
            continue

        entities = []
        for e in doc.entities:
            entities.append({
                "text": e.text,
                "normalized_text": getattr(e, "normalized_text", None),
                "category": e.category,
                "confidence_score": e.confidence_score
            })

        relations = []
        for r in doc.entity_relations:
            relations.append({
                "relation_type": r.relation_type,
                "roles": [{"name": role.name, "text": role.entity.text} for role in r.roles]
            })

        payload["azure_result"].append({
            "id": getattr(doc, "id", None),
            "entities": entities,
            "entity_relations": relations
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


import re

def extract_number(text):
    match = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(match.group()) if match else None


def classify_total_cholesterol(value, unit="mg/dL"):
    if value is None: return None
    unit_clean = (unit or "mg/dL").lower()
    if "mg/l" in unit_clean: value = value / 10.0
    elif "mmol" in unit_clean: value = value * 38.67
    if value < 200: return 1
    elif value <= 239: return 2
    else: return 3


def classify_ldl(value, unit="mg/dL"):
    if value is None: return None
    unit_clean = (unit or "mg/dL").lower()
    if "mg/l" in unit_clean: value = value / 10.0
    elif "mmol" in unit_clean: value = value * 38.67
    if value < 100: return 1
    elif value <= 159: return 2
    else: return 3

    
def classify_blood_sugar(value, unit="mg/dL"):
    """血糖分類 (依據 ADA/WHO/IDF 標準)
    value: 數值
    unit: 單位 ("mg/dL" 或 "mmol/L")
    回傳: 1=正常, 2=異常(前期糖尿病), 3=超異常(糖尿病)
    """
    print("Blood Sugar Value:", value)
    print("Blood Sugar Unit:", unit)
    if value is None:
        return None

    if not unit:
        unit = "mg/dL"

    # 標準化單位字串
    unit_clean = unit.strip().lower().replace(" ", "")
    print(unit_clean)
    # 如果單位包含 mmol → 換算成 mg/dL
    if "mmol" in unit_clean:
        value = value * 18
    # 如果單位包含 mg → 直接用 mg/dL 判斷
    elif "mg" in unit_clean:
        pass  # 不做換算，直接用 value
    print(value)
    # 分類邏輯 (mg/dL)
    if value < 100:            # < 5.6 mmol/L
        return 1
    elif 100 <= value <= 125:  # 5.6–6.9 mmol/L
        return 2
    elif value >= 126:         # ≥ 7.0 mmol/L
        return 3
    else:
        return None



    
def gender_text(val):
    if val == 1:
        return "女"
    elif val == 2:
        return "男"
    return "未知"


def cholesterol_text(val):
    if val == 1:
        return "正常(總膽固醇200 mg/dL以下 or 低密度脂蛋白100 mg/dL以下)"
    elif val == 2:
        return "偏高(總膽固醇200~239 mg/dL or 低密度脂蛋白100~159 mg/dL)"
    elif val == 3:
        return "超高(總膽固醇239 mg/dL以上 or 低密度脂蛋白159 mg/dL以上)"
    return "未提供"


def gluc_text(val):
    if val == 1:
        return "正常(100 mg/dL or 5.6 mmol/L以下)"
    elif val == 2:
        return "偏高(100~125 mg/dL or 5.6–6.9 mmol/L)"
    elif val == 3:
        return "超高(126 mg/dL or 7.0 mmol/L以上)"
    return "未提供"

def parse_health_result(docs, record_id=0):
    record = {
        "id": record_id,
        "age": None,
        "gender": None,
        "height": None,
        "weight": None,
        "ap_hi": None,
        "ap_lo": None,
        "cholesterol": None,
        "gluc": None,
    }

    tc_class = None
    ldl_class = None

    for doc in docs:
        # 年齡 / 性別
        for e in doc.entities:
            if e.category == "Age":
                record["age"] = extract_number(e.text)
            elif e.category == "Gender":
                t = (e.text or "").lower()
                if any(k in t for k in ["female", "woman", "girl"]):
                    record["gender"] = 1
                elif any(k in t for k in ["male", "man", "boy"]):
                    record["gender"] = 2

        # 建立 exam→{value, unit}
        exam_map = {}
        for rel in doc.entity_relations:
            if rel.relation_type == "ValueOfExamination":
                exam = [r.entity.text.lower() for r in rel.roles if r.name == "Examination"]
                val  = [r.entity.text for r in rel.roles if r.name == "Value"]
                if exam and val:
                    exam_map.setdefault(exam[0], {})["value"] = extract_number(val[0])

            elif rel.relation_type == "UnitOfExamination":
                exam = [r.entity.text.lower() for r in rel.roles if r.name == "Examination"]
                unit = [r.entity.text for r in rel.roles if r.name == "Unit"]
                if exam and unit:
                    exam_map.setdefault(exam[0], {})["unit"] = unit[0]

        # 用 exam_map 判斷
        for exam_name, payload in exam_map.items():
            val = payload.get("value")
            unit = payload.get("unit")

            if "height" in exam_name:
                record["height"] = float(val) if val is not None else None
            elif "weight" in exam_name:
                record["weight"] = float(val) if val is not None else None
            elif "systolic" in exam_name:
                record["ap_hi"] = val
            elif "diastolic" in exam_name:
                record["ap_lo"] = val
            elif "glucose" in exam_name or "blood sugar" in exam_name:
                record["gluc"] = classify_blood_sugar(val, unit)
            elif "total cholesterol" in exam_name:
                tc_class = classify_total_cholesterol(val, unit)
            elif "ldl" in exam_name:
                ldl_class = classify_ldl(val, unit)

    # 整合膽固醇
    if tc_class is not None or ldl_class is not None:
        record["cholesterol"] = max(tc_class or 0, ldl_class or 0)

    return record
    

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


def run_cardio_model(df, threshold):
    """
    調用 cardio_model.h5，將預處理後的數據丟入模型分析，
    輸出機率值與分類結果。
    """
    model_path = "cardio_model.h5"
    # 載入模型
    model = load_model(model_path)

    # ====== 前處理流程 ======
    # 1. 清理異常值
    def delete_strange(df):
        # 年齡限制
        df = df[(df['age']  >= 0) & (df['age']  <= 110)]
        # 身高體重
        df = df[(df['height'] >= 140) & (df['height'] <= 220)]
        df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
        # 血壓範圍
        df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
        df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
        df = df[df['ap_hi'] > df['ap_lo']]
        
        return df
    df  = delete_strange(df)
    if df.empty:
        print("⚠️ 前處理後沒有任何樣本，無法進行模型推論")
        return pd.DataFrame(), model_path   # 回傳空結果，避免 scaler.transform 報錯

    def drop_unwanted_features(df, features_to_drop):
        print(f"Dropping features: {features_to_drop}")

        if features_to_drop in list(df.columns):
            df = df.drop(columns=features_to_drop)

        return df
    # 2. 移除不需要的欄位
    df = drop_unwanted_features(df, 'id')

    # 3. 特徵工程
    df['age_years'] = df['age']
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['health_index'] = (df['active'] * 1) - (df['smoke'] * 0.5) - (df['alco'] * 0.5)
    df['cholesterol_gluc_interaction'] = df['cholesterol'] * df['gluc']
    df['map'] = df['ap_lo'] + (df['pulse_pressure'] / 3)

    def hypertension_stage(row):
        if row['ap_hi'] >= 140 or row['ap_lo'] >= 90:
            return 2
        elif row['ap_hi'] >= 130 or row['ap_lo'] >= 85:
            return 1
        else:
            return 0
    df['hypertension_stage'] = df.apply(hypertension_stage, axis=1)

    def metabolic_syndrome(row):
        count = 0
        if row['bmi'] >= 30: count += 1
        if row['ap_hi'] >= 130 or row['ap_lo'] >= 85: count += 1
        if row['cholesterol'] >= 2: count += 1
        if row['gluc'] >= 2: count += 1
        return count
    df['metabolic_syndrome'] = df.apply(metabolic_syndrome, axis=1)
    df['risk_index'] = (
    (df['age_years'] > 50).astype(int) +
    (df['bmi'] > 30).astype(int) +
    (df['pulse_pressure'] > 60).astype(int) +
    (df['cholesterol'] >= 2).astype(int) +
    (df['gluc'] >= 2).astype(int) +
    df['smoke'] +
    df['alco']
    )
    df['gender_smoke_interaction'] = df['gender'] * df['smoke']
    df['gender_alco_interaction'] = df['gender'] * df['alco']
    df = drop_unwanted_features(df, 'age')
    
    # 5. 標準化
    import joblib

    # 載入訓練時的 scaler
    scaler = joblib.load("scaler.pkl")

    # 只做 transform，不要再 fit
    X_stan = pd.DataFrame(scaler.transform(df), columns=df.columns)


    # 6. 移除部分特徵
    X_stan = drop_unwanted_features(X_stan, 'gender')
    X_stan = drop_unwanted_features(X_stan, 'alco')
    X_stan = drop_unwanted_features(X_stan, 'gender_alco_interaction')
    X_stan = drop_unwanted_features(X_stan, 'risk_index')
    X_stan = drop_unwanted_features(X_stan, 'pulse_pressure')
    X_stan = drop_unwanted_features(X_stan, 'height')
    X_stan = drop_unwanted_features(X_stan, 'smoke')
    X_stan = drop_unwanted_features(X_stan, 'active')
    print(X_stan)
    # ====== 模型推論 ======
    probs = model.predict(X_stan.values).reshape(-1)

    preds = (probs > threshold).astype(int)

    # ====== 輸出 ======
    df_result = pd.DataFrame({
        "probability": probs,
        "prediction": preds
    })

    return df_result, model_path




@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):

    user_text = event.message.text
    
    # 先翻譯成英文
    translated_text = translate_to_english(user_text)
    print(f"Translated text: {translated_text}")
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(text="收到您的資訊，數據分析中，請稍候...")
                    ]
             )
        )
    
    # 再丟進 Azure Healthcare API 分析
    health_result = azure_health(
        user_input=translated_text,       # 給 Azure/Gemini 分析用的英文
        user_id=event.source.user_id,
        translated_text=translated_text,  # 翻譯後文字
        original_text=user_text           # ✅ 新增：原始 LINE 輸入
    )

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.push_message(
            PushMessageRequest(
                to=event.source.user_id,
                messages=[
                    TextMessage(text=health_result)
                ]
            )
        )



import time
import traceback

def get_db_connection(retries=3, delay=2):
    """
    建立並回傳 pyodbc 連線（含重試）。
    會使用 config.ini 的 Database 區段。
    """
    server = config.get('Database', 'server', fallback='127.0.0.1')
    port = config.getint('Database', 'port', fallback=1433)
    user = config.get('Database', 'user', fallback='SA')
    password = config.get('Database', 'password', fallback=None)
    database = config.get('Database', 'database', fallback='cardiodb')

    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server},{port};"
        f"UID={user};PWD={password};"
        f"DATABASE={database};Encrypt=no;"
    )

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            logger.debug("Attempting DB connect to %s:%s (attempt %d)", server, port, attempt)
            conn = pyodbc.connect(conn_str, timeout=30)
            logger.debug("DB connect succeeded to %s:%s", server, port)
            return conn
        except Exception as e:
            last_exc = e
            logger.warning("DB connect attempt %d failed: %s", attempt, e)
            if attempt < retries:
                time.sleep(delay * attempt)
    logger.exception("DB connect failed after %d attempts", retries)
    raise last_exc



def save_to_sql(user_id, input_text, translated_text, gemini_model,
                record, prob, percent_to_threshold, final_message,
                threshold, status="success"):
    """
    將分析結果寫入 CardioAnalysisLogs。
    使用 get_db_connection()，並在例外時記錄完整 traceback。
    """
    start_time = time.time()
    conn = None
    try:
        logger.info("Saving analysis to DB for user=%s", user_id)
        conn = get_db_connection()
        cursor = conn.cursor()
        execution_time_ms = int((time.time() - start_time) * 1000)
        cursor.execute("""
            INSERT INTO CardioAnalysisLogs (
                user_id, input_text, translated_text, gemini_model,
                age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active,
                probability, percent_to_threshold, final_message,
                azure_version, model_threshold, execution_time_ms, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, input_text, translated_text, gemini_model,
            record.get("age"), record.get("gender"), record.get("height"), record.get("weight"),
            record.get("ap_hi"), record.get("ap_lo"), record.get("cholesterol"), record.get("gluc"),
            record.get("smoke"), record.get("alco"), record.get("active"),
            prob, percent_to_threshold, final_message,
            "Azure Text Analytics v3.2", threshold, execution_time_ms, status
        ))
        conn.commit()
        logger.info("Inserted CardioAnalysisLogs for user %s (time %d ms)", user_id, execution_time_ms)
    except Exception as e:
        logger.exception("SQL insert failed for user=%s: %s", user_id, e)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                logger.exception("Failed to close DB connection in save_to_sql")






import google.generativeai as genai
def gemini_lifestyle_analysis(text, model_name="gemini-2.5-flash-lite"):
    try:
        genai.configure(api_key=config['Gemini']['API_KEY'])
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
        f"""請判定句子中的描述是否會有菸癮、酒癮跟運動習慣，並輸出 JSON: {{ "smoke": , "alco": , "active": }} 
        規則:
        - 明確有 → 1
        - 明確沒有 → 0
        - 未提及 → None
        僅依句子內容分析，禁止推測，僅輸出 JSON。
        句子: "{text}"
         """

        )

        raw_text = response.text.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        print(f"Gemini({model_name}) raw response:", raw_text)

        try:
            result = json.loads(raw_text)

            def normalize_int(val):
                if val in [1, "1", True]:
                    return 1
                elif val in [0, "0", False]:
                    return 0
                else:
                    return None

            return {
                "smoke": normalize_int(result.get("smoke")),
                "alco": normalize_int(result.get("alco")),
                "active": normalize_int(result.get("active")),
            }

        except Exception as e:
            logging.warning(f"Gemini JSON parse failed: {e}")
            # ... 保留原本的 fallback 邏輯 ...
            return {"smoke": None, "alco": None, "active": None}

    except Exception as api_error:
        logging.error(f"Gemini API failed: {api_error}")
        return {"smoke": None, "alco": None, "active": None}


def get_last_nonnull_record(user_id, record):
    """
    取得 CardioAnalysisLogs 中每個欄位最近一次非 NULL 的值，
    並把缺值從 last_record 填回到 record（同時記錄來源日期到 record['source_dates']）。
    回傳填補後的 record（若 record 為 None，回傳 last_record）。
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
        SELECT
            A.age, A.age_date,
            G.gender, G.gender_date,
            H.height, H.height_date,
            W.weight, W.weight_date,
            HI.ap_hi, HI.ap_hi_date,
            LO.ap_lo, LO.ap_lo_date,
            C.cholesterol, C.cholesterol_date,
            GL.gluc, GL.gluc_date,
            S.smoke, S.smoke_date,
            AL.alco, AL.alco_date,
            AC.active, AC.active_date
        FROM (SELECT 1 AS dummy) D
        OUTER APPLY (
            SELECT TOP 1 age, CONVERT(varchar(30), created_at, 126) AS age_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND age IS NOT NULL
            ORDER BY created_at DESC
        ) A
        OUTER APPLY (
            SELECT TOP 1 gender, CONVERT(varchar(30), created_at, 126) AS gender_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND gender IS NOT NULL
            ORDER BY created_at DESC
        ) G
        OUTER APPLY (
            SELECT TOP 1 height, CONVERT(varchar(30), created_at, 126) AS height_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND height IS NOT NULL
            ORDER BY created_at DESC
        ) H
        OUTER APPLY (
            SELECT TOP 1 weight, CONVERT(varchar(30), created_at, 126) AS weight_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND weight IS NOT NULL
            ORDER BY created_at DESC
        ) W
        OUTER APPLY (
            SELECT TOP 1 ap_hi, CONVERT(varchar(30), created_at, 126) AS ap_hi_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND ap_hi IS NOT NULL
            ORDER BY created_at DESC
        ) HI
        OUTER APPLY (
            SELECT TOP 1 ap_lo, CONVERT(varchar(30), created_at, 126) AS ap_lo_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND ap_lo IS NOT NULL
            ORDER BY created_at DESC
        ) LO
        OUTER APPLY (
            SELECT TOP 1 cholesterol, CONVERT(varchar(30), created_at, 126) AS cholesterol_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND cholesterol IS NOT NULL
            ORDER BY created_at DESC
        ) C
        OUTER APPLY (
            SELECT TOP 1 gluc, CONVERT(varchar(30), created_at, 126) AS gluc_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND gluc IS NOT NULL
            ORDER BY created_at DESC
        ) GL
        OUTER APPLY (
            SELECT TOP 1 smoke, CONVERT(varchar(30), created_at, 126) AS smoke_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND smoke IS NOT NULL
            ORDER BY created_at DESC
        ) S
        OUTER APPLY (
            SELECT TOP 1 alco, CONVERT(varchar(30), created_at, 126) AS alco_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND alco IS NOT NULL
            ORDER BY created_at DESC
        ) AL
        OUTER APPLY (
            SELECT TOP 1 active, CONVERT(varchar(30), created_at, 126) AS active_date
            FROM CardioAnalysisLogs
            WHERE user_id = ? AND active IS NOT NULL
            ORDER BY created_at DESC
        ) AC
        """
        params = (user_id,) * 11
        cursor.execute(sql, params)
        columns = [col[0] for col in cursor.description]
        row = cursor.fetchone()
        last_record = dict(zip(columns, row)) if row else None

        if last_record is None:
            return record if record is not None else None

        if record is not None:
            fields = [
                "age","gender","height","weight","ap_hi","ap_lo",
                "cholesterol","gluc","smoke","alco","active"
            ]
            for key in fields:
                value = last_record.get(key)
                date = last_record.get(f"{key}_date")
                if record.get(key) is None and value is not None:
                    record[key] = value
                    record.setdefault("source_dates", {})[key] = date
            return record

        return last_record

    except Exception as e:
        logger.exception("get_last_nonnull_record failed for user_id=%s", user_id)
        return record if record is not None else None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                logger.exception("Failed to close DB connection in get_last_nonnull_record")




from datetime import datetime, timedelta

from datetime import datetime, timedelta

def format_date(src_date):
    if not src_date:
        return None

    if isinstance(src_date, str):
        try:
            # 把 " " 換成 "T"，並移除多餘空格
            clean = src_date.strip().replace(" ", "T").replace("+08:00", "+08:00")
            dt = datetime.fromisoformat(clean)
            # 直接加 8 小時 (假設 DB 存的是 UTC)
            dt = dt + timedelta(hours=8)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return src_date

    elif isinstance(src_date, datetime):
        dt = src_date + timedelta(hours=8)
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    return str(src_date)





def fmt_with_source(rec, key, unit=None):
    val = rec.get(key)
    if val is None:
        return "未提供"
    suffix = f" ({unit})" if unit else ""
    src_date = (rec.get("source_dates") or {}).get(key)
    if src_date:
        return f"{val}{suffix} (沿用{format_date(src_date)}的紀錄)"
    return f"{val}{suffix}"

def yn_with_source(rec, key):
    val = rec.get(key)
    text = "有" if val == 1 else ("無" if val == 0 else "未知")
    src_date = (rec.get("source_dates") or {}).get(key)
    if src_date:
        return f"{text} (沿用{format_date(src_date)}的紀錄)"
    return text

def azure_health(user_input, user_id, translated_text, original_text, gemini_model="gemini-2.5-flash-lite"):
    text_analytics_client = TextAnalyticsClient(
        endpoint=config['AzureLanguage']['END_POINT'],
        credential=credential
    )
    documents = [user_input]

    poller = text_analytics_client.begin_analyze_healthcare_entities(documents, language="en")
    result = poller.result()
    result_list = list(result)

    docs = [doc for doc in result_list if not doc.is_error]
    if not docs:
        return "分析失敗：文件解析錯誤或未抽取到資料"

    record = parse_health_result(docs, record_id=0)

    # Gemini → lifestyle habits
    gemini_result = gemini_lifestyle_analysis(original_text, model_name=gemini_model)
    record["smoke"] = gemini_result.get("smoke", None)
    record["alco"] = gemini_result.get("alco", None)
    record["active"] = gemini_result.get("active", None)

    # 補值邏輯
    llm_keys = ["smoke", "alco", "active"]
    filled = get_last_nonnull_record(user_id, record)
    
    for k in llm_keys:
        # 如果本次分析結果是 None → 用舊紀錄值
        if record.get(k) is None and filled.get(k) is not None:
            record[k] = filled.get(k)
            # 標記這是沿用的紀錄
            record.setdefault("source_dates", {})[k] = filled["source_dates"].get(k)
    
        # 如果本次分析結果不是 None → 保留 LLM 值，但仍然補上舊紀錄日期
        elif record.get(k) is not None and filled.get("source_dates", {}).get(k):
            record.setdefault("source_dates", {})[k] = filled["source_dates"][k]
    

    if any(v is None for k, v in record.items() if k not in ["source_dates"]):
        record = get_last_nonnull_record(user_id, record)

    clean_record = {k: v for k, v in record.items() if k != "source_dates"}
    df = pd.DataFrame([clean_record])

    print("-----輸入資料-----")
    print(df)
    threshold = 0.5152
    df_result, model_path = run_cardio_model(df, threshold=threshold)
    print(df_result)

    output_lines = []
    if df_result.empty:
        output_lines.append("⚠️ 模型未能生成預測結果，請檢查輸入資料。")
        save_to_sql(
            user_id=user_id,
            input_text=original_text,
            translated_text=translated_text,
            gemini_model=gemini_model,
            record=clean_record,
            prob=None,
            percent_to_threshold=None,
            final_message="⚠️ 模型未能生成預測結果",
            threshold=threshold,
            status="failed"
        )
        return "\n".join(output_lines)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"health_records_{user_id}_{ts}.csv", index=False, encoding="utf-8-sig")

    # 組合 LINE 輸出
    output_lines.append("---NLP分析結果---")
    output_lines.append(f"年齡:{fmt_with_source(record, 'age')}")
    gender_val = record.get("gender")
    gender_date = (record.get("source_dates") or {}).get("gender")
    gender_str = gender_text(gender_val)
    if gender_date:
        output_lines.append(f"性別:{gender_str} (沿用{format_date(gender_date)}的紀錄)")
    else:
        output_lines.append(f"性別:{gender_str}")
    output_lines.append(f"身高:{fmt_with_source(record, 'height', 'cm')}")
    output_lines.append(f"體重:{fmt_with_source(record, 'weight', 'kg')}")
    output_lines.append(f"收縮壓:{fmt_with_source(record, 'ap_hi', 'mmHg')}")
    output_lines.append(f"舒張壓:{fmt_with_source(record, 'ap_lo', 'mmHg')}")
    chol_val = record.get("cholesterol")
    chol_date = (record.get("source_dates") or {}).get("cholesterol")
    chol_str = cholesterol_text(chol_val)
    if chol_date:
        output_lines.append(f"血脂:{chol_str} (沿用{format_date(chol_date)}的紀錄)")
    else:
        output_lines.append(f"血脂:{chol_str}")
    gluc_val = record.get("gluc")
    gluc_date = (record.get("source_dates") or {}).get("gluc")
    gluc_str = gluc_text(gluc_val)
    if gluc_date:
        output_lines.append(f"血糖:{gluc_str} (沿用{format_date(gluc_date)}的紀錄)")
    else:
        output_lines.append(f"血糖:{gluc_str}")

    output_lines.append(f"---LLM({gemini_model})分析結果---")
    output_lines.append(f"煙癮:{yn_with_source(record, 'smoke')}")
    output_lines.append(f"酒癮:{yn_with_source(record, 'alco')}")
    output_lines.append(f"運動習慣:{yn_with_source(record, 'active')}")

    prob = None
    percent_to_threshold = None
    final_message = None
    for _, row in df_result.iterrows():
        prob = row["probability"]
        if prob >= threshold:
            percent_to_threshold = ((prob - threshold) / (1 - threshold)) * 100
            risk_emoji = "❤️"
            final_message = f"{risk_emoji}您有心血管疾病風險!! (高於閥值{percent_to_threshold:.1f}%)"
        else:
            percent_to_threshold = ((threshold - prob) / threshold) * 100
            risk_emoji = "💚"
            final_message = f"{risk_emoji}您無心血管疾病風險。 (低於閥值{percent_to_threshold:.1f}%)"

        output_lines.append(f"🤖深度學習模型預測機率: {prob:.4f}")
        output_lines.append(final_message)

    log_path = log_healthcare_result(
        result_list,
        original_text=original_text,
        translated_text=translated_text,
        gemini_result=gemini_result,
        prob=prob,
        percent_to_threshold=percent_to_threshold,
        final_message=final_message,
        threshold=threshold,
        model_version=model_path
    )
    print(f"Healthcare analysis logged to: {log_path}")

    save_to_sql(
        user_id=user_id,
        input_text=original_text,
        translated_text=translated_text,
        gemini_model=gemini_model,
        record=clean_record,
        prob=prob,
        percent_to_threshold=percent_to_threshold,
        final_message=final_message,
        threshold=threshold,
        status="success"
    )

    return "\n".join(output_lines)
    



if __name__ == "__main__":
    app.run()

