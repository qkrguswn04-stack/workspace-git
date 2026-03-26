import pymysql
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# jinja2 pandas fastapi pymysql python-multipart uvicorn

db_config ={
     'host':'3.25.202.124'
    ,'user':'myuser'
    ,'password':'myuser'
    ,'database':'mydb'
}

def get_connection():
    return pymysql.connect(**db_config)

app = FastAPI(title="Love Letter App (FastAPI)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- 메인 화면 (레터 조회) ---
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)
# --- 레터 작성 폼 화면 ---
@app.get("/form", response_class=HTMLResponse)
def letter_form(request: Request):
    return templates.TemplateResponse(name="letterForm.html", request=request)
# --- 레터 데이터 DB 저장 (INSERT) ---
@app.post("/sendLetter", response_class=HTMLResponse)
def send_letter(
    request: Request,
    toNm: str = Form(...),      # 절대 제출해! (필수 값)
    email: str = Form(...),
    messageOne: str = Form(" "), # 제출 안하면 빈칸으로 제출해!
    messageTwo: str = Form(" "),
    messageThree: str = Form(" ")
):
    if toNm and email:
        conn = get_connection()
        cursor = conn.cursor()
        sql = '''
            INSERT INTO cards (email, nm, message1, message2, message3)
            VALUES (%s, %s, %s, %s, %s)
        '''
        cursor.execute(sql, (email, toNm, messageOne, messageTwo, messageThree))
        conn.commit()
        conn.close()
        
        return RedirectResponse(url="/", status_code=303)
        
    return templates.TemplateResponse(name="letterForm.html", request=request, context={"error": "이메일과 받는 분 이름이 필요합니다."})

# --- 레터 데이터 조회 ---
@app.post("/get_card", response_class=HTMLResponse)
def get_card(request: Request
           , toNm: str = Form(...)
           , email: str = Form(...)): 
    if toNm != '' and email !='':
        conn = get_connection()
        card = pd.read_sql(sql='''
            SELECT *
            FROM cards
            WHERE email = %s
            AND   nm = %s
        ''', con=conn, params=(email,toNm))
        conn.close()
        
        if not card.empty:
            return templates.TemplateResponse(name="letter_result.html", request=request, context={
                "message1": card.iloc[0]['message1'],
                "message2": card.iloc[0]['message2'],
                "message3": card.iloc[0]['message3'],
                "nm": toNm,
                "email": email
            })
        else:
            return templates.TemplateResponse(name="index.html", request=request, context={"error": "등록된 러브레터가 없습니다 ㅠㅠ"})
    else:
        return templates.TemplateResponse(name="index.html",request=request, context={"error": "이름과 이메일을 정확히 입력하세요."})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
