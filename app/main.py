from __future__ import annotations

import json, math, sqlite3, uuid, wave, struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.ml_core import load_model

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data' / 'samarthan.db'
MODEL = ROOT / 'models' / 'model.pkl'
METRICS = ROOT / 'models' / 'metrics.json'
UPLOADS = ROOT / 'data' / 'uploads'
UPLOADS.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(exist_ok=True)

app = FastAPI(title='Project Samarthan', version='0.6.0')
app.mount('/static', StaticFiles(directory=ROOT / 'app' / 'static'), name='static')
templates = Jinja2Templates(directory=str(ROOT / 'app' / 'templates'))

RISK_ORDER = {'stable': 0, 'elevated': 1, 'high': 2, 'critical': 3}
RISK_SCORE = {'stable': 18, 'elevated': 42, 'high': 68, 'critical': 88}
LANGUAGES = ['English', 'Hindi', 'Marathi', 'Bengali', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Gujarati', 'Punjabi', 'Odia', 'Assamese']


def now(): return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def action_for(label):
    return {
      'stable':'Routine follow-up; continue periodic check-ins.',
      'elevated':'Counsellor review recommended; verify support and next case milestone.',
      'high':'Priority human review; consider counselling, legal-aid/witness-support or safety planning.',
      'critical':'Urgent human review; assess immediate safety and authorised protection/relocation pathways.'
    }[label]

def init_db():
    with conn() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS cases (
          id TEXT PRIMARY KEY, name TEXT, category TEXT, district TEXT, language TEXT,
          baseline REAL DEFAULT 22, case_stage TEXT, safety_status TEXT DEFAULT 'monitored', created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS checkins (
          id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, created_at TEXT, channel TEXT,
          text TEXT, risk_label TEXT, score REAL, model_conf REAL, baseline_delta REAL,
          engagement REAL, latency REAL, voice_rms REAL, voice_zcr REAL, voice_seconds REAL,
          explanation TEXT, recommended_action TEXT, duress INTEGER DEFAULT 0,
          FOREIGN KEY(case_id) REFERENCES cases(id)
        );
        CREATE TABLE IF NOT EXISTS actions (
          id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, created_at TEXT,
          actor TEXT, action TEXT, notes TEXT
        );
        CREATE TABLE IF NOT EXISTS audits (
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, actor TEXT, event TEXT, detail TEXT
        );
        ''')
        count = c.execute('SELECT COUNT(*) n FROM cases').fetchone()['n']
        if count == 0:
            seed = [
              ('SAM-1042','Asha K.','Intimidated witness','Mumbai Suburban','Marathi',31,'Pre-testimony','monitored'),
              ('SAM-1088','Ravi M.','SC/ST PoA complaint','Pune','Hindi',27,'Investigation','monitored'),
              ('SAM-1120','Meena R.','Atrocity victim','Nashik','Marathi',24,'Rehabilitation','monitored'),
              ('SAM-1147','Demo Critical','Intimidated witness','Thane','English',20,'Pre-testimony','priority')
            ]
            c.executemany('INSERT INTO cases(id,name,category,district,language,baseline,case_stage,safety_status,created_at) VALUES(?,?,?,?,?,?,?,?,?)', [(a,b,c1,d,e,f,g,h,now()) for a,b,c1,d,e,f,g,h in seed])
            # Seed history gives the dashboard a longitudinal story.
            hist = [
              ('SAM-1042','I am okay but I am a little tense about the next hearing','elevated',46),
              ('SAM-1042','I am sleeping normally and feel supported','stable',23),
              ('SAM-1042','I feel more worried about testimony and cannot focus well','high',71),
              ('SAM-1088','The delay is stressful but I can manage','elevated',44),
              ('SAM-1088','I feel supported by my family today','stable',21),
              ('SAM-1120','I am anxious about the compensation delay','elevated',45),
              ('SAM-1147','Someone is threatening me right now and I need urgent help','critical',91)
            ]
            for cid, text, label, score in hist:
                c.execute('INSERT INTO checkins(case_id,created_at,channel,text,risk_label,score,model_conf,baseline_delta,engagement,latency,explanation,recommended_action,duress) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                          (cid, now(), 'Demo Seed', text, label, score, .92, score-20, 0, 0, 'Seeded demonstration record', action_for(label), 1 if label=='critical' else 0))
            c.execute('INSERT INTO audits(created_at,actor,event,detail) VALUES(?,?,?,?)', (now(),'system','DEMO_SEED','Seeded four cases and longitudinal check-in history for demonstration.'))
init_db()


def model():
    if not MODEL.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT/'scripts'/'train_model.py')], check=True)
    return load_model(MODEL)



def voice_stats(path: Path):
    try:
        with wave.open(str(path), 'rb') as w:
            n = w.getnframes(); rate = w.getframerate(); width = w.getsampwidth(); channels = w.getnchannels()
            raw = w.readframes(n)
        if width != 2 or channels != 1 or not raw:
            return {'rms': 0, 'zcr': 0, 'seconds': round(n/max(rate,1),2), 'note':'WAV is not 16-bit mono; limited stats.'}
        samples = struct.unpack('<' + 'h'*(len(raw)//2), raw[:(len(raw)//2)*2])
        rms = (sum(x*x for x in samples) / max(len(samples),1)) ** 0.5
        # zero crossing rate over signed 16-bit samples
        vals = samples
        crossings = sum(1 for a,b in zip(vals, vals[1:]) if (a<0)!=(b<0))
        zcr = crossings/max(len(vals)-1,1)
        return {'rms': round(rms,2), 'zcr': round(zcr,4), 'seconds': round(n/max(rate,1),2), 'note':'Engineering audio statistics only.'}
    except Exception as e:
        return {'rms':0,'zcr':0,'seconds':0,'note':f'Audio read skipped: {e}'}


def analyse(text: str, case_id: str, engagement: float, latency: float, duress: bool, voice: dict):
    m = model()
    probs = m.predict_proba([text])[0]
    classes = list(m.clf.classes_)
    p = {k: float(v) for k,v in zip(classes, probs)}
    label = max(p, key=p.get)
    confidence = p[label]
    base = RISK_SCORE[label]
    with conn() as c:
        row = c.execute('SELECT baseline FROM cases WHERE id=?', (case_id,)).fetchone()
    baseline = float(row['baseline']) if row else 25
    previous = None
    with conn() as c:
        r = c.execute('SELECT score FROM checkins WHERE case_id=? ORDER BY id DESC LIMIT 1', (case_id,)).fetchone()
        if r: previous = float(r['score'])
    baseline_delta = (base - baseline) if previous is None else (base - previous) * .55 + (base - baseline) * .45
    engagement_delta = max(0, min(12, (engagement-55) * .16))
    latency_delta = max(0, min(10, (latency-30) * .12))
    voice_delta = 0
    if voice.get('rms',0) > 7000: voice_delta += 3
    if voice.get('zcr',0) > .16: voice_delta += 2
    if duress: voice_delta += 18
    raw = base + baseline_delta*.22 + engagement_delta + latency_delta + voice_delta
    score = max(0, min(100, round(raw,1)))
    if score >= 82: final = 'critical'
    elif score >= 62: final = 'high'
    elif score >= 35: final = 'elevated'
    else: final = 'stable'
    terms = []
    try:
        terms = m.top_terms(text, label, limit=5)
    except Exception:
        terms = []
    explanation = json.dumps({'model_top_terms':terms,'model_probabilities':{k:round(v,3) for k,v in p.items()},'baseline_delta':round(baseline_delta,1),'engagement_delta':round(engagement_delta,1),'latency_delta':round(latency_delta,1),'voice_delta':voice_delta,'duress_signal':duress})
    return final, score, confidence, baseline_delta, explanation, action_for(final), p


def audit(actor,event,detail):
    with conn() as c: c.execute('INSERT INTO audits(created_at,actor,event,detail) VALUES(?,?,?,?)',(now(),actor,event,detail))


def base_context(request, **extra):
    return {'request':request,'languages':LANGUAGES, **extra}

@app.get('/', response_class=HTMLResponse)
def dashboard(request: Request):
    with conn() as c:
        cases = c.execute('''SELECT ca.*, COALESCE((SELECT score FROM checkins ch WHERE ch.case_id=ca.id ORDER BY ch.id DESC LIMIT 1), ca.baseline) score,
          COALESCE((SELECT risk_label FROM checkins ch WHERE ch.case_id=ca.id ORDER BY ch.id DESC LIMIT 1),'stable') risk_label FROM cases ca ORDER BY score DESC''').fetchall()
        stats = { 'cases': c.execute('SELECT COUNT(*) n FROM cases').fetchone()['n'], 'checkins':c.execute('SELECT COUNT(*) n FROM checkins').fetchone()['n'], 'priority':c.execute("SELECT COUNT(*) n FROM checkins WHERE risk_label IN ('high','critical') AND id IN (SELECT MAX(id) FROM checkins GROUP BY case_id)").fetchone()['n'], 'actions':c.execute('SELECT COUNT(*) n FROM actions').fetchone()['n'] }
    return templates.TemplateResponse(request, 'dashboard.html', base_context(request,cases=cases,stats=stats))

@app.get('/checkin', response_class=HTMLResponse)
def checkin_page(request: Request):
    with conn() as c: cases=c.execute('SELECT * FROM cases ORDER BY id').fetchall()
    return templates.TemplateResponse(request, 'checkin.html', base_context(request,cases=cases))

@app.post('/checkin')
async def checkin(case_id: str=Form(...), text: str=Form(...), engagement: float=Form(55), latency: float=Form(20), language: str=Form('English'), channel: str=Form('Web/Chat'), duress: bool=Form(False), audio: Optional[UploadFile]=File(None)):
    voice={'rms':0,'zcr':0,'seconds':0,'note':'No audio supplied.'}
    if audio and audio.filename:
        p=UPLOADS/(uuid.uuid4().hex+'.wav'); p.write_bytes(await audio.read()); voice=voice_stats(p)
    label,score,conf,bd,explanation,action,probs=analyse(text,case_id,engagement,latency,duress,voice)
    with conn() as c:
        c.execute('''INSERT INTO checkins(case_id,created_at,channel,text,risk_label,score,model_conf,baseline_delta,engagement,latency,voice_rms,voice_zcr,voice_seconds,explanation,recommended_action,duress) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
          (case_id,now(),f'{channel} · {language}',text,label,score,conf,bd,engagement,latency,voice['rms'],voice['zcr'],voice['seconds'],explanation,action,int(duress)))
    audit('AI-engine','CHECKIN_ANALYSED',f'{case_id}: {label} / {score} / confidence {conf:.2f}')
    return RedirectResponse(f'/case/{case_id}', status_code=303)

@app.get('/case/{case_id}', response_class=HTMLResponse)
def case_page(request: Request, case_id: str):
    with conn() as c:
        case=c.execute('SELECT * FROM cases WHERE id=?',(case_id,)).fetchone()
        checks=c.execute('SELECT * FROM checkins WHERE case_id=? ORDER BY id DESC',(case_id,)).fetchall()
        acts=c.execute('SELECT * FROM actions WHERE case_id=? ORDER BY id DESC',(case_id,)).fetchall()
    if not case: return HTMLResponse('Case not found',404)
    latest=checks[0] if checks else None
    expl=json.loads(latest['explanation']) if latest and latest['explanation'] else {}
    return templates.TemplateResponse(request, 'case.html', base_context(request,case=case,checks=checks,acts=acts,latest=latest,expl=expl))

@app.post('/case/{case_id}/action')
def case_action(case_id: str, action: str=Form(...), notes: str=Form('')):
    with conn() as c: c.execute('INSERT INTO actions(case_id,created_at,actor,action,notes) VALUES(?,?,?,?,?)',(case_id,now(),'Human Reviewer',action,notes))
    audit('Human Reviewer','ACTION_RECORDED',f'{case_id}: {action}')
    return RedirectResponse(f'/case/{case_id}', status_code=303)

@app.get('/queue', response_class=HTMLResponse)
def queue(request: Request):
    with conn() as c:
        rows=c.execute('''SELECT ca.*, ch.score, ch.risk_label, ch.created_at checkin_at, ch.recommended_action FROM cases ca JOIN checkins ch ON ch.id=(SELECT MAX(id) FROM checkins WHERE case_id=ca.id) ORDER BY ch.score DESC''').fetchall()
    return templates.TemplateResponse(request, 'queue.html',base_context(request,rows=rows))

@app.get('/validation', response_class=HTMLResponse)
def validation(request: Request):
    metrics=json.loads(METRICS.read_text()) if METRICS.exists() else {}
    return templates.TemplateResponse(request, 'validation.html',base_context(request,m=metrics))

@app.get('/audit', response_class=HTMLResponse)
def audit_page(request: Request):
    with conn() as c: rows=c.execute('SELECT * FROM audits ORDER BY id DESC LIMIT 80').fetchall()
    return templates.TemplateResponse(request, 'audit.html',base_context(request,rows=rows))

@app.get('/integrations', response_class=HTMLResponse)
def integrations(request: Request):
    adapters=[
      ('NHAA 14566','Inbound case/victim reference + approved communication channel','Mock adapter','Ready for authorised API contract'),
      ('e-Courts','Case stage / hearing / testimony milestone context','Mock adapter','Integration point defined; no live government data'),
      ('PFMS','Relief/compensation status signal for follow-up','Mock adapter','Read-only demo status; production access requires authorisation'),
      ('Bhashini','Multilingual speech/text routing','Adapter interface','Demo language routing; production API credentials required')]
    return templates.TemplateResponse(request, 'integrations.html',base_context(request,adapters=adapters))

@app.get('/api/health')
def health(): return {'status':'ok','service':'Samarthan','model':'TF-IDF + Logistic Regression','environment':'engineering POC'}

@app.get('/api/cases')
def api_cases():
    with conn() as c: rows=c.execute('SELECT * FROM cases').fetchall()
    return [dict(r) for r in rows]

@app.post('/api/duress')
def api_duress(case_id: str, signal: str='#'):
    if signal != '#': return JSONResponse({'accepted':False,'reason':'Demo silent-duress trigger is #'},status_code=400)
    audit('Channel Adapter','SILENT_DURESS',f'{case_id}: DTMF # received; priority human review required.')
    return {'accepted':True,'case_id':case_id,'signal':'#','routing':'priority_human_review'}
