"""Small dependency-light HTTP service for the real Kronos predictor.
No checkpoint is downloaded until /healthz?load=1 or /forecast is requested.
"""
import json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

MAX_CONTEXT = 512
class Runtime:
    def __init__(self): self.lock=threading.Lock(); self.predictor=None; self.error=None
    def load(self):
        with self.lock:
            if self.predictor: return self.predictor
            try:
                tok=KronosTokenizer.from_pretrained(os.getenv('KRONOS_TOKENIZER','NeoQuasar/Kronos-Tokenizer-base'))
                model=Kronos.from_pretrained(os.getenv('KRONOS_MODEL','NeoQuasar/Kronos-small'))
                self.predictor=KronosPredictor(model, tok, max_context=MAX_CONTEXT)
            except Exception as exc:
                self.error=f'{type(exc).__name__}: {exc}'
                raise RuntimeError('Kronos checkpoint unavailable: '+self.error) from exc
            return self.predictor
runtime=Runtime()
def error(msg,status=400): return status, {'error':msg}
def validate(body):
    candles=body.get('candles'); lookback=int(body.get('lookback', min(len(candles or []), 128))); pred_len=int(body.get('pred_len',1))
    if not isinstance(candles,list) or not candles: raise ValueError('candles must be a non-empty array')
    if lookback<1 or lookback>MAX_CONTEXT: raise ValueError('lookback must be between 1 and 512')
    if lookback>len(candles): raise ValueError('lookback exceeds candle count')
    if pred_len<1 or pred_len>512: raise ValueError('pred_len must be between 1 and 512')
    req={'open','high','low','close','volume','timestamp'}
    if not all(req <= set(x) for x in candles[-lookback:]): raise ValueError('each candle requires timestamp, open, high, low, close, volume')
    return candles[-lookback:],lookback,pred_len
def forecast(body):
    candles,lookback,pred_len=validate(body); predictor=runtime.load()
    frame=pd.DataFrame(candles); frame['timestamps']=pd.to_datetime(frame.pop('timestamp'), unit='ms', utc=True)
    x=frame[['open','high','low','close','volume']].astype(float); ts=frame['timestamps']
    last=ts.iloc[-1]; freq=body.get('frequency','1h'); future=pd.date_range(last, periods=pred_len+1, freq=freq, tz='UTC')[1:]
    out=predictor.predict(df=x,x_timestamp=ts,y_timestamp=pd.Series(future),pred_len=pred_len,T=float(body.get('temperature',1.0)),top_p=float(body.get('top_p',0.9)),sample_count=1)
    return {'model':os.getenv('KRONOS_MODEL','NeoQuasar/Kronos-small'),'lookback':lookback,'pred_len':pred_len,'forecast':json.loads(out.to_json(orient='records',date_format='iso'))}
class Handler(BaseHTTPRequestHandler):
    def send(self,status,payload):
        raw=json.dumps(payload).encode(); self.send_response(status); self.send_header('content-type','application/json'); self.send_header('content-length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path=='/healthz':
            self.send(503 if runtime.error else 200,{'status':'error' if runtime.error else 'ok','model_loaded':runtime.predictor is not None,'error':runtime.error}); return
        if self.path=='/readyz':
            try: runtime.load(); self.send(200,{'status':'ready'})
            except Exception as e: self.send(503,{'status':'unavailable','error':str(e)})
            return
        self.send(404,{'error':'not found'})
    def do_POST(self):
        if self.path!='/forecast': self.send(404,{'error':'not found'}); return
        try: body=json.loads(self.rfile.read(int(self.headers.get('content-length','0')))); self.send(200,forecast(body))
        except ValueError as e: self.send(422,{'error':str(e)})
        except RuntimeError as e: self.send(503,{'error':str(e)})
        except Exception as e: self.send(500,{'error':f'forecast failed: {type(e).__name__}: {e}'})
    def log_message(self,*args): pass
def main():
    import sys
    if '--help' in sys.argv:
        print('Kronos HTTP service: GET /healthz, GET /readyz, POST /forecast')
        return
    server=ThreadingHTTPServer(('0.0.0.0',int(os.getenv('KRONOS_PORT','8000'))),Handler); print('kronos service listening'); server.serve_forever()
if __name__=='__main__': main()
