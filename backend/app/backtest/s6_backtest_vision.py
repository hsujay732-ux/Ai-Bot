"""
S6 Backtest — Binance Vision (7-day quick backtest)
====================================================
- Last N days ka data auto download (Binance Vision)
- +1 warmup day for SVP (prev day SVP)
- EMA200 warmup ke liye extra klines fetch (API se)
- Same SVP logic as hbe.py
- Multiple pairs ek saath: comma-separated --symbols

Usage:
  python3 s6_backtest_vision.py
  python3 s6_backtest_vision.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 7 --tf 5m
  python3 s6_backtest_vision.py --symbol ETHUSDT --days 7 --tf 5m
  python3 s6_backtest_vision.py   # symbols input interactively if none given
"""

import os, csv, math, zipfile, argparse, time, io, gc
import requests
from datetime import datetime, timezone, timedelta, date
from collections import defaultdict
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_URL      = "https://data.binance.vision/data/futures/um/daily"
BINANCE_API   = "https://fapi.binance.com"
DATA_DIR      = "./binance_data"
RESULTS_DIR   = "./backtest_results"

RR            = 14.0
VALUE_AREA_PC = 0.70
EMA_PERIOD    = 200
EMA_TOUCH_BLK = 15
RETEST_BLK    = 5
HIGH_DELTA_X  = 2.0
LOW_DELTA_X   = 0.5
SWING_LB      = 10
MAX_SL_BLK    = 20
SETUP_EXPIRE  = 50
MAX_CONSEC    = 2

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
G="\033[92m"; R="\033[91m"; Y="\033[93m"
GR="\033[90m"; B="\033[1m"; X="\033[0m"

def cs(p): return p * 0.0001
def ist(ms): return datetime.fromtimestamp(ms/1000, tz=IST).strftime("%Y-%m-%d %H:%M")
def uday(ms): return datetime.fromtimestamp(ms/1000, tz=UTC).strftime("%Y-%m-%d")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── DATE HELPERS ─────────────────────────────────────────────────────────────
def get_days(n_days):
    """
    Aaj se pichle n_days + 1 warmup day (SVP ke liye)
    Returns: (all_days, backtest_days)
    """
    today = datetime.now(tz=UTC).date()
    # Yesterday tak (today ki candle incomplete)
    end   = today - timedelta(days=1)
    start = end - timedelta(days=n_days - 1)
    warmup_start = start - timedelta(days=1)  # 1 extra day for prev SVP

    all_days = []
    cur = warmup_start
    while cur <= end:
        all_days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    bt_days = [d for d in all_days if d >= start.strftime("%Y-%m-%d")]
    return all_days, bt_days, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────
def download_file(url, dest):
    if os.path.exists(dest):
        return True
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 404: return False
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536): f.write(chunk)
        return True
    except Exception as e:
        print(f"  DL error: {e}")
        if os.path.exists(dest): os.remove(dest)
        return False

def zdir(symbol, dtype, tf=None):
    tag = f"{symbol}_klines_{tf}" if tf else f"{symbol}_aggTrades"
    return os.path.join(DATA_DIR, tag)

def zpath(symbol, dtype, day, tf=None):
    d = zdir(symbol, dtype, tf)
    fname = f"{symbol}-{tf}-{day}.zip" if tf else f"{symbol}-{dtype}-{day}.zip"
    return os.path.join(d, fname)

def dl_day(symbol, dtype, day, tf=None):
    if tf:
        url = f"{BASE_URL}/{dtype}/{symbol}/{tf}/{symbol}-{tf}-{day}.zip"
    else:
        url = f"{BASE_URL}/{dtype}/{symbol}/{symbol}-{dtype}-{day}.zip"
    os.makedirs(zdir(symbol, dtype, tf), exist_ok=True)
    zp = zpath(symbol, dtype, day, tf)
    return zp if download_file(url, zp) else None

# ─── EMA WARMUP KLINES (Binance API se) ───────────────────────────────────────
def fetch_ema_warmup_klines(symbol, tf, before_ms, limit=220):
    """
    Backtest start se pehle EMA200 warmup ke liye
    Binance API se historical klines fetch karo
    """
    url = f"{BINANCE_API}/fapi/v1/klines"
    params = {
        "symbol":    symbol,
        "interval":  tf,
        "endTime":   before_ms - 1,
        "limit":     limit
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        candles = []
        for k in r.json():
            candles.append({
                "open_time":  int(k[0]),
                "open":       float(k[1]),
                "high":       float(k[2]),
                "low":        float(k[3]),
                "close":      float(k[4]),
                "volume":     float(k[5]),
                "close_time": int(k[6]),
                "ist_time":   ist(int(k[0])),
                "warmup":     True   # flag — backtest mein use nahi hoga
            })
        print(f"  EMA warmup: {len(candles)} candles fetched from API")
        return candles
    except Exception as e:
        print(f"  EMA warmup fetch error: {e}")
        return []

# ─── PARSE KLINES ZIP ─────────────────────────────────────────────────────────
def parse_klines_zip(zp):
    candles = []
    try:
        with zipfile.ZipFile(zp, "r") as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as f:
                for row in csv.reader(io.TextIOWrapper(f, "utf-8")):
                    if not row or not row[0].isdigit(): continue
                    candles.append({
                        "open_time":  int(row[0]),
                        "open":       float(row[1]),
                        "high":       float(row[2]),
                        "low":        float(row[3]),
                        "close":      float(row[4]),
                        "volume":     float(row[5]),
                        "close_time": int(row[6]),
                        "ist_time":   ist(int(row[0])),
                        "warmup":     False
                    })
    except Exception as e:
        print(f"  Klines parse error: {e}")
    return candles

# ─── FOOTPRINT STREAMING ──────────────────────────────────────────────────────
def build_fp_streaming(zp, day_candles):
    """RAM-safe streaming parse — ek line ek baar"""
    sorted_c = sorted(day_candles, key=lambda c: c["open_time"])
    if not sorted_c: return {}

    bounds   = [(c["open_time"], c["close_time"], cs(c["close"])) for c in sorted_c]
    fp_map   = {c["open_time"]: defaultdict(lambda: {"ask":0.0,"bid":0.0}) for c in sorted_c}
    ci = 0

    try:
        with zipfile.ZipFile(zp, "r") as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as raw:
                for row in csv.reader(io.TextIOWrapper(raw, "utf-8")):
                    if not row or not row[0].isdigit(): continue
                    try:
                        ts    = int(row[5])
                        price = float(row[1])
                        qty   = float(row[2])
                        maker = row[6].strip().lower() in ("true","1")
                    except: continue

                    # Advance candle pointer
                    while ci < len(bounds)-1 and ts > bounds[ci][1]: ci += 1

                    ot, ct, sz = bounds[ci]
                    if ts < ot or ts > ct or sz <= 0: continue

                    level = round(math.floor(price/sz)*sz, 8)
                    if maker: fp_map[ot][level]["bid"] += qty
                    else:     fp_map[ot][level]["ask"] += qty
    except Exception as e:
        print(f"  Streaming error: {e}")

    return {ot: dict(v) for ot, v in fp_map.items() if v}

# ─── EMA ──────────────────────────────────────────────────────────────────────
def calc_ema(candles):
    emas = [None]*len(candles); k = 2/(EMA_PERIOD+1)
    if len(candles) < EMA_PERIOD: return emas
    emas[EMA_PERIOD-1] = sum(c["close"] for c in candles[:EMA_PERIOD])/EMA_PERIOD
    for i in range(EMA_PERIOD, len(candles)):
        emas[i] = candles[i]["close"]*k + emas[i-1]*(1-k)
    return emas

# ─── SVP — same as hbe.py ─────────────────────────────────────────────────────
def calc_svp(vp):
    if not vp: return None,None,None
    tv=sum(vp.values()); tgt=tv*VALUE_AREA_PC
    poc=max(vp,key=vp.get); lvls=sorted(vp.keys())
    pi=lvls.index(poc); va={poc}; vv=vp[poc]; lo,hi=pi-1,pi+1
    while vv<tgt:
        lv=vp[lvls[lo]] if lo>=0 else 0
        hv=vp[lvls[hi]] if hi<len(lvls) else 0
        if lv==0 and hv==0: break
        if hv>=lv:
            if hi<len(lvls): va.add(lvls[hi]); vv+=hv; hi+=1
        else:
            if lo>=0: va.add(lvls[lo]); vv+=lv; lo-=1
    return poc, max(va), min(va)

def build_svp(candles, fp_by_ot):
    """
    hbe.py jaisa SVP build:
    - Har UTC day ke sabhi candles ka footprint volume sum karo
    - calc_svp se POC, VAH, VAL nikalo
    - warmup candles bhi SVP mein contribute karte hain
      (taaki pehle backtest day ka prev_svp mile)
    """
    dc = defaultdict(list)
    for c in candles:
        dc[uday(c["open_time"])].append(c)

    svp = {}
    for day in sorted(dc.keys()):
        vp = defaultdict(float)
        for c in dc[day]:
            for lvl, v in fp_by_ot.get(c["open_time"], {}).items():
                vp[lvl] += v["bid"] + v["ask"]
        if not vp: continue
        poc, vah, val = calc_svp(dict(vp))
        if poc:
            svp[day] = {"poc":poc, "vah":vah, "val":val}
            print(f"  SVP {day}: VAH={vah:.2f} POC={poc:.2f} VAL={val:.2f}")
    return svp

def prev_svp(c, svp):
    """hbe.py se exact same — previous day ka SVP"""
    day  = uday(c["open_time"])
    days = sorted(svp.keys())
    if day not in days: return None
    i = days.index(day)
    return svp[days[i-1]] if i > 0 else None

# ─── FOOTPRINT HELPERS ────────────────────────────────────────────────────────
def aad(fp):
    d=[abs(v["ask"]-v["bid"]) for v in fp.values()]
    return sum(d)/len(d) if d else 0

def find_absorption(fp, side):
    if not fp: return None
    avg=aad(fp)
    if avg==0: return None
    lvls=sorted(fp.keys())
    size=lvls[1]-lvls[0] if len(lvls)>1 else 0.0001
    if side=="BUY":
        reds =[(l,v) for l,v in fp.items() if v["ask"]-v["bid"]<=-avg*HIGH_DELTA_X]
        greys=[(l,v) for l,v in fp.items() if abs(v["ask"]-v["bid"])<avg*LOW_DELTA_X]
        for rl,_ in reds:
            for gl,_ in greys:
                if abs(rl-gl)<=size*2: return (rl,gl)
    else:
        greens=[(l,v) for l,v in fp.items() if v["ask"]-v["bid"]>=avg*HIGH_DELTA_X]
        greys =[(l,v) for l,v in fp.items() if abs(v["ask"]-v["bid"])<avg*LOW_DELTA_X]
        for gl,_ in greens:
            for grl,_ in greys:
                if abs(gl-grl)<=size*2: return (gl,grl)
    return None

def find_two_attempts(fp, side):
    if not fp: return None
    avg=aad(fp)
    if avg==0: return None
    if side=="BUY":
        gs=sorted([l for l,v in fp.items() if v["ask"]-v["bid"]>=avg*HIGH_DELTA_X])
        return (gs[0],gs[-1]) if len(gs)>=2 else None
    else:
        rs=sorted([l for l,v in fp.items() if v["ask"]-v["bid"]<=-avg*HIGH_DELTA_X],reverse=True)
        return (rs[0],rs[-1]) if len(rs)>=2 else None

def chk_cls(c,atts,side):
    if not atts: return False
    return c["close"]>max(atts) if side=="BUY" else c["close"]<min(atts)

def chk_lvl(c,level,sz,side):
    tol=sz*RETEST_BLK
    if side=="BUY": return (c["low"]<=level+tol) and c["close"]>level
    else:           return (c["high"]>=level-tol) and c["close"]<level

def chk_ema(c,ema,sz,side):
    if ema is None: return False
    tol=sz*EMA_TOUCH_BLK
    if side=="BUY": return (c["low"]<=ema+tol) and c["close"]>ema
    else:           return (c["high"]>=ema-tol) and c["close"]<ema

def sw_low(candles,i,lb):  return min(c["low"]  for c in candles[max(0,i-lb):i+1])
def sw_high(candles,i,lb): return max(c["high"] for c in candles[max(0,i-lb):i+1])

# ─── BACKTEST — same logic as hbe.py ──────────────────────────────────────────
def run_backtest(candles, fp_by_ot, svp, emas, bt_start):
    """
    candles mein warmup bhi hain — sirf bt_start ke baad ke trade count karo
    SVP: prev_svp same as hbe.py
    """
    trades=[]; equity=0.0; open_trade=None
    setups={}; day_trade=False; current_day=None
    consec_loss={"BUY":0,"SELL":0}

    print(f"\n[4/4] Backtest (from {bt_start})...")

    for i, c in enumerate(candles[:-1]):
        if emas[i] is None: continue
        ema=emas[i]; sz=cs(c["close"])
        fp=fp_by_ot.get(c["open_time"],{})

        # prev_svp — hbe.py jaisa
        ps=prev_svp(c, svp)
        if not ps: continue

        c_day=uday(c["open_time"])
        if c_day!=current_day:
            current_day=c_day; day_trade=False
            consec_loss={"BUY":0,"SELL":0}; setups.clear()

        if day_trade: continue
        vah=ps["vah"]; val=ps["val"]

        # Open trade resolve
        if open_trade:
            done=False; side=open_trade["side"]
            if side=="BUY":
                if c["low"]<=open_trade["sl"]:
                    equity-=1
                    trades.append({**open_trade,"result":"SL","r":-1.0,
                        "exit_time":ist(c["open_time"]),"exit_price":open_trade["sl"]}); done=True
                elif c["high"]>=open_trade["tp"]:
                    equity+=RR
                    trades.append({**open_trade,"result":"TP","r":RR,
                        "exit_time":ist(c["open_time"]),"exit_price":open_trade["tp"]}); done=True
            else:
                if c["high"]>=open_trade["sl"]:
                    equity-=1
                    trades.append({**open_trade,"result":"SL","r":-1.0,
                        "exit_time":ist(c["open_time"]),"exit_price":open_trade["sl"]}); done=True
                elif c["low"]<=open_trade["tp"]:
                    equity+=RR
                    trades.append({**open_trade,"result":"TP","r":RR,
                        "exit_time":ist(c["open_time"]),"exit_price":open_trade["tp"]}); done=True
            if done:
                last=trades[-1]
                consec_loss[last["side"]]=consec_loss.get(last["side"],0)+1 if last["r"]<0 else 0
                day_trade=True; open_trade=None
            continue

        # Warmup candles pe setup detect mat karo
        if c_day < bt_start: continue

        # Expire setups
        for k in [k for k,s in setups.items() if i-s["created_i"]>SETUP_EXPIRE]:
            setups.pop(k,None)

        # Retest entry
        for key,s in list(setups.items()):
            if i<=s["created_i"]: continue
            eb=s["entry_blk"]; side=s["side"]
            if c["low"]<=eb<=c["high"]:
                if side=="BUY":
                    sl=round(sw_low(candles,i,SWING_LB)-sz,6)
                    if sl>=eb: setups.pop(key,None); continue
                    tp=round(eb+(eb-sl)*RR,6)
                else:
                    sl=round(sw_high(candles,i,SWING_LB)+sz,6)
                    if sl<=eb: setups.pop(key,None); continue
                    tp=round(eb-(sl-eb)*RR,6)
                    if tp>=eb: setups.pop(key,None); continue
                slb=round(abs(eb-sl)/sz)
                if slb>MAX_SL_BLK: setups.pop(key,None); continue
                if consec_loss.get(side,0)>=MAX_CONSEC: setups.pop(key,None); continue

                col=G if side=="BUY" else R
                print(f"  [{ist(c['open_time'])}] {col}{B}{side}{X} "
                      f"Entry={eb:.4f} SL={sl:.4f} TP={tp:.4f} SLb={slb}")
                open_trade={
                    "side":side,"lname":s["lname"],"svp_level":round(s["level"],6),
                    "abs_lvl":round(s.get("abs_lvl",0),6),
                    "attempt1":round(s["atts"][0],6),"attempt2":round(s["atts"][1],6),
                    "entry":round(eb,6),"sl":round(sl,6),"tp":round(tp,6),"sl_blocks":slb,
                    "entry_time":ist(c["open_time"]),"exit_time":"","exit_price":"","result":"","r":0
                }
                day_trade=True; setups.clear(); break

        if day_trade or setups or not fp: continue

        # New setup detection
        for side in ["BUY","SELL"]:
            found=False
            for lname,lp in [("VAL",val),("VAH",vah)]:
                if chk_lvl(c,lp,sz,side):
                    abs_r=find_absorption(fp,side)
                    atts=find_two_attempts(fp,side)
                    if abs_r and atts and chk_cls(c,atts,side):
                        eb=max(atts) if side=="BUY" else min(atts)
                        setups[f"{side}_{lname}"]={
                            "side":side,"lname":lname,"level":lp,
                            "created_i":i,"entry_blk":eb,
                            "abs_lvl":abs_r[0],"atts":atts}
                        print(f"  [{ist(c['open_time'])}] Setup: {side} {lname}={lp:.4f}")
                        found=True; break
            if found: break
            if not found and chk_ema(c,ema,sz,side):
                abs_r=find_absorption(fp,side)
                atts=find_two_attempts(fp,side)
                if abs_r and atts and chk_cls(c,atts,side):
                    eb=max(atts) if side=="BUY" else min(atts)
                    setups[f"{side}_EMA"]={
                        "side":side,"lname":"EMA","level":ema,
                        "created_i":i,"entry_blk":eb,
                        "abs_lvl":abs_r[0],"atts":atts}
                    print(f"  [{ist(c['open_time'])}] Setup: {side} EMA={ema:.4f}")

    if open_trade:
        last=candles[-2]
        if open_trade["side"]=="BUY":
            r=round((last["close"]-open_trade["entry"])/(open_trade["entry"]-open_trade["sl"]),2)
        else:
            r=round((open_trade["entry"]-last["close"])/(open_trade["sl"]-open_trade["entry"]),2)
        equity+=r
        trades.append({**open_trade,"result":"EOD","r":r,
            "exit_time":ist(last["open_time"]),"exit_price":last["close"]})
    return trades, equity

# ─── RESULTS ──────────────────────────────────────────────────────────────────
def show_results(trades, equity, symbol, tf, start, end):
    print(); print("═"*64)
    print(f"  {B}S6 — {symbol} {tf} | {start} → {end}{X}")
    print(f"  RR=1:{int(RR)} | EMA={EMA_PERIOD} | MaxSL={MAX_SL_BLK}blk")
    print("═"*64)
    if not trades:
        print("  Koi trade nahi hua.")
        print("  Hints: HIGH_DELTA_X kam karo / RETEST_BLK badao")
        print("═"*64)
        return None
    for t in trades:
        col=G if t["r"]>0 else (Y if t["result"]=="EOD" else R)
        print(f"  {col}{B}{t['side']:<4}{X} {t['lname']:<4} "
              f"Entry={t['entry']:.2f} SL={t['sl']:.2f} TP={t['tp']:.2f} "
              f"SLb={t['sl_blocks']} {col}{t['result']} R={t['r']}{X}")
        print(f"  {GR}  In={t['entry_time']}  Out={t['exit_time']}{X}")
    print("─"*64)
    wins=sum(1 for t in trades if t["r"]>0)
    loss=sum(1 for t in trades if t["r"]<=0)
    gp=sum(t["r"] for t in trades if t["r"]>0)
    gl=abs(sum(t["r"] for t in trades if t["r"]<0))
    wr=round(wins/len(trades)*100,2); pf=round(gp/gl,2) if gl else 0
    eq=peak=mdd=0
    for t in trades:
        eq+=t["r"]; peak=max(peak,eq); mdd=max(mdd,peak-eq)
    print(f"  Total:{len(trades)} {G}W:{wins}{X} {R}L:{loss}{X} WR:{wr}% PF:{pf}")
    print(f"  MaxDD:{round(mdd,2)}R  Net:{G if equity>0 else R}{round(equity,2)}R{X}")
    print("═"*64)
    fname=os.path.join(RESULTS_DIR,f"s6_{symbol}_{tf}_{start}_{end}.csv")
    with open(fname,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=trades[0].keys())
        w.writeheader(); w.writerows(trades)
        for line in [f"Total,{len(trades)}",f"Wins,{wins}",f"Losses,{loss}",
            f"WR,{wr}%",f"PF,{pf}",f"MaxDD,{round(mdd,2)}",f"NetR,{round(equity,2)}"]:
            f.write(line+"\n")
    print(f"  Saved → {fname}")
    return {"symbol":symbol,"trades":len(trades),"wins":wins,"losses":loss,
            "wr":wr,"pf":pf,"maxdd":round(mdd,2),"net_r":round(equity,2),"csv":fname}

# ─── PER-SYMBOL RUN ────────────────────────────────────────────────────────────
def run_symbol(symbol, args, all_days, bt_start, bt_end):
    symbol = symbol.upper()
    print("\n" + "═"*64)
    print(f"  {B}S6 Backtest — {symbol}{X}")
    print(f"  {symbol} {args.tf} | {bt_start} → {bt_end}")
    print(f"  Download days: {len(all_days)} (incl. 1 SVP warmup)")
    print("═"*64)

    # [1/4] Download aggTrades + klines
    if not args.skip_download:
        print(f"\n[1/4] Downloading {len(all_days)} days...")
        agg_ok=kl_ok=0
        for day in tqdm(all_days, desc=f"{symbol} Download"):
            if dl_day(symbol,"aggTrades",day):         agg_ok+=1
            if dl_day(symbol,"klines",   day,tf=args.tf): kl_ok+=1
            time.sleep(0.05)
        print(f"  aggTrades:{agg_ok}/{len(all_days)}  klines:{kl_ok}/{len(all_days)}")
    else:
        print("[1/4] Download skipped.")

    # [2/4] Klines — Vision data + EMA warmup from API
    print(f"\n[2/4] Parsing klines...")
    raw_candles = []
    for day in all_days:
        zp = zpath(symbol,"klines",day,tf=args.tf)
        if os.path.exists(zp): raw_candles.extend(parse_klines_zip(zp))

    # Sort + dedupe
    seen=set(); vision_candles=[]
    for c in sorted(raw_candles, key=lambda x: x["open_time"]):
        if c["open_time"] not in seen: seen.add(c["open_time"]); vision_candles.append(c)

    # EMA200 warmup — API se pehle ke candles fetch karo
    warmup = []
    if vision_candles:
        warmup = fetch_ema_warmup_klines(symbol, args.tf, vision_candles[0]["open_time"], limit=220)
        # Merge: warmup pehle, vision baad (no duplicates)
        w_times = {c["open_time"] for c in vision_candles}
        warmup  = [c for c in warmup if c["open_time"] not in w_times]
        all_candles = sorted(warmup + vision_candles, key=lambda x: x["open_time"])
    else:
        all_candles = []

    for i,c in enumerate(all_candles): c["idx"]=i
    print(f"  Total candles: {len(all_candles)} "
          f"(warmup={len(warmup)} + vision={len(vision_candles)})")
    if all_candles:
        print(f"  Range: {all_candles[0]['ist_time']} → {all_candles[-1]['ist_time']}")

    if len(all_candles) < EMA_PERIOD+5:
        print(f"  ERROR: {symbol} — Candles bahut kam hain.")
        return None

    # [3/4] Footprint — only for Vision days (warmup ke liye API candles pe fp nahi)
    print(f"\n[3/4] Building footprints (streaming)...")
    fp_by_ot = {}
    day_candles_map = defaultdict(list)
    for c in vision_candles:   # sirf vision candles ke liye fp
        day_candles_map[uday(c["open_time"])].append(c)

    for day in tqdm(all_days, desc=f"{symbol} Footprints"):
        zp = zpath(symbol,"aggTrades",day)
        if not os.path.exists(zp): continue
        dc = day_candles_map.get(day,[])
        if not dc: continue
        fp_by_ot.update(build_fp_streaming(zp, dc))
        gc.collect()

    print(f"  Footprints ready: {len(fp_by_ot)} candles")

    # EMA + SVP
    emas = calc_ema(all_candles)
    print(f"\n  Building SVP...")
    svp  = build_svp(vision_candles, fp_by_ot)   # SVP sirf vision candles se
    print(f"  SVP: {len(svp)} days built")

    if not svp:
        print(f"  ERROR: {symbol} — SVP build nahi hua — aggTrades data check karo")
        return None

    # [4/4] Backtest
    trades, equity = run_backtest(all_candles, fp_by_ot, svp, emas, bt_start)
    return show_results(trades, equity, symbol, args.tf, bt_start, bt_end)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def parse_symbols(args):
    """
    Pairs user input: --symbols (comma-separated) > --symbol > interactive prompt.
    Duplicates removed, order preserved.
    """
    raw = args.symbols or args.symbol
    if not raw:
        raw = input("Pairs enter karein (comma-separated, e.g. BTCUSDT,ETHUSDT): ").strip()
    symbols, seen = [], set()
    for s in raw.split(","):
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s); symbols.append(s)
    return symbols or ["BTCUSDT"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default=None, help="Single pair (legacy), e.g. BTCUSDT")
    parser.add_argument("--symbols", default=None, help="Comma-separated pairs, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--days",    type=int, default=7, help="Kitne din ka backtest")
    parser.add_argument("--tf",      default="5m")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    symbols = parse_symbols(args)
    all_days, bt_days, bt_start, bt_end = get_days(args.days)

    summary = []
    for symbol in symbols:
        res = run_symbol(symbol, args, all_days, bt_start, bt_end)
        if res: summary.append(res)

    if len(symbols) > 1:
        print("\n" + "═"*64)
        print(f"  {B}Combined Summary — {bt_start} → {bt_end} | {args.tf}{X}")
        print("═"*64)
        if not summary:
            print("  Kisi bhi pair mein trade nahi hua.")
        else:
            for s in summary:
                col = G if s["net_r"] > 0 else R
                print(f"  {B}{s['symbol']:<10}{X} Trades:{s['trades']:<3} "
                      f"WR:{s['wr']}%  PF:{s['pf']}  MaxDD:{s['maxdd']}R  "
                      f"Net:{col}{s['net_r']}R{X}")
            total_net = round(sum(s["net_r"] for s in summary), 2)
            col = G if total_net > 0 else R
            print("─"*64)
            print(f"  {B}Total Net across {len(summary)} pair(s): {col}{total_net}R{X}")
        print("═"*64)

if __name__=="__main__":
    main()
