#!/usr/bin/env python3
"""Bounded real FunASR worker Darwin-directory A/B measurement."""
import argparse, ctypes, hashlib, json, os, pickle, re, shutil, signal
import subprocess, time, urllib.request, uuid, wave
from pathlib import Path
CONFSTR={"user":65536,"temp":65537,"cache":65538}
PRODUCTION=Path("/Users/zhanglixing/Production/funasr_spk_server")
PYTHON=PRODUCTION/"venv/bin/python"
SOURCE=PRODUCTION/"temp/samples/4-person-example.m4a"
CASE_TIMEOUT=240
def confstr_path(number):
    fn=ctypes.CDLL(None).confstr; fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t]; fn.restype=ctypes.c_size_t
    size=fn(number,None,0); buf=ctypes.create_string_buffer(size or 1)
    if not size or not fn(number,buf,size): raise RuntimeError("confstr unavailable")
    return Path(buf.value.decode()).resolve()
def health_snapshot():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8767/health",timeout=10) as response:
            body=json.load(response); return {"http_code":response.status,"status":body.get("status"),"checks":body.get("checks"),"active":body.get("active"),"queued":body.get("queued")}
    except Exception as exc: return {"http_code":None,"error_type":type(exc).__name__}
def stop_process(process):
    actions=[]
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM); actions.append("terminate")
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL); actions.append("kill"); process.wait(timeout=5)
    return actions,process.returncode
def bounded_entries(private_dirs,limit=10):
    found=[]
    for root_name,root in private_dirs.items():
        try:
            with os.scandir(root) as iterator:
                for entry in iterator:
                    if len(found)>=limit: return found
                    is_dir=entry.is_dir(follow_symlinks=False); low=entry.name.lower(); found.append({"root":root_name,"name":entry.name,"is_dir":is_dir})
                    is_mps="mps" in low or "metalperformanceshadersgraph" in low
                    if is_dir and is_mps:
                        with os.scandir(entry.path) as nested:
                            for child in nested:
                                if len(found)>=limit: return found
                                found.append({"root":root_name,"name":entry.name+"/"+child.name,"is_dir":child.is_dir(follow_symlinks=False)})
        except FileNotFoundError: continue
    return found
def error_type(log_path):
    try: matches=re.findall(r"错误类型:\s*([A-Za-z_][A-Za-z0-9_.]*)",log_path.read_text(encoding="utf-8",errors="ignore"))
    except OSError: return None
    return matches[-1] if matches else None
def result_hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def run_case(label,env,root,audio_path,private_dirs):
    task_dir=root/"tasks"; task_dir.mkdir(mode=0o700,exist_ok=True)
    task_id=f"directory-ab-{label}-{uuid.uuid4().hex[:10]}"; task_file=task_dir/f"worker_0_{task_id}.task"
    result_file=task_dir/f"worker_0_{task_id}.pkl"; input_file=task_dir/f"{task_id}.wav"; ready_file=task_dir/"worker_0.ready"; log_path=root/f"worker-{label}.log"
    started=time.monotonic(); result_seen=None; command=[str(PYTHON),"src/core/worker_process.py","--worker-id","0","--task-dir",str(task_dir)]
    record={"label":label,"task_id":task_id,"status":"error","worker_command":command,"task_dir":str(task_dir)}
    for stale in task_dir.glob("worker_0_*.task"): stale.unlink()
    for stale in task_dir.glob("directory-ab-*.wav"): stale.unlink()
    ready_file.unlink(missing_ok=True)
    with log_path.open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=PRODUCTION,env=env,stdout=log,stderr=log,text=True,start_new_session=True); record["process_pid"]=process.pid; deadline=started+CASE_TIMEOUT
        while not ready_file.exists() and process.poll() is None and time.monotonic()<deadline: time.sleep(.25)
        if not ready_file.exists():
            actions,code=stop_process(process); return {**record,"status":"timeout" if time.monotonic()>=deadline else "worker_exit","timeout_phase":"ready","process_returncode":code,"termination_actions":actions,"total_sec":round(time.monotonic()-started,3)}
        ready_pid=int(ready_file.read_text().strip()); record.update(ready_sec=round(time.monotonic()-started,3),ready_pid=ready_pid,ready_pid_matches_process=ready_pid==process.pid)
        shutil.copy2(audio_path,input_file); payload={"task_id":task_id,"audio_path":str(input_file),"source_audio_path":str(audio_path),"hotword":"","use_pickle":True}; payload_bytes=json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode(); task_file.write_bytes(payload_bytes); published=time.monotonic()
        record.update(task_payload=payload,task_payload_sha256=hashlib.sha256(payload_bytes).hexdigest())
        while not result_file.exists() and process.poll() is None and time.monotonic()<deadline: time.sleep(.25)
        if result_file.exists():
            result_seen=time.monotonic()-published
            if label=="dirhelper_suffix":
                entries=bounded_entries(private_dirs); matching=[item for item in entries if str(process.pid) in item["name"]]
                mps=[item for item in entries if "mps" in item["name"].lower() or "metalperformanceshadersgraph" in item["name"].lower()]
                record.update(private_scandir_limit=10,private_entries=entries,private_mps_entries=mps,private_pid_matching_entries=matching,private_pid_match=bool(matching))
        if process.poll() is None and time.monotonic()<deadline:
            try: process.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired: pass
        if process.poll() is None:
            actions,code=stop_process(process); record.update(status="timeout",timeout_phase="result",process_returncode=code,termination_actions=actions)
        else: record["process_returncode"]=process.returncode
        record["total_sec"]=round(time.monotonic()-started,3)
        if result_seen is None: record.update(status="worker_exit",error_type=error_type(log_path)); return record
        record["publish_to_result_sec"]=round(result_seen,3)
        try: data=pickle.loads(result_file.read_bytes())
        except Exception as exc: record.update(status="result_read_error",error_type=type(exc).__name__); return record
        record.update(success=bool(data.get("success")),result_worker_pid=data.get("worker_pid"),result_pid_matches_process=data.get("worker_pid")==process.pid,result_task_id_matches=data.get("task_id")==task_id)
        if record["success"]:
            value=data.get("result"); record.update(status="success",segments_count=len(value) if isinstance(value,list) else None,result_type=type(value).__name__,result_sha256=result_hash(value))
        else: record.update(status="failed",error_type=error_type(log_path) or "WorkerError")
    return record
def cleanup(paths):
    result=[]
    for path in paths:
        item={"path":str(path)}
        try: shutil.rmtree(path); item["removed"]=True
        except FileNotFoundError: item.update(removed=False,already_absent=True)
        except Exception as exc: item.update(removed=False,error_type=type(exc).__name__)
        item["exists_after"]=path.exists(); result.append(item)
    return result
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",required=True,type=Path); root=parser.parse_args().root.resolve()
    roots={name:confstr_path(number) for name,number in CONFSTR.items()}; suffix=f"funasr-directory-ab-{uuid.uuid4().hex[:12]}"; private={name:path/suffix for name,path in roots.items()}
    for path in private.values(): path.mkdir(mode=0o700)
    report={"status":"ok","suffix":suffix,"darwin_roots":{k:str(v) for k,v in roots.items()},"private_dirs":{k:str(v) for k,v in private.items()},"health_preflight":health_snapshot(),"case_timeout_sec":CASE_TIMEOUT,"source_audio":str(SOURCE),"worker_python":str(PYTHON),"worker_script":"src/core/worker_process.py","cases":[],"reverse_run":False}
    try:
        health=report["health_preflight"]
        if health.get("http_code")!=200 or health.get("status")!="healthy": raise RuntimeError("production_health_not_healthy")
        values=[health.get(k) for k in ("active","queued") if isinstance(health.get(k),(int,float))]
        if any(value>0 for value in values): report["status"]="blocked_active"; return
        report["health_activity_fields_missing"]=any(health.get(k) is None for k in ("active","queued"))
        if not SOURCE.exists(): raise FileNotFoundError(str(SOURCE))
        audio_path=root/"audio-60s-16k-mono.wav"; ffmpeg=["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(SOURCE),"-t","60","-ac","1","-ar","16000","-c:a","pcm_s16le",str(audio_path)]
        with (root/"ffmpeg.log").open("w",encoding="utf-8") as log: subprocess.run(ffmpeg,cwd=PRODUCTION,stdout=subprocess.DEVNULL,stderr=log,check=True)
        with wave.open(str(audio_path)) as wav: duration=wav.getnframes()/wav.getframerate()
        report["audio"]={"path":str(audio_path),"sha256":hashlib.sha256(audio_path.read_bytes()).hexdigest(),"size_bytes":audio_path.stat().st_size,"duration_sec":round(duration,6),"ffmpeg_argv":ffmpeg}
        base=os.environ.copy(); base.pop("DIRHELPER_USER_DIR_SUFFIX",None); isolated={**base,"DIRHELPER_USER_DIR_SUFFIX":suffix,"TMPDIR":str(private["temp"])+"/"}
        report["cases"].append(run_case("default",base,root,audio_path,private)); report["cases"].append(run_case("dirhelper_suffix",isolated,root,audio_path,private)); report["status"]="ok" if all(c.get("status")=="success" for c in report["cases"]) else "incomplete"
    except Exception as exc: report.update(status="error",error_type=type(exc).__name__)
    finally:
        for path in (root/"tasks",root/"audio-60s-16k-mono.wav"):
            if path.is_dir(): shutil.rmtree(path)
            elif path.exists(): path.unlink()
        report["cleanup"]=cleanup(list(private.values())); report["health_postflight"]=health_snapshot(); (root/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,separators=(",",":")))
if __name__=="__main__": main()
