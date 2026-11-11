#!/usr/bin/env python3
"""Train genuine SCiForest (isotree SCiForest mode) from sensor_history.csv.
Usage: pip install -r sciforest-requirements.txt
       python sciforest_train.py sensor_history.csv sciforest-results.json
CSV must contain tank ID, timestamp, temperature, turbidity columns.
"""
import csv, json, sys
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from isotree import IsolationForest

ALIASES={
 "tank":["tank_id","tankid","aquarium_id","device_id","id","수족관id","수조id"],
 "time":["timestamp","measured_at","recorded_at","datetime","time","date","측정시각","측정시간"],
 "temp":["temp","temperature","water_temp","water_temperature","수온","온도"],
 "turb":["turb","turbidity","ntu","탁도"]
}
def key(row, names):
 norm={str(k).strip().lower().replace(" ","").replace("_",""):k for k in row}
 for n in names:
  found=norm.get(n.lower().replace("_",""))
  if found is not None:return found
 return None

def main():
 if len(sys.argv)<2:raise SystemExit("Usage: python sciforest_train.py sensor_history.csv [output.json]")
 source=sys.argv[1]; output=sys.argv[2] if len(sys.argv)>2 else "sciforest-results.json"
 with open(source,encoding="utf-8-sig",newline="") as f:
  reader=csv.DictReader(f)
  cols={k:key({h:None for h in reader.fieldnames},v) for k,v in ALIASES.items()}
  if not all(cols.values()):raise ValueError("Required CSV columns: tank_id, timestamp, temp, turbidity (aliases supported). Detected: "+str(reader.fieldnames))
  groups=defaultdict(list)
  for row in reader:
   try:
    t=float(row[cols["temp"]]);u=float(row[cols["turb"]])
    if not np.isfinite(t) or not np.isfinite(u):continue
    groups[str(row[cols["tank"]]).strip()].append((str(row[cols["time"]]),t,u))
   except (ValueError,TypeError):continue
 results=[]
 for tank,rows in sorted(groups.items()):
  rows.sort(key=lambda r:r[0])
  if len(rows)<30:continue
  x=np.array([[r[1],r[2]] for r in rows],dtype=float)
  median=np.median(x,axis=0);scale=np.std(x,axis=0);scale[scale<1e-9]=1
  features=(x-median)/scale
  # SCiForest: randomized oblique splits with gain criterion, not ordinary axis-aligned iForest.
  model=IsolationForest(ndim=2,ntry=10,coef_type="normal",sample_size=min(256,len(rows)),ntrees=200,random_seed=42)
  model.fit(features)
  raw=np.asarray(model.predict(features,output="score"),dtype=float)
  # Higher scores indicate more anomalous samples; empirical percentile is relative to training history.
  ranks=np.argsort(np.argsort(raw,kind="stable"),kind="stable")
  percentile=100*ranks/ max(1,len(raw)-1)
  for i,r in enumerate(rows):
   results.append({"tankId":tank,"timestamp":r[0],"temp":r[1],"turb":r[2],"score":round(float(percentile[i]),1),"rawScore":round(float(raw[i]),6),"anomaly":bool(percentile[i]>=95)})
 payload={"schema":"aquapulse-sciforest-v1","algorithm":"SCiForest","generatedAt":datetime.now(timezone.utc).isoformat(),"scoreMeaning":"Per-tank empirical anomaly percentile (0-100), not probability or water quality risk","threshold":95,"rows":results}
 with open(output,"w",encoding="utf-8") as f:json.dump(payload,f,ensure_ascii=False,separators=(",",":"))
 print(f"Saved {len(results)} scored measurements from {len(groups)} tanks to {output}")
if __name__=="__main__":main()
