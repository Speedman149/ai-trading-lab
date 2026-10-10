import sys
from pathlib import Path
from datetime import date
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"backend"))
from app.core.schemas import BacktestRequest
from app.data.market import synthetic
for asset in ["AAPL", "BTC/USD"]:
    req = BacktestRequest(asset=asset,start=date(2024,1,2),end=date(2024,2,1))
    synthetic(req).to_csv(root/"data"/"samples"/f"{asset.replace('/', '-')}-daily.csv",index=False)