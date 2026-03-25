from datetime import datetime, timezone
from db.universe import InfluxCfg, get_universe_asof

cfg = InfluxCfg(
    url="http://localhost:8086",
    token="rCKxoofYcu3G63P8GyOVf1I1lEr9Z-KGyw1Z6Td2fWtol5EI4DauJnPebwlZAyjXbGsADNOaZRVY52ekazLnQA==",
    org="quant_org",
    bucket="quant_bucket",
)

print(get_universe_asof(cfg, datetime(2018, 1, 1, tzinfo=timezone.utc)))
print(get_universe_asof(cfg, datetime(2021, 1, 1, tzinfo=timezone.utc)))
print(get_universe_asof(cfg, datetime(2021, 1, 2, tzinfo=timezone.utc)))

