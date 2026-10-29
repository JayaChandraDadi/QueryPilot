"""Builds the sample retail warehouse (a DuckDB file) used by the demo, tests and eval.

The data is generated from a fixed seed, so every build is identical.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import random

import duckdb
import pyarrow as pa

REGIONS = {
    "NORTH": ["Minneapolis", "Chicago", "Detroit"],
    "SOUTH": ["Houston", "Atlanta", "Miami"],
    "EAST": ["New York", "Boston", "Philadelphia"],
    "WEST": ["Seattle", "San Francisco", "Denver"],
}
CATEGORIES = {
    "Electronics": ["Voltix", "Nimbus", "Arcline"],
    "Home": ["Hearthly", "Oakmere"],
    "Grocery": ["FreshFold", "Greenway"],
    "Clothing": ["Northwind", "Threadline"],
    "Sports": ["Peakform", "Stridewell"],
    "Toys": ["Brightbox"],
    "Beauty": ["Lumen", "Velvetry"],
}
PRICE_RANGE = {"Electronics": (40, 900), "Home": (15, 400), "Grocery": (2, 40), "Clothing": (10, 150),
               "Sports": (12, 300), "Toys": (8, 90), "Beauty": (6, 80)}
SEGMENTS = ["Consumer", "Corporate", "Small Business"]
FIRST = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Ethan", "Isla", "Arjun", "Priya", "Mateo", "Sofia", "Kenji",
         "Amara", "Lucas", "Hana", "Omar", "Elena", "Ravi", "Chloe", "Diego"]
LAST = ["Patel", "Garcia", "Kim", "Nguyen", "Smith", "Okafor", "Rossi", "Chen", "Silva", "Brown", "Haddad",
        "Reddy", "Muller", "Lopez", "Sato", "Cohen"]
JOB_TITLES = ["Store Manager", "Sales Associate", "Cashier", "Stock Associate", "Assistant Manager"]


def _date(rnd: random.Random, start: dt.date, end: dt.date) -> dt.date:
    return start + dt.timedelta(days=rnd.randrange((end - start).days + 1))


def build(path: str | pathlib.Path, seed: int = 7, scale: float = 1.0) -> pathlib.Path:
    """Creates (or replaces) the warehouse database at `path`."""
    path = pathlib.Path(path)
    for p in (path, path.with_suffix(path.suffix + ".wal")):
        if p.exists():
            p.unlink()
    rnd = random.Random(seed)

    stores = {"store_id": [], "store_name": [], "city": [], "region": [], "opened_date": []}
    sid = 1
    for region, cities in REGIONS.items():
        for city in cities:
            for n in range(2):
                stores["store_id"].append(sid)
                stores["store_name"].append(f"{city} #{n + 1}")
                stores["city"].append(city)
                stores["region"].append(region)
                stores["opened_date"].append(_date(rnd, dt.date(2012, 1, 1), dt.date(2022, 6, 30)))
                sid += 1
    store_region = dict(zip(stores["store_id"], stores["region"]))

    n_customers = int(6000 * scale)
    customers = {k: [] for k in ["customer_id", "full_name", "email", "phone", "region", "segment",
                                 "signup_date", "credit_limit"]}
    for cid in range(1, n_customers + 1):
        first, last = rnd.choice(FIRST), rnd.choice(LAST)
        customers["customer_id"].append(cid)
        customers["full_name"].append(f"{first} {last}")
        customers["email"].append(f"{first.lower()}.{last.lower()}{cid}@example.com")
        customers["phone"].append(f"555-{rnd.randrange(100, 1000)}-{rnd.randrange(1000, 10000)}")
        customers["region"].append(rnd.choice(list(REGIONS)))
        customers["segment"].append(rnd.choices(SEGMENTS, weights=[6, 2, 2])[0])
        customers["signup_date"].append(_date(rnd, dt.date(2019, 1, 1), dt.date(2025, 6, 30)))
        customers["credit_limit"].append(float(rnd.choice([500, 1000, 2500, 5000, 10000])))

    products = {k: [] for k in ["product_id", "product_name", "category", "brand", "unit_price"]}
    pid = 1
    for category, brands in CATEGORIES.items():
        for _ in range(int(60 * scale) or 1):
            brand = rnd.choice(brands)
            lo, hi = PRICE_RANGE[category]
            products["product_id"].append(pid)
            products["product_name"].append(f"{brand} {category[:-1] if category.endswith('s') else category} {pid}")
            products["category"].append(category)
            products["brand"].append(brand)
            products["unit_price"].append(round(rnd.uniform(lo, hi), 2))
            pid += 1
    price = dict(zip(products["product_id"], products["unit_price"]))

    orders = {k: [] for k in ["order_id", "customer_id", "store_id", "order_date", "status", "channel"]}
    items = {k: [] for k in ["order_id", "line_no", "product_id", "quantity", "unit_price", "discount"]}
    stores_by_region = {}
    for s, r in store_region.items():
        stores_by_region.setdefault(r, []).append(s)
    n_orders = int(50000 * scale)
    for oid in range(1, n_orders + 1):
        cust = rnd.randrange(1, n_customers + 1)
        region = customers["region"][cust - 1] if rnd.random() < 0.85 else rnd.choice(list(REGIONS))
        orders["order_id"].append(oid)
        orders["customer_id"].append(cust)
        orders["store_id"].append(rnd.choice(stores_by_region[region]))
        orders["order_date"].append(_date(rnd, dt.date(2023, 1, 1), dt.date(2025, 12, 31)))
        orders["status"].append(rnd.choices(["completed", "cancelled", "returned"], weights=[90, 6, 4])[0])
        orders["channel"].append(rnd.choices(["online", "in_store"], weights=[45, 55])[0])
        for line in range(1, rnd.choice([1, 1, 2, 2, 3, 4]) + 1):
            p = rnd.randrange(1, pid)
            items["order_id"].append(oid)
            items["line_no"].append(line)
            items["product_id"].append(p)
            items["quantity"].append(rnd.choice([1, 1, 1, 2, 2, 3, 5]))
            items["unit_price"].append(price[p])
            items["discount"].append(rnd.choice([0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2]))

    employees = {k: [] for k in ["employee_id", "full_name", "store_id", "job_title", "hire_date", "salary"]}
    eid = 1
    for s in stores["store_id"]:
        for title in JOB_TITLES:
            for _ in range(1 if title in ("Store Manager", "Assistant Manager") else 3):
                employees["employee_id"].append(eid)
                employees["full_name"].append(f"{rnd.choice(FIRST)} {rnd.choice(LAST)}")
                employees["store_id"].append(s)
                employees["job_title"].append(title)
                employees["hire_date"].append(_date(rnd, dt.date(2015, 1, 1), dt.date(2025, 6, 30)))
                base = {"Store Manager": 85000, "Assistant Manager": 62000}.get(title, 36000)
                employees["salary"].append(float(round(base * rnd.uniform(0.85, 1.25), -2)))
                eid += 1

    con = duckdb.connect(str(path))
    for name, data in [("stores", stores), ("customers", customers), ("products", products), ("orders", orders),
                       ("order_items", items), ("employees", employees)]:
        tbl = pa.table(data)  # noqa: F841 (referenced by DuckDB's replacement scan)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tbl")
    con.execute("ALTER TABLE order_items ALTER quantity TYPE INTEGER")
    con.close()
    return path


if __name__ == "__main__":
    import sys

    out = build(sys.argv[1] if len(sys.argv) > 1 else "data/warehouse.duckdb")
    print(f"wrote {out}")
