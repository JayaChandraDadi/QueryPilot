"""Builds the evaluation set (question -> gold SQL pairs) and the few-shot example bank.

Questions come from 25 parameterized templates over the sample warehouse
(years, regions, channels, segments, categories, ...). The few-shot bank uses
different question shapes, so evaluation does not reward copying an example.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import random

REV = "SUM(oi.quantity * oi.unit_price * (1 - oi.discount))"
YEARS = [2023, 2024, 2025]
REGIONS = ["NORTH", "SOUTH", "EAST", "WEST"]
CHANNELS = [("online", "online"), ("in_store", "in-store")]
SEGMENTS = ["Consumer", "Corporate", "Small Business"]
CATEGORIES = ["Electronics", "Home", "Grocery", "Clothing", "Sports", "Toys", "Beauty"]
TITLES = ["Store Manager", "Sales Associate", "Cashier", "Stock Associate", "Assistant Manager"]
CITIES = ["Minneapolis", "Chicago", "Detroit", "Houston", "Atlanta", "Miami", "New York", "Boston", "Philadelphia",
          "Seattle", "San Francisco", "Denver"]


def _year(y):
    return f"o.order_date >= DATE '{y}-01-01' AND o.order_date < DATE '{y + 1}-01-01'"


TEMPLATES = [
    ("easy", "What was total revenue in {y}?", [YEARS],
     lambda y: f"SELECT {REV} AS revenue FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
               f"WHERE o.status = 'completed' AND {_year(y)}"),
    ("medium", "What was total revenue for stores in the {r} region in {y}?", [REGIONS, YEARS],
     lambda r, y: f"SELECT {REV} AS revenue FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
                  f"JOIN stores s ON o.store_id = s.store_id WHERE o.status = 'completed' AND s.region = '{r}' "
                  f"AND {_year(y)}"),
    ("medium", "Which product category had the highest revenue in {y}?", [YEARS],
     lambda y: f"SELECT p.category FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
               f"JOIN products p ON oi.product_id = p.product_id WHERE o.status = 'completed' AND {_year(y)} "
               f"GROUP BY p.category ORDER BY {REV} DESC LIMIT 1"),
    ("hard", "Show revenue by product category for {r} stores.", [REGIONS],
     lambda r: f"SELECT p.category, {REV} AS revenue FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
               f"JOIN products p ON oi.product_id = p.product_id JOIN stores s ON o.store_id = s.store_id "
               f"WHERE o.status = 'completed' AND s.region = '{r}' GROUP BY p.category"),
    ("easy", "How many orders were placed through the {ch} channel in {y}?", [CHANNELS, YEARS],
     lambda ch, y: f"SELECT COUNT(*) AS orders FROM orders o WHERE o.channel = '{ch[0]}' AND {_year(y)}"),
    ("easy", "How many customers are in the {seg} segment?", [SEGMENTS],
     lambda seg: f"SELECT COUNT(*) AS customers FROM customers WHERE segment = '{seg}'"),
    ("easy", "How many customers live in the {r} region?", [REGIONS],
     lambda r: f"SELECT COUNT(*) AS customers FROM customers WHERE region = '{r}'"),
    ("medium", "What was the average order value of completed orders in {y}?", [YEARS],
     lambda y: f"SELECT {REV} / COUNT(DISTINCT o.order_id) AS avg_order_value FROM order_items oi "
               f"JOIN orders o ON oi.order_id = o.order_id WHERE o.status = 'completed' AND {_year(y)}"),
    ("medium", "List the top 5 products by units sold in {y}.", [YEARS],
     lambda y: f"SELECT p.product_name, SUM(oi.quantity) AS units FROM order_items oi "
               f"JOIN orders o ON oi.order_id = o.order_id JOIN products p ON oi.product_id = p.product_id "
               f"WHERE o.status = 'completed' AND {_year(y)} GROUP BY p.product_name ORDER BY units DESC, "
               f"p.product_name LIMIT 5"),
    ("medium", "Which store had the most completed orders in {y}?", [YEARS],
     lambda y: f"SELECT s.store_name FROM orders o JOIN stores s ON o.store_id = s.store_id "
               f"WHERE o.status = 'completed' AND {_year(y)} GROUP BY s.store_name ORDER BY COUNT(*) DESC LIMIT 1"),
    ("medium", "What percentage of orders were returned in {y}?", [YEARS],
     lambda y: f"SELECT 100.0 * SUM(CASE WHEN o.status = 'returned' THEN 1 ELSE 0 END) / COUNT(*) AS return_pct "
               f"FROM orders o WHERE {_year(y)}"),
    ("medium", "How many orders were cancelled at stores in the {r} region?", [REGIONS],
     lambda r: f"SELECT COUNT(*) AS cancelled FROM orders o JOIN stores s ON o.store_id = s.store_id "
               f"WHERE o.status = 'cancelled' AND s.region = '{r}'"),
    ("medium", "What is the average discount on {c} items sold?", [CATEGORIES],
     lambda c: f"SELECT AVG(oi.discount) AS avg_discount FROM order_items oi "
               f"JOIN products p ON oi.product_id = p.product_id WHERE p.category = '{c}'"),
    ("hard", "Which brand sold the most units in the {c} category?", [CATEGORIES],
     lambda c: f"SELECT p.brand FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
               f"JOIN products p ON oi.product_id = p.product_id WHERE o.status = 'completed' AND p.category = '{c}' "
               f"GROUP BY p.brand ORDER BY SUM(oi.quantity) DESC LIMIT 1"),
    ("medium", "How many employees work at stores in the {r} region?", [REGIONS],
     lambda r: f"SELECT COUNT(*) AS employees FROM employees e JOIN stores s ON e.store_id = s.store_id "
               f"WHERE s.region = '{r}'"),
    ("easy", "What is the average salary of employees with the job title {t}?", [TITLES],
     lambda t: f"SELECT AVG(salary) AS avg_salary FROM employees WHERE job_title = '{t}'"),
    ("easy", "How many stores are there in {city}?", [CITIES],
     lambda city: f"SELECT COUNT(*) AS stores FROM stores WHERE city = '{city}'"),
    ("hard", "Which 3 customers in the {seg} segment spent the most in {y}? Show their customer ids.",
     [SEGMENTS, YEARS],
     lambda seg, y: f"SELECT c.customer_id FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
                    f"JOIN customers c ON o.customer_id = c.customer_id WHERE o.status = 'completed' "
                    f"AND c.segment = '{seg}' AND {_year(y)} GROUP BY c.customer_id ORDER BY {REV} DESC LIMIT 3"),
    ("medium", "What was the revenue for each month of {y}?", [YEARS],
     lambda y: f"SELECT EXTRACT(MONTH FROM o.order_date) AS month, {REV} AS revenue FROM order_items oi "
               f"JOIN orders o ON oi.order_id = o.order_id WHERE o.status = 'completed' AND {_year(y)} "
               f"GROUP BY month ORDER BY month"),
    ("easy", "How many distinct customers placed an order in {y}?", [YEARS],
     lambda y: f"SELECT COUNT(DISTINCT o.customer_id) AS customers FROM orders o WHERE {_year(y)}"),
    ("medium", "What is the total revenue from {c} products sold through the {ch} channel?", [CATEGORIES, CHANNELS],
     lambda c, ch: f"SELECT {REV} AS revenue FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
                   f"JOIN products p ON oi.product_id = p.product_id WHERE o.status = 'completed' "
                   f"AND p.category = '{c}' AND o.channel = '{ch[0]}'"),
    ("easy", "What is the average credit limit of customers in the {r} region?", [REGIONS],
     lambda r: f"SELECT AVG(credit_limit) AS avg_credit_limit FROM customers WHERE region = '{r}'"),
    ("easy", "How many customers signed up in {y}?", [[2020, 2021, 2022, 2023, 2024]],
     lambda y: f"SELECT COUNT(*) AS customers FROM customers WHERE signup_date >= DATE '{y}-01-01' "
               f"AND signup_date < DATE '{y + 1}-01-01'"),
    ("hard", "Which city's stores generated the most revenue in {y}?", [YEARS],
     lambda y: f"SELECT s.city FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
               f"JOIN stores s ON o.store_id = s.store_id WHERE o.status = 'completed' AND {_year(y)} "
               f"GROUP BY s.city ORDER BY {REV} DESC LIMIT 1"),
    ("hard", "How many units of {c} products were sold at {r} stores in {y}?", [CATEGORIES, REGIONS, YEARS],
     lambda c, r, y: f"SELECT SUM(oi.quantity) AS units FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
                     f"JOIN products p ON oi.product_id = p.product_id JOIN stores s ON o.store_id = s.store_id "
                     f"WHERE o.status = 'completed' AND p.category = '{c}' AND s.region = '{r}' AND {_year(y)}"),
]

FEW_SHOT = [
    ("How many products are in each category?",
     "SELECT category, COUNT(*) AS products FROM products GROUP BY category ORDER BY category"),
    ("What is the average list price per brand?",
     "SELECT brand, AVG(unit_price) AS avg_price FROM products GROUP BY brand ORDER BY brand"),
    ("How many orders did each store fulfill in March 2024?",
     "SELECT s.store_name, COUNT(*) AS orders FROM orders o JOIN stores s ON o.store_id = s.store_id "
     "WHERE o.order_date >= DATE '2024-03-01' AND o.order_date < DATE '2024-04-01' GROUP BY s.store_name"),
    ("What was revenue by sales channel last quarter of 2024?",
     f"SELECT o.channel, {REV} AS revenue FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
     "WHERE o.status = 'completed' AND o.order_date >= DATE '2024-10-01' AND o.order_date < DATE '2025-01-01' "
     "GROUP BY o.channel"),
    ("Which three cities have the most customers?",
     "SELECT s.city, COUNT(DISTINCT o.customer_id) AS customers FROM orders o JOIN stores s "
     "ON o.store_id = s.store_id GROUP BY s.city ORDER BY customers DESC LIMIT 3"),
    ("How many employees were hired after 2020 at each job title?",
     "SELECT job_title, COUNT(*) AS hires FROM employees WHERE hire_date >= DATE '2021-01-01' GROUP BY job_title"),
    ("How many customers are there per segment and region?",
     "SELECT segment, region, COUNT(*) AS customers FROM customers GROUP BY segment, region ORDER BY segment, region"),
    ("What share of 2023 orders were placed online?",
     "SELECT 100.0 * SUM(CASE WHEN channel = 'online' THEN 1 ELSE 0 END) / COUNT(*) AS online_pct FROM orders "
     "WHERE order_date >= DATE '2023-01-01' AND order_date < DATE '2024-01-01'"),
    ("What is the most expensive product in each category?",
     "SELECT category, MAX(unit_price) AS max_price FROM products GROUP BY category"),
    ("How many units of Voltix products were sold in completed orders?",
     "SELECT SUM(oi.quantity) AS units FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
     "JOIN products p ON oi.product_id = p.product_id WHERE o.status = 'completed' AND p.brand = 'Voltix'"),
    ("Which region's customers placed the most orders?",
     "SELECT c.region FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.region "
     "ORDER BY COUNT(*) DESC LIMIT 1"),
    ("What was revenue per store region in 2025?",
     f"SELECT s.region, {REV} AS revenue FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
     "JOIN stores s ON o.store_id = s.store_id WHERE o.status = 'completed' AND o.order_date >= DATE '2025-01-01' "
     "AND o.order_date < DATE '2026-01-01' GROUP BY s.region"),
    ("How many line items had a discount of 20%?",
     "SELECT COUNT(*) AS lines FROM order_items WHERE discount = 0.2"),
    ("When did the oldest store open?", "SELECT MIN(opened_date) AS first_opened FROM stores"),
    ("What is the total payroll per store?",
     "SELECT s.store_name, SUM(e.salary) AS payroll FROM employees e JOIN stores s ON e.store_id = s.store_id "
     "GROUP BY s.store_name"),
    ("How many orders were returned per month in 2024?",
     "SELECT EXTRACT(MONTH FROM order_date) AS month, COUNT(*) AS returned FROM orders WHERE status = 'returned' "
     "AND order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01' GROUP BY month ORDER BY month"),
    ("Which segment has the highest average credit limit?",
     "SELECT segment FROM customers GROUP BY segment ORDER BY AVG(credit_limit) DESC LIMIT 1"),
    ("What is the average quantity per order line for Grocery products?",
     "SELECT AVG(oi.quantity) AS avg_qty FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
     "WHERE p.category = 'Grocery'"),
]


def build(out_dir: str | pathlib.Path, per_template: int = 8, seed: int = 11) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    rows = []
    for t_idx, (difficulty, text, params, sql_fn) in enumerate(TEMPLATES):
        combos = list(itertools.product(*params))
        rnd.shuffle(combos)
        for combo in combos[:per_template]:
            fmt = {}
            for ph, value in zip(_placeholders(params), combo):
                fmt[ph] = value[1] if ph == "ch" else value  # channels carry (sql value, wording)
            rows.append({"id": f"t{t_idx:02d}-{len(rows):03d}", "template": t_idx, "difficulty": difficulty,
                         "question": text.format(**fmt), "sql": sql_fn(*combo)})
    q_path = out_dir / "questions.jsonl"
    q_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    e_path = out_dir / "examples.jsonl"
    e_path.write_text("".join(json.dumps({"question": q, "sql": s}) + "\n" for q, s in FEW_SHOT))
    return q_path, e_path


def _placeholders(params):
    names = {id(YEARS): "y", id(REGIONS): "r", id(CHANNELS): "ch", id(SEGMENTS): "seg", id(CATEGORIES): "c",
             id(TITLES): "t", id(CITIES): "city"}
    return [names.get(id(p), "y") for p in params]


def load_jsonl(path: str | pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    q, e = build("data/eval")
    print(f"wrote {q} ({len(load_jsonl(q))} questions) and {e} ({len(load_jsonl(e))} examples)")
