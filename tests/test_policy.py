"""Adversarial governance suite: every way of reaching a denied column must be
blocked, allowed queries must still run, and row filters/masks must hold in
the actual query results."""
import re

import pytest

from querypilot.policy import PolicyViolation, QueryError

DENY = [
    # role, sql
    ("analyst", "select email from customers"),
    ("analyst", "select c.email as contact from customers c"),
    ("analyst", "select upper(phone) from customers"),
    ("analyst", "with x as (select email as e from customers) select e from x"),
    ("analyst", "with a as (select * from customers), b as (select * from a) select customer_id from b"),
    ("analyst", "select * from customers"),
    ("analyst", "select x.* from (select * from customers) x"),
    ("analyst", "select customer_id from customers where email like '%@example.com'"),
    ("analyst", "select count(*) from customers where phone is not null"),
    ("analyst", "select region from customers union select phone from customers"),
    ("analyst", "select region from customers union all select email from customers"),
    ("analyst", "select (select max(salary) from employees) as top_pay"),
    ("analyst", "select store_id, avg(salary) from employees group by store_id"),
    ("analyst", "select o.order_id from orders o join customers c on c.email = o.status"),
    ("analyst", "select customer_id from customers order by email"),
    ("analyst", "select customer_id from customers group by customer_id, phone"),
    ("analyst", "select s from (select salary as s from employees) t"),
    ("analyst", "select customer_id from customers where full_name = 'Ava Patel'"),  # masked used in filter
    ("analyst", "select upper(full_name) from customers"),  # masked transformed
    ("analyst", "select count(distinct full_name) from customers"),
    ("analyst", "select * from read_csv('/etc/passwd')"),
    ("analyst", "select * from read_parquet('s3://bucket/x.parquet')"),
    ("analyst", "select * from information_schema.columns"),
    ("analyst", "delete from orders"),
    ("analyst", "drop table orders"),
    ("analyst", "insert into stores values (1, 'x', 'y', 'z', date '2020-01-01')"),
    ("analyst", "select 1; drop table orders"),
    ("analyst", "copy orders to 'out.csv'"),
    ("analyst", "attach 'other.db' as other"),
    ("support_west", "select * from employees"),
    ("support_west", "select e.full_name from stores s join employees e on s.store_id = e.store_id"),
    ("support_west", "select credit_limit from customers"),
    ("support_west", "select avg(credit_limit) from customers"),
    ("support_west", "with c as (select * from customers) select count(*) from c"),
    ("nobody", "select 1"),
]

ALLOW = [
    ("analyst", "select region, count(*) as n from customers group by region"),
    ("analyst", "select full_name from customers limit 5"),
    ("analyst", "select c.full_name as name, c.region from customers c limit 5"),
    ("analyst", "select job_title, count(*) from employees group by job_title"),
    ("analyst", "with r as (select customer_id, region from customers) select region, count(*) from r group by region"),
    ("analyst", "select count(*) from orders o join customers c on o.customer_id = c.customer_id where c.region = 'EAST'"),
    ("support_west", "select region, count(*) as n from customers group by region"),
    ("support_west", "select count(*) from orders"),
    ("support_west", "select s.region, count(*) from orders o join stores s on o.store_id = s.store_id group by s.region"),
    ("support_west", "select email from customers limit 3"),
    ("finance", "select email, phone, credit_limit from customers limit 3"),
    ("finance", "select avg(salary) from employees"),
]


@pytest.mark.parametrize("role,sql", DENY)
def test_denied(policies, role, sql):
    with pytest.raises(PolicyViolation):
        policies.govern(sql, role)


@pytest.mark.parametrize("role,sql", ALLOW)
def test_allowed_queries_run(policies, executor, role, sql):
    g = policies.govern(sql, role)
    executor.run(g.sql)  # must execute


def test_masked_columns_are_hashed(policies, executor):
    g = policies.govern("select c.full_name as name, c.region from customers c limit 10", "analyst")
    assert g.masked_columns == ["c.full_name"]
    rows = executor.run(g.sql).rows
    assert all(re.fullmatch(r"[0-9a-f]{32}", name) for name, _ in rows)


def test_row_filters_restrict_results(policies, executor):
    g = policies.govern("select region, count(*) from customers group by region", "support_west")
    assert {r[0] for r in executor.run(g.sql).rows} == {"WEST"}
    g = policies.govern("select s.region, count(*) from orders o join stores s on o.store_id = s.store_id "
                        "group by s.region", "support_west")
    assert {r[0] for r in executor.run(g.sql).rows} == {"WEST"}
    # Trying to look at another region returns nothing.
    g = policies.govern("select count(*) from stores where region = 'EAST'", "support_west")
    assert executor.run(g.sql).rows == [(0,)]
    assert set(g.row_filters) == {"stores"}


def test_row_filter_applies_inside_ctes_and_subqueries(policies, executor):
    sql = ("with big as (select customer_id, region from customers) "
           "select region, count(*) from (select * from big) t group by region")
    g = policies.govern(sql, "support_west")
    assert {r[0] for r in executor.run(g.sql).rows} == {"WEST"}


def test_lineage(policies):
    g = policies.govern("select c.region, sum(oi.quantity) from order_items oi join orders o on oi.order_id = o.order_id "
                        "join customers c on o.customer_id = c.customer_id group by c.region", "analyst")
    assert g.tables == ["customers", "order_items", "orders"]
    assert g.columns == ["customers.customer_id", "customers.region", "order_items.order_id", "order_items.quantity",
                         "orders.customer_id", "orders.order_id"]


def test_invalid_queries_are_query_errors(policies):
    with pytest.raises(QueryError):
        policies.govern("select nope from customers", "analyst")
    with pytest.raises(QueryError):
        policies.govern("select * from no_such_table", "analyst")
    with pytest.raises(QueryError):
        policies.govern("selec 1", "analyst")


def test_executor_blocks_file_access(executor):
    from querypilot.executor import ExecutionError

    with pytest.raises(ExecutionError):
        executor.run("select * from read_csv('/etc/passwd')")
