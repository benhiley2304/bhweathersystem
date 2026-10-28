from datetime import date, timedelta
from bhws_core.playbook import evaluate, Zone, find_events

D0 = date(2023, 1, 3)


def dates(n):
    return [D0 + timedelta(weeks=i) for i in range(n)]


def base(n=160):
    # managers oscillate mid-range, price flat at 100
    net = [((i * 37) % 200) - 100 for i in range(n - 14)] + [0] * 14
    px = [100.0] * n
    return net, px


def test_no_extreme():
    net, px = base()
    net[-1] = 0
    r = evaluate(dates(len(net)), net, px)
    assert r["status"] == "No extreme"


def test_extreme_short_crowd_not_turned_and_no_zone_means_no_level():
    net, px = base()
    net += [-500, -520]            # crowd crams short: fade = long
    px += [95.0, 94.0]
    r = evaluate(dates(len(net)), net, px)
    assert r["fade_side"] == "long"
    assert r["status"] == "Not turned"
    assert r["starter"]["state"] == "Wait for the zone"
    assert r["starter"]["stop"] is None


def test_turn_two_reports_and_close_through_sma_is_add_signal():
    net, px = base()
    net += [-500, -480, -450]      # extreme then two reports of covering
    px += [95.0, 99.0, 103.0]      # last close above 10w average (~100)
    r = evaluate(dates(len(net)), net, px, book_side=1)
    assert r["managers_turning_reports"] == 2
    assert r["close_through_10w"] is True
    assert r["status"] == "Add eligible"


def test_turn_without_price_is_one_more_week():
    net, px = base()
    net += [-500, -480, -450]
    px += [95.0, 94.0, 93.0]
    r = evaluate(dates(len(net)), net, px)
    assert r["status"] == "One more week"


def test_expiry_after_six_reports():
    net, px = base()
    net += [-500] + [-510 - i for i in range(7)]
    px += [95.0] * 8
    r = evaluate(dates(len(net)), net, px)
    assert r["status"] == "Past test window"      # no starter date: not "Expired"
    d = dates(len(net))
    r = evaluate(d, net, px, starter_date=d[-8])
    assert r["status"] == "Expired" and r["clock"] == "from starter"
    r = evaluate(d, net, px, starter_date=d[-3])
    assert r["status"] == "Not turned"


def test_with_the_crowd():
    net, px = base()
    net += [-500]
    px += [95.0]
    r = evaluate(dates(len(net)), net, px, book_side=-1)
    assert r["status"] == "With the crowd"


def test_zone_stop_rule_long():
    net, px = base()
    net += [-500]
    px += [101.0]
    z = Zone(proximal=102.0, distal=100.0)
    r = evaluate(dates(len(net)), net, px, zone=z)
    assert r["starter"]["state"] == "Starter ready"
    assert r["starter"]["stop"] == 99.5   # 25% of a 2.0 deep zone beyond distal


def test_misaligned_inputs_refuse():
    r = evaluate(dates(10), [1] * 10, [1.0] * 9)
    assert r["status"] == "Insufficient data"


def test_events_rearm_after_13_reports():
    p = [50] * 5 + [5, 50, 5] + [50] * 20 + [5]
    ev = find_events(p)
    assert [i for i, _ in ev] == [5, 28]


def test_left_extreme_after_window_is_no_extreme():
    net, px = base()
    net += [-500] + [-510 - i for i in range(3)] + [0, 10, 20, 30]
    px += [95.0] * 8
    r = evaluate(dates(len(net)), net, px)
    assert 10 < r["managers_pct"] < 90
    assert r["status"] == "No extreme"
